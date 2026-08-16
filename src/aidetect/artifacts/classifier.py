"""Stage 2 — zero-shot artifact classification with SigLIP (or CLIP).

Improvements over the first implementation:

* **Prompt ensembling** — each descriptor is encoded through several phrasings
  and averaged, which is markedly more stable than a single bare label.
* **Correct SigLIP scoring** — SigLIP is trained with a sigmoid objective and
  ships a ``logit_bias`` alongside ``logit_scale``; ignoring the bias (as the
  old code did) shifts every score. It is now applied when present.
* **Score normalisation** — raw sigmoid scores cluster in a narrow band, which
  makes them useless for ranking thresholds. ``score_mode`` exposes minmax,
  softmax and z-score views alongside the raw value (always preserved).
* **Real category aggregation** — scores roll up into the eight semantic
  families from :mod:`aidetect.artifacts.taxonomy`.
* **Disk-cached text embeddings** and batched image encoding.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import AppConfig, ArtifactConfig
from ..device import resolve_device
from ..exceptions import BackendUnavailableError
from ..io_utils import load_image
from ..logging_utils import get_logger
from ..types import ArtifactMatch, ArtifactReport
from . import embedding_cache, taxonomy

logger = get_logger(__name__)


class ArtifactClassifier:
    """Zero-shot classifier scoring an image against the 70 artifact descriptors."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        device: Optional[str] = None,
        descriptors: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or AppConfig()
        self.cfg: ArtifactConfig = self.config.artifacts
        self.device = resolve_device(device or self.config.runtime.device)
        self.descriptors: List[str] = list(descriptors or taxonomy.DESCRIPTORS)
        self.templates = (
            list(taxonomy.PROMPT_TEMPLATES)
            if self.cfg.prompt_ensemble
            else list(taxonomy.SINGLE_TEMPLATE)
        )
        self._model: Any = None
        self._processor: Any = None
        self._text_embeddings: Optional[torch.Tensor] = None

    # -- model lifecycle --------------------------------------------------- #

    def load(self) -> "ArtifactClassifier":
        """Lazily load the vision-language model and its processor."""
        if self._model is not None:
            return self

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise BackendUnavailableError(
                self.cfg.model_name,
                "Install transformers: pip install 'transformers>=4.40'",
            ) from exc

        logger.info("Loading %s on %s", self.cfg.model_name, self.device)
        try:
            self._processor = AutoProcessor.from_pretrained(self.cfg.model_name)
            self._model = AutoModel.from_pretrained(self.cfg.model_name).to(self.device)
        except OSError as exc:
            raise BackendUnavailableError(
                self.cfg.model_name,
                "Model weights are not cached and could not be downloaded. "
                "Pre-download with: "
                f"huggingface-cli download {self.cfg.model_name}",
            ) from exc

        self._model.eval()
        total = sum(p.numel() for p in self._model.parameters())
        logger.debug("%s loaded (%.1fM parameters)", self.cfg.model_name, total / 1e6)
        return self

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._text_embeddings = None

    def __enter__(self) -> "ArtifactClassifier":
        return self.load()

    def __exit__(self, *exc_info: Any) -> None:
        self.unload()

    # -- encoding ---------------------------------------------------------- #

    @torch.no_grad()
    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self._processor(
            text=list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)
        features = self._model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def text_embeddings(self) -> torch.Tensor:
        """``(num_descriptors, dim)`` normalised, prompt-ensembled embeddings."""
        if self._text_embeddings is not None:
            return self._text_embeddings

        key = embedding_cache.cache_key(self.cfg.model_name, self.descriptors, self.templates)
        cached = embedding_cache.load(
            self.cfg.embedding_cache_dir, key, expected_rows=len(self.descriptors)
        )
        if cached is not None:
            self._text_embeddings = torch.from_numpy(cached).to(self.device)
            logger.debug("Reusing cached descriptor embeddings (%s)", key)
            return self._text_embeddings

        self.load()
        prompts: List[str] = []
        for descriptor in self.descriptors:
            prompts.extend(taxonomy.render_prompts(descriptor, self.templates))

        logger.info(
            "Encoding %d descriptors x %d templates", len(self.descriptors), len(self.templates)
        )
        chunks: List[torch.Tensor] = []
        batch_size = max(1, self.cfg.text_batch_size)
        for start in range(0, len(prompts), batch_size):
            chunks.append(self._encode_texts(prompts[start : start + batch_size]))
        matrix = torch.cat(chunks, dim=0)

        # Average the templates of each descriptor, then re-normalise.
        matrix = matrix.reshape(len(self.descriptors), len(self.templates), -1).mean(dim=1)
        matrix = torch.nn.functional.normalize(matrix, dim=-1)

        embedding_cache.save(self.cfg.embedding_cache_dir, key, matrix.cpu().numpy())
        self._text_embeddings = matrix
        return matrix

    @torch.no_grad()
    def _encode_images(self, images: Sequence) -> torch.Tensor:
        inputs = self._processor(images=list(images), return_tensors="pt").to(self.device)
        features = self._model.get_image_features(**inputs)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    # -- scoring ----------------------------------------------------------- #

    @torch.no_grad()
    def _raw_scores(self, images: Sequence) -> np.ndarray:
        """``(num_images, num_descriptors)`` sigmoid similarity scores."""
        self.load()
        image_embeddings = self._encode_images(images)
        text_embeddings = self.text_embeddings().to(image_embeddings.device)

        logit_scale = getattr(self._model, "logit_scale", None)
        scale = float(logit_scale.exp()) if logit_scale is not None else 100.0
        logit_bias = getattr(self._model, "logit_bias", None)
        bias = float(logit_bias) if logit_bias is not None else 0.0

        logits = image_embeddings @ text_embeddings.T * scale + bias
        return torch.sigmoid(logits).cpu().numpy()

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Apply the configured score view to one image's descriptor scores."""
        mode = self.cfg.score_mode
        if mode == "raw":
            return scores
        if mode == "minmax":
            low, high = float(scores.min()), float(scores.max())
            if high - low <= 1e-12:
                return np.zeros_like(scores)
            return (scores - low) / (high - low)
        if mode == "softmax":
            temperature = max(self.cfg.softmax_temperature, 1e-6)
            shifted = (scores - scores.max()) / temperature
            exponentials = np.exp(shifted)
            return exponentials / exponentials.sum()
        if mode == "zscore":
            std = float(scores.std())
            if std <= 1e-12:
                return np.zeros_like(scores)
            return (scores - float(scores.mean())) / std
        return scores

    # -- public API -------------------------------------------------------- #

    def classify(
        self,
        image,
        top_k: Optional[int] = None,
        return_all: bool = False,
    ) -> ArtifactReport:
        """Score one image (PIL or path) against every descriptor."""
        if isinstance(image, str):
            image = load_image(image)

        started = time.perf_counter()
        raw = self._raw_scores([image])[0]
        report = self._build_report(raw, top_k=top_k, return_all=return_all)
        report.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return report

    def classify_batch(
        self,
        images: Sequence,
        top_k: Optional[int] = None,
    ) -> List[ArtifactReport]:
        """Score several images, encoding them in configurable batches."""
        loaded = [load_image(img) if isinstance(img, str) else img for img in images]
        reports: List[ArtifactReport] = []
        batch_size = max(1, self.config.detector.batch_size)
        for start in range(0, len(loaded), batch_size):
            chunk = loaded[start : start + batch_size]
            started = time.perf_counter()
            raw_scores = self._raw_scores(chunk)
            elapsed = (time.perf_counter() - started) * 1000.0 / max(len(chunk), 1)
            for row in raw_scores:
                report = self._build_report(row, top_k=top_k)
                report.latency_ms = round(elapsed, 2)
                reports.append(report)
        return reports

    def _build_report(
        self,
        raw: np.ndarray,
        top_k: Optional[int] = None,
        return_all: bool = False,
    ) -> ArtifactReport:
        normalised = self._normalize(raw)
        order = np.argsort(-normalised)
        limit = len(self.descriptors) if return_all else max(1, top_k or self.cfg.top_k)

        matches: List[ArtifactMatch] = []
        for rank, index in enumerate(order[:limit], start=1):
            descriptor = self.descriptors[int(index)]
            score = float(normalised[int(index)])
            if score < self.cfg.min_score and rank > 1:
                break
            matches.append(
                ArtifactMatch(
                    descriptor=descriptor,
                    score=score,
                    raw_score=float(raw[int(index)]),
                    category=taxonomy.category_of(descriptor),
                    artifact_id=taxonomy.artifact_id(descriptor),
                    rank=rank,
                )
            )

        category_scores = self._aggregate_categories(normalised)
        detected = list(dict.fromkeys(match.category for match in matches))

        return ArtifactReport(
            matches=matches,
            categories=detected,
            category_scores=category_scores,
            model_name=self.cfg.model_name,
            score_mode=self.cfg.score_mode,
        )

    def _aggregate_categories(self, scores: np.ndarray) -> Dict[str, float]:
        """Max score per semantic family, strongest family first."""
        aggregated: Dict[str, float] = {}
        for index, descriptor in enumerate(self.descriptors):
            category = taxonomy.category_of(descriptor)
            score = float(scores[index])
            if score > aggregated.get(category, float("-inf")):
                aggregated[category] = score
        return dict(sorted(aggregated.items(), key=lambda item: item[1], reverse=True))

    def describe(self) -> Dict[str, Any]:
        return {
            "model_name": self.cfg.model_name,
            "device": self.device,
            "descriptors": len(self.descriptors),
            "templates": len(self.templates),
            "prompt_ensemble": self.cfg.prompt_ensemble,
            "score_mode": self.cfg.score_mode,
            "embedding_cache_dir": self.cfg.embedding_cache_dir,
            "loaded": self._model is not None,
        }
