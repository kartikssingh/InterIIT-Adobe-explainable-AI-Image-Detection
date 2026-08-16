"""Stage 1 — REAL/FAKE detection with saliency support.

The :class:`Detector` owns the model lifecycle (lazy load, device placement,
precision) and exposes three things the rest of the pipeline needs:

* :meth:`predict` / :meth:`predict_batch` — calibrated probabilities
* :meth:`saliency` — a ``[0, 1]`` heatmap from any supported method
* :meth:`localize` — ranked suspicious regions derived from that heatmap
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import AppConfig, DetectorConfig
from ..device import autocast_enabled, free_memory, resolve_device, resolve_dtype
from ..explainability import build_saliency
from ..io_utils import load_image
from ..logging_utils import get_logger
from ..types import DetectionResult, Region
from . import localization
from .model import CheckpointInfo, DINOv2Classifier, count_parameters, load_model, resolve_checkpoint_path
from .preprocess import preprocess, preprocess_batch, tta_variants

logger = get_logger(__name__)


class Detector:
    """AI-generated image detector (DINOv2 + linear head)."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or AppConfig()
        self.cfg: DetectorConfig = self.config.detector
        self.device = resolve_device(device or self.config.runtime.device)
        self.dtype = resolve_dtype(self.config.runtime.dtype, self.device)
        self.checkpoint_path = resolve_checkpoint_path(self.cfg, checkpoint)
        self._model: Optional[DINOv2Classifier] = None
        self._info: Optional[CheckpointInfo] = None

    # -- lifecycle --------------------------------------------------------- #

    @property
    def model(self) -> DINOv2Classifier:
        if self._model is None:
            self.load()
        assert self._model is not None
        return self._model

    @property
    def checkpoint_info(self) -> CheckpointInfo:
        if self._info is None:
            self.load()
        assert self._info is not None
        return self._info

    def load(self) -> "Detector":
        """Load weights onto the target device (idempotent)."""
        if self._model is not None:
            return self
        logger.info("Loading detector on %s (%s)", self.device, str(self.dtype).replace("torch.", ""))
        model, info = load_model(self.checkpoint_path, self.device, self.cfg)
        self._model = model
        self._info = info
        params = count_parameters(model)
        logger.debug("Detector parameters: %s", params)
        return self

    def unload(self) -> None:
        """Drop the model and free GPU memory."""
        self._model = None
        self._info = None
        free_memory(self.device)

    def __enter__(self) -> "Detector":
        return self.load()

    def __exit__(self, *exc_info: Any) -> None:
        self.unload()

    def warmup(self) -> None:
        """Run one dummy forward pass so the first real call is not slow."""
        dummy = torch.zeros(1, 3, self.cfg.image_size, self.cfg.image_size, device=self.device)
        with torch.no_grad():
            self.model(dummy)

    # -- inference --------------------------------------------------------- #

    def _autocast(self):
        if autocast_enabled(self.device, self.dtype):
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @torch.no_grad()
    def logits(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional hflip TTA, returning float32 logits."""
        batch = batch.to(self.device)
        with self._autocast():
            views = tta_variants(batch, hflip=self.cfg.tta_hflip)
            outputs = [self.model(view).float() for view in views]
        return torch.stack(outputs, dim=0).mean(dim=0)

    def probabilities(self, batch: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled softmax probabilities."""
        scaled = self.logits(batch) / max(self.cfg.temperature, 1e-6)
        return torch.softmax(scaled, dim=1)

    def predict(self, image, image_path: Optional[str] = None) -> DetectionResult:
        """Classify a single PIL image or path."""
        if isinstance(image, (str, Path)):
            image_path = image_path or str(image)
            image = load_image(image)
        start = time.perf_counter()
        tensor = preprocess(image, self.cfg)
        logits = self.logits(tensor)  # single forward pass; probs derived below
        probs = torch.softmax(logits / max(self.cfg.temperature, 1e-6), dim=1)
        return self._build_result(
            probs[0], logits[0], image_path or "<in-memory>", (time.perf_counter() - start) * 1000.0
        )

    def predict_batch(
        self,
        images: Sequence,
        image_paths: Optional[Sequence[str]] = None,
    ) -> List[DetectionResult]:
        """Classify several images in one forward pass per chunk."""
        loaded = [load_image(img) if isinstance(img, (str, Path)) else img for img in images]
        paths = list(image_paths or [
            str(img) if isinstance(img, (str, Path)) else "<in-memory>" for img in images
        ])

        results: List[DetectionResult] = []
        batch_size = max(1, self.cfg.batch_size)
        for start_index in range(0, len(loaded), batch_size):
            chunk = loaded[start_index : start_index + batch_size]
            started = time.perf_counter()
            tensor = preprocess_batch(chunk, self.cfg)
            logits = self.logits(tensor)
            probs = torch.softmax(logits / max(self.cfg.temperature, 1e-6), dim=1)
            elapsed = (time.perf_counter() - started) * 1000.0 / max(len(chunk), 1)
            for offset in range(len(chunk)):
                results.append(
                    self._build_result(
                        probs[offset], logits[offset], paths[start_index + offset], elapsed
                    )
                )
        return results

    def _build_result(
        self,
        probs: torch.Tensor,
        logits: torch.Tensor,
        image_path: str,
        latency_ms: float,
    ) -> DetectionResult:
        values = probs.detach().float().cpu().tolist()
        names = list(self.cfg.class_names)
        probabilities = {name: float(values[i]) for i, name in enumerate(names)}

        fake_prob = float(values[self.cfg.fake_index])
        prediction = "FAKE" if fake_prob >= self.cfg.decision_threshold else "REAL"
        confidence = (fake_prob if prediction == "FAKE" else 1.0 - fake_prob) * 100.0

        return DetectionResult(
            image_path=image_path,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            threshold=self.cfg.decision_threshold,
            logits=[float(v) for v in logits.detach().float().cpu().tolist()],
            model_path=str(self.checkpoint_path),
            model_type=self.checkpoint_info.kind,
            checkpoint_meta=self.checkpoint_info.to_dict(),
            tta=self.cfg.tta_hflip,
            latency_ms=round(latency_ms, 2),
        )

    # -- explainability ---------------------------------------------------- #

    def saliency(
        self,
        image,
        method: Optional[str] = None,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Compute a ``[0, 1]`` saliency map for the FAKE class by default."""
        if isinstance(image, (str, Path)):
            image = load_image(image)
        explain_cfg = self.config.explain
        method = method or explain_cfg.method
        target = self.cfg.fake_index if target_class is None else target_class

        tensor = preprocess(image, self.cfg).to(self.device)
        engine = build_saliency(
            method,
            self.model,
            patch_size=explain_cfg.occlusion_patch,
            stride=explain_cfg.occlusion_stride,
            batch_size=max(1, self.cfg.batch_size),
        )
        return engine.generate(
            tensor,
            device=self.device,
            target_class=target,
            output_size=self.cfg.image_size,
        )

    def localize(
        self,
        image,
        cam: Optional[np.ndarray] = None,
        method: Optional[str] = None,
    ) -> tuple[np.ndarray, List[Region]]:
        """Return ``(saliency_map, ranked_regions)`` for an image."""
        if isinstance(image, (str, Path)):
            image = load_image(image)
        if cam is None:
            cam = self.saliency(image, method=method)

        explain_cfg = self.config.explain
        regions = localization.find_regions(
            cam,
            image_size=image.size,
            threshold=explain_cfg.hot_threshold,
            fallback_percentile=explain_cfg.fallback_percentile,
            max_regions=explain_cfg.max_regions,
            min_area_fraction=explain_cfg.min_region_fraction,
            padding=explain_cfg.region_padding,
            component_grid=explain_cfg.component_grid,
        )
        regions = localization.merge_overlapping(regions)
        if not regions:
            regions = [localization.whole_image_region(image.size, float(cam.max()))]
        return cam, regions

    # -- diagnostics ------------------------------------------------------- #

    def describe(self) -> Dict[str, Any]:
        """Machine-readable description of the loaded detector."""
        return {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_info": self.checkpoint_info.to_dict(),
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "backbone": self.cfg.backbone,
            "image_size": self.cfg.image_size,
            "threshold": self.cfg.decision_threshold,
            "temperature": self.cfg.temperature,
            "tta_hflip": self.cfg.tta_hflip,
            "parameters": count_parameters(self.model),
        }
