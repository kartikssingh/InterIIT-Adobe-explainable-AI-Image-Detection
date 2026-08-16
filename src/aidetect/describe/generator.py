"""Stage 3 façade: prompt -> backend -> polished explanation."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from ..config import AppConfig, DescribeConfig
from ..exceptions import BackendUnavailableError
from ..logging_utils import get_logger
from ..types import ArtifactReport, DetectionResult, Explanation, Region
from . import postprocess, prompts
from .base import DescriptionBackend, available_backends, build_backend
from .rule_based import RuleBasedBackend, compose_description

logger = get_logger(__name__)


class DescriptionGenerator:
    """Generate natural-language explanations with a configurable backend.

    The backend is created lazily, so constructing the generator never touches
    the network or the GPU. Any backend failure degrades to the rule-based
    writer when ``describe.fallback_to_rules`` is set (the default), which means
    the pipeline is never left without a description.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        backend: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or AppConfig()
        self.cfg: DescribeConfig = self.config.describe
        self.backend_name = backend or self.cfg.backend
        self.device = device
        self._backend: Optional[DescriptionBackend] = None

    # -- lifecycle --------------------------------------------------------- #

    @property
    def backend(self) -> DescriptionBackend:
        if self._backend is None:
            self._backend = build_backend(self.backend_name, self.config, self.device)
        return self._backend

    def load(self) -> "DescriptionGenerator":
        self.backend.load()
        return self

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
        self._backend = None

    def __enter__(self) -> "DescriptionGenerator":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.unload()

    # -- generation -------------------------------------------------------- #

    def generate(
        self,
        image=None,
        report: Optional[ArtifactReport] = None,
        detection: Optional[DetectionResult] = None,
        regions: Sequence[Region] = (),
    ) -> Explanation:
        """Produce an :class:`Explanation` for one crop."""
        started = time.perf_counter()
        prompt = prompts.build_prompt_for_backend(
            self.backend_name,
            report=report,
            detection=detection,
            regions=regions,
            max_words=self.cfg.max_words,
            max_sentences=self.cfg.max_sentences,
        )

        fallback_used = False
        raw_text = ""
        try:
            raw_text = self.backend.generate(
                image,
                prompt,
                report=report,
                detection=detection,
                regions=regions,
                max_new_tokens=self.cfg.max_new_tokens,
            )
            if not raw_text.strip():
                raise ValueError("backend returned empty text")
        except (BackendUnavailableError, RuntimeError, ValueError, OSError) as exc:
            if not self.cfg.fallback_to_rules:
                raise
            logger.warning(
                "Backend '%s' failed (%s) — falling back to the rule-based writer",
                self.backend_name,
                exc,
            )
            fallback_used = True
            raw_text = compose_description(report=report, detection=detection, regions=regions)

        text, truncated = postprocess.polish(
            raw_text, max_words=self.cfg.max_words, max_sentences=self.cfg.max_sentences
        )

        # A backend that returns something unusably short is worse than a
        # deterministic sentence, so treat that as a failure too.
        if postprocess.word_count(text) < 5 and not fallback_used:
            logger.warning("Backend '%s' produced a degenerate answer — using rules", self.backend_name)
            fallback_used = True
            text, truncated = postprocess.polish(
                compose_description(report=report, detection=detection, regions=regions),
                max_words=self.cfg.max_words,
                max_sentences=self.cfg.max_sentences,
            )

        return Explanation(
            text=text,
            backend="rule_based" if fallback_used else self.backend_name,
            prompt=prompt,
            word_count=postprocess.word_count(text),
            truncated=truncated,
            fallback_used=fallback_used,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

    def per_artifact_explanations(
        self,
        image=None,
        report: Optional[ArtifactReport] = None,
        detection: Optional[DetectionResult] = None,
        regions: Sequence[Region] = (),
        limit: int = 3,
    ) -> Dict[str, str]:
        """One explanation per top artifact — the Task-2 submission shape.

        Visual backends are asked a targeted question per artifact; the
        rule-based writer composes its own sentences.
        """
        if not report or not report.matches:
            return {}

        explanations: Dict[str, str] = {}
        matches = report.matches[:limit]

        if not self.backend.is_visual or image is None:
            rules = RuleBasedBackend(self.config, self.device)
            sentences = rules.per_artifact(report, detection, regions, limit=limit)
            for match, sentence in zip(matches, sentences):
                text, _ = postprocess.polish(
                    sentence, max_words=self.cfg.max_words, max_sentences=self.cfg.max_sentences
                )
                explanations[match.descriptor] = text
            return explanations

        for match in matches:
            question = prompts.artifact_question(match.descriptor)
            try:
                raw = self.backend.generate(image, question, max_new_tokens=self.cfg.max_new_tokens)
            except Exception as exc:  # noqa: BLE001 - any backend failure falls back
                logger.warning("Per-artifact generation failed for %s: %s", match.descriptor, exc)
                raw = ""
            if not raw.strip():
                raw = RuleBasedBackend(self.config, self.device).per_artifact(
                    ArtifactReport(matches=[match]), detection, regions, limit=1
                )[0]
            text, _ = postprocess.polish(
                raw, max_words=self.cfg.max_words, max_sentences=self.cfg.max_sentences
            )
            explanations[match.descriptor] = text

        return explanations

    # -- diagnostics ------------------------------------------------------- #

    def describe(self) -> Dict[str, Any]:
        return {
            "requested_backend": self.backend_name,
            "available_backends": available_backends(),
            "max_words": self.cfg.max_words,
            "max_sentences": self.cfg.max_sentences,
            "fallback_to_rules": self.cfg.fallback_to_rules,
        }
