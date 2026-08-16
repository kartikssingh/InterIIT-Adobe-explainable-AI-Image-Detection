"""aidetect — explainable detection of AI-generated images.

Three stages, one package:

1. **Detect** — an adversarially fine-tuned DINOv2 ViT-B/14 classifier says
   REAL or FAKE, and GradCAM/rollout/occlusion says *where*.
2. **Classify artifacts** — SigLIP scores the suspicious crop against the 70
   official artifact descriptors, grouped into eight semantic families.
3. **Explain** — a VLM (or the deterministic rule-based writer) turns that
   evidence into a short natural-language explanation.

Quick start::

    from aidetect import AnalysisPipeline, AppConfig

    config = AppConfig()
    config.describe.backend = "rule_based"

    with AnalysisPipeline(config) as pipeline:
        result = pipeline.analyze("image.jpg")

    print(result.detection.prediction, result.description)

Submodules import torch lazily, so ``import aidetect`` stays cheap.
"""

from __future__ import annotations

from typing import Any

from .config import AppConfig, ArtifactConfig, DescribeConfig, DetectorConfig, ExplainConfig, RuntimeConfig
from .exceptions import (
    AidetectError,
    BackendUnavailableError,
    CheckpointError,
    ConfigError,
    ExplainabilityError,
    ImageLoadError,
    PipelineError,
)
from .logging_utils import configure_logging, get_logger
from .types import (
    AnalysisResult,
    ArtifactMatch,
    ArtifactReport,
    BatchSummary,
    DetectionResult,
    Explanation,
    Region,
)
from .version import __version__

__all__ = [
    "__version__",
    # config
    "AppConfig",
    "ArtifactConfig",
    "DescribeConfig",
    "DetectorConfig",
    "ExplainConfig",
    "RuntimeConfig",
    # results
    "AnalysisResult",
    "ArtifactMatch",
    "ArtifactReport",
    "BatchSummary",
    "DetectionResult",
    "Explanation",
    "Region",
    # errors
    "AidetectError",
    "BackendUnavailableError",
    "CheckpointError",
    "ConfigError",
    "ExplainabilityError",
    "ImageLoadError",
    "PipelineError",
    # logging
    "configure_logging",
    "get_logger",
    # lazily exported
    "AnalysisPipeline",
    "Detector",
    "ArtifactClassifier",
    "DescriptionGenerator",
    "analyze_image",
    "run_batch",
]

_LAZY = {
    "AnalysisPipeline": ("aidetect.pipeline", "AnalysisPipeline"),
    "analyze_image": ("aidetect.pipeline", "analyze_image"),
    "Detector": ("aidetect.core.detector", "Detector"),
    "ArtifactClassifier": ("aidetect.artifacts.classifier", "ArtifactClassifier"),
    "DescriptionGenerator": ("aidetect.describe.generator", "DescriptionGenerator"),
    "run_batch": ("aidetect.batch", "run_batch"),
}


def __getattr__(name: str) -> Any:
    """Import heavy submodules only when their symbols are actually used."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
