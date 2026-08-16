"""Saliency methods and visualisation for the detector."""

from __future__ import annotations

from typing import Any

from ..exceptions import ExplainabilityError

__all__ = [
    "ViTGradCAM",
    "AttentionRollout",
    "OcclusionSensitivity",
    "build_saliency",
    "SALIENCY_METHODS",
]

SALIENCY_METHODS = ("gradcam", "gradcam++", "rollout", "occlusion")


def build_saliency(method: str, model: Any, **kwargs: Any):
    """Factory returning a saliency object with a common ``.generate()`` API."""
    name = (method or "gradcam").lower()

    if name in ("gradcam", "gradcam++"):
        from .gradcam import ViTGradCAM

        return ViTGradCAM(model, variant=name)
    if name == "rollout":
        from .rollout import AttentionRollout

        return AttentionRollout(
            model,
            head_fusion=kwargs.get("head_fusion", "mean"),
            discard_ratio=kwargs.get("discard_ratio", 0.9),
        )
    if name == "occlusion":
        from .occlusion import OcclusionSensitivity

        return OcclusionSensitivity(
            model,
            patch_size=kwargs.get("patch_size", 64),
            stride=kwargs.get("stride", 32),
            batch_size=kwargs.get("batch_size", 8),
        )

    raise ExplainabilityError(
        f"Unknown saliency method {method!r}. Available: {', '.join(SALIENCY_METHODS)}"
    )


def __getattr__(name: str) -> Any:  # lazy re-exports keep torch out of import time
    if name == "ViTGradCAM":
        from .gradcam import ViTGradCAM

        return ViTGradCAM
    if name == "AttentionRollout":
        from .rollout import AttentionRollout

        return AttentionRollout
    if name == "OcclusionSensitivity":
        from .occlusion import OcclusionSensitivity

        return OcclusionSensitivity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
