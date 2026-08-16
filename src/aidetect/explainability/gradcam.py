"""Gradient-based saliency for Vision Transformers.

A ViT has no convolutional feature map, so we treat the *token* sequence of a
transformer block as the feature map: dropping the prefix tokens ([CLS] and any
register tokens) leaves ``h*w`` spatial tokens that reshape into a grid.

Two variants are implemented:

``gradcam``
    weights = mean of gradients over the spatial tokens (Selvaraju et al.).

``gradcam++``
    per-token weights from second/third-order gradient terms (Chattopadhyay
    et al.), which localises multiple simultaneous artifacts noticeably better
    than vanilla GradCAM.

Unlike the original implementation, the grid size is derived from the actual
token count instead of assuming ``image_size / patch_size``, so register-token
variants of DINOv2 and non-square inputs no longer produce a reshape error.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..exceptions import ExplainabilityError
from ..logging_utils import get_logger

logger = get_logger(__name__)


class ViTGradCAM:
    """GradCAM / GradCAM++ for timm-style vision transformers."""

    VARIANTS = ("gradcam", "gradcam++")

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        variant: str = "gradcam",
    ) -> None:
        if variant not in self.VARIANTS:
            raise ExplainabilityError(
                f"variant must be one of {self.VARIANTS}, got {variant!r}"
            )
        self.model = model
        self.variant = variant
        self.target_layer = target_layer or self._default_target_layer(model)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- hook plumbing ---------------------------------------------------- #

    @staticmethod
    def _default_target_layer(model: torch.nn.Module) -> torch.nn.Module:
        """Pick a layer whose *spatial* tokens actually receive gradient.

        This matters more than it looks. The model pools with ``global_pool=
        "token"``, i.e. only the [CLS] token reaches the head. Hooking the
        **output of the last block** — the obvious choice, and what the previous
        implementation did — therefore yields exactly zero gradient on every
        spatial token, because nothing downstream consumes them. The resulting
        heatmap is uniformly zero and the "hottest region" degenerates to the
        top-left corner.

        Hooking ``blocks[-1].norm1`` (the last block's input LayerNorm) keeps the
        spatial tokens upstream of the final attention mix into [CLS], so the
        gradients are meaningful. This is the same target the reference
        pytorch-grad-cam recipes use for ViTs.
        """
        backbone = getattr(model, "backbone", model)
        blocks = getattr(backbone, "blocks", None)
        if not blocks:
            raise ExplainabilityError(
                "Could not locate transformer blocks on the model; "
                "pass target_layer explicitly."
            )

        last = blocks[-1]
        norm1 = getattr(last, "norm1", None)
        if norm1 is not None and not isinstance(norm1, torch.nn.Identity):
            return norm1
        if len(blocks) >= 2:
            logger.debug("No norm1 on the last block — falling back to blocks[-2]")
            return blocks[-2]
        return last

    def _register(self) -> None:
        def forward_hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self.activations = tensor

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- main entry point -------------------------------------------------- #

    def generate(
        self,
        tensor: torch.Tensor,
        device: str | torch.device = "cpu",
        target_class: int = 0,
        num_prefix_tokens: Optional[int] = None,
        output_size: Optional[int] = None,
    ) -> np.ndarray:
        """Return a ``(H, W)`` heatmap normalised to ``[0, 1]``.

        ``target_class`` is the logit index the saliency explains — normally the
        FAKE class, so the map answers "which pixels made this look generated?".
        """
        if tensor.dim() != 4:
            raise ExplainabilityError(f"Expected a (B, 3, H, W) tensor, got {tuple(tensor.shape)}")

        self.model.zero_grad(set_to_none=True)
        self._register()
        try:
            inputs = tensor.detach().to(device).requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(inputs)
                if target_class >= logits.shape[1]:
                    raise ExplainabilityError(
                        f"target_class={target_class} out of range for {logits.shape[1]} classes"
                    )
                score = logits[:, target_class].sum()
                score.backward()
        finally:
            self._remove()

        if self.activations is None or self.gradients is None:
            raise ExplainabilityError(
                "Saliency hooks captured nothing — the target layer never ran."
            )

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        if activations.dim() != 3:  # (B, tokens, dim)
            raise ExplainabilityError(
                f"Expected token activations of shape (B, N, D), got {tuple(activations.shape)}"
            )

        prefix = self._resolve_prefix_tokens(activations.shape[1], num_prefix_tokens)
        spatial_acts = activations[0, prefix:, :]
        spatial_grads = gradients[0, prefix:, :]

        cam_tokens = (
            self._gradcam_tokens(spatial_acts, spatial_grads)
            if self.variant == "gradcam"
            else self._gradcam_pp_tokens(spatial_acts, spatial_grads)
        )
        cam_tokens = F.relu(cam_tokens)

        grid = _token_grid(cam_tokens.numel())
        cam = cam_tokens.reshape(1, 1, *grid).float()

        size = output_size or tensor.shape[-1]
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
        return normalize_map(cam.squeeze().detach().cpu().numpy())

    # -- variants ---------------------------------------------------------- #

    @staticmethod
    def _gradcam_tokens(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
        weights = grads.mean(dim=0)                      # (D,)
        return (weights.unsqueeze(0) * acts).sum(dim=-1)  # (N,)

    @staticmethod
    def _gradcam_pp_tokens(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
        grads2 = grads.pow(2)
        grads3 = grads2 * grads
        sum_acts = acts.sum(dim=0, keepdim=True)                 # (1, D)
        denominator = 2.0 * grads2 + sum_acts * grads3
        denominator = torch.where(
            denominator.abs() > 1e-8, denominator, torch.full_like(denominator, 1e-8)
        )
        alpha = grads2 / denominator                              # (N, D)
        weights = (alpha * F.relu(grads)).sum(dim=0)              # (D,)
        return (weights.unsqueeze(0) * acts).sum(dim=-1)          # (N,)

    # -- helpers ----------------------------------------------------------- #

    def _resolve_prefix_tokens(self, num_tokens: int, override: Optional[int]) -> int:
        if override is not None:
            return int(override)
        backbone = getattr(self.model, "backbone", self.model)
        prefix = getattr(backbone, "num_prefix_tokens", None)
        if prefix is None:
            prefix = 1  # [CLS] only
        # Sanity check: the remainder must be a perfect square.
        if not _is_square(num_tokens - prefix):
            for candidate in range(0, min(9, num_tokens)):
                if _is_square(num_tokens - candidate):
                    logger.debug("Adjusted prefix token count to %d", candidate)
                    return candidate
        return int(prefix)


def _is_square(value: int) -> bool:
    if value <= 0:
        return False
    root = int(math.isqrt(value))
    return root * root == value


def _token_grid(num_tokens: int) -> tuple[int, int]:
    root = int(math.isqrt(num_tokens))
    if root * root != num_tokens:
        raise ExplainabilityError(
            f"{num_tokens} spatial tokens do not form a square grid; "
            "pass num_prefix_tokens explicitly."
        )
    return root, root


def normalize_map(cam: np.ndarray) -> np.ndarray:
    """Scale an arbitrary saliency map into ``[0, 1]`` (flat maps become zeros)."""
    cam = np.asarray(cam, dtype=np.float32)
    finite = np.isfinite(cam)
    if not finite.all():
        cam = np.where(finite, cam, 0.0)
    low, high = float(cam.min()), float(cam.max())
    if high - low <= 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - low) / (high - low)).astype(np.float32)
