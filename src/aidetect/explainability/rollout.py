"""Attention rollout saliency (Abnar & Zuidema, 2020).

Rollout multiplies the (residual-augmented) attention matrices of every block to
estimate how much each input patch contributes to the final [CLS] token. It is
gradient-free and class-agnostic, which makes it a useful sanity check next to
GradCAM: agreement between the two is strong evidence the highlighted region is
genuinely what the network keys on.

timm's attention modules use fused scaled-dot-product attention and therefore
never materialise the attention matrix. Rather than monkey-patching the model,
we hook each block's ``qkv`` projection and recompute the attention weights
ourselves — cheap, and it works with every attention implementation.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch

from ..exceptions import ExplainabilityError
from ..logging_utils import get_logger
from .gradcam import _token_grid, normalize_map

logger = get_logger(__name__)


class AttentionRollout:
    """Compute an attention-rollout saliency map for a timm ViT."""

    def __init__(
        self,
        model: torch.nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
    ) -> None:
        if head_fusion not in ("mean", "max", "min"):
            raise ExplainabilityError("head_fusion must be 'mean', 'max' or 'min'")
        if not 0.0 <= discard_ratio < 1.0:
            raise ExplainabilityError("discard_ratio must be within [0, 1)")
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    # -- attention capture -------------------------------------------------- #

    def _attention_modules(self) -> List[torch.nn.Module]:
        backbone = getattr(self.model, "backbone", self.model)
        blocks = getattr(backbone, "blocks", None)
        if not blocks:
            raise ExplainabilityError("Model has no transformer blocks to roll out")
        modules = [getattr(block, "attn", None) for block in blocks]
        if any(m is None for m in modules):
            raise ExplainabilityError("Transformer blocks expose no `.attn` submodule")
        return modules  # type: ignore[return-value]

    @staticmethod
    def _attention_from_qkv(attn_module: torch.nn.Module, qkv: torch.Tensor) -> torch.Tensor:
        """Recompute softmax attention weights from a fused qkv projection."""
        num_heads = int(getattr(attn_module, "num_heads", 0))
        if num_heads <= 0:
            raise ExplainabilityError("Attention module does not expose `num_heads`")

        batch, tokens, triple_dim = qkv.shape
        dim = triple_dim // 3
        head_dim = dim // num_heads
        scale = float(getattr(attn_module, "scale", head_dim**-0.5))

        qkv = qkv.reshape(batch, tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key = qkv[0], qkv[1]

        # timm optionally applies per-head norms before attention.
        for name, tensor_ref in (("q_norm", "query"), ("k_norm", "key")):
            norm = getattr(attn_module, name, None)
            if norm is not None and not isinstance(norm, torch.nn.Identity):
                if tensor_ref == "query":
                    query = norm(query)
                else:
                    key = norm(key)

        logits = (query.float() @ key.float().transpose(-2, -1)) * scale
        return logits.softmax(dim=-1)  # (B, heads, N, N)

    # -- main entry point --------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        tensor: torch.Tensor,
        device: str | torch.device = "cpu",
        num_prefix_tokens: Optional[int] = None,
        output_size: Optional[int] = None,
        **_ignored,
    ) -> np.ndarray:
        attentions: List[torch.Tensor] = []
        modules = self._attention_modules()
        handles = []

        def make_hook(module: torch.nn.Module):
            def hook(_mod, _inputs, output):
                attentions.append(self._attention_from_qkv(module, output).detach())

            return hook

        for module in modules:
            qkv = getattr(module, "qkv", None)
            if qkv is None:
                raise ExplainabilityError(
                    "Attention module has no fused `qkv` projection; "
                    "attention rollout is unsupported for this backbone."
                )
            handles.append(qkv.register_forward_hook(make_hook(module)))

        try:
            self.model(tensor.to(device))
        finally:
            for handle in handles:
                handle.remove()

        if not attentions:
            raise ExplainabilityError("No attention maps were captured")

        rollout = self._rollout(attentions)
        prefix = self._prefix_tokens(rollout.shape[-1], num_prefix_tokens)

        # Row 0 = [CLS] attention over every token; keep the spatial part.
        mask = rollout[0, prefix:]
        grid = _token_grid(mask.numel())
        cam = mask.reshape(1, 1, *grid).float()

        size = output_size or tensor.shape[-1]
        cam = torch.nn.functional.interpolate(
            cam, size=(size, size), mode="bilinear", align_corners=False
        )
        return normalize_map(cam.squeeze().cpu().numpy())

    # -- internals ---------------------------------------------------------- #

    def _rollout(self, attentions: List[torch.Tensor]) -> torch.Tensor:
        result: Optional[torch.Tensor] = None

        for attention in attentions:
            if self.head_fusion == "mean":
                fused = attention.mean(dim=1)
            elif self.head_fusion == "max":
                fused = attention.max(dim=1).values
            else:
                fused = attention.min(dim=1).values
            fused = fused[0]  # first item of the batch -> (N, N)

            if self.discard_ratio > 0:
                fused = self._discard_lowest(fused, self.discard_ratio)

            identity = torch.eye(fused.shape[-1], device=fused.device, dtype=fused.dtype)
            fused = fused + identity                       # residual connection
            fused = fused / fused.sum(dim=-1, keepdim=True)

            result = fused if result is None else fused @ result

        assert result is not None
        return result

    @staticmethod
    def _discard_lowest(attention: torch.Tensor, ratio: float) -> torch.Tensor:
        """Zero out the weakest connections, which are mostly noise."""
        tokens = attention.shape[-1]
        keep = max(1, int(tokens * (1.0 - ratio)))
        threshold = attention.topk(keep, dim=-1).values[..., -1:]
        return torch.where(attention >= threshold, attention, torch.zeros_like(attention))

    def _prefix_tokens(self, num_tokens: int, override: Optional[int]) -> int:
        if override is not None:
            return int(override)
        backbone = getattr(self.model, "backbone", self.model)
        prefix = int(getattr(backbone, "num_prefix_tokens", 1))
        spatial = num_tokens - prefix
        root = int(math.isqrt(spatial)) if spatial > 0 else 0
        if root * root != spatial:
            for candidate in range(0, min(9, num_tokens)):
                remainder = num_tokens - candidate
                root = int(math.isqrt(remainder))
                if root * root == remainder:
                    return candidate
        return prefix
