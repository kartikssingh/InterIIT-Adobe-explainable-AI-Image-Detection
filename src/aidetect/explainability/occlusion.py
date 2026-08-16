"""Occlusion-sensitivity saliency.

Slide a grey patch across the image and record how far the FAKE probability
drops. Regions whose removal makes the image look *less* generated are exactly
the regions carrying the artifact evidence.

This method needs no gradients and makes no assumption about the architecture,
so it doubles as a ground-truth check when GradCAM output looks suspicious. It
costs one forward pass per window, which is why windows are evaluated in batches.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..logging_utils import get_logger
from .gradcam import normalize_map

logger = get_logger(__name__)


class OcclusionSensitivity:
    """Sliding-window occlusion saliency."""

    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: int = 64,
        stride: int = 32,
        batch_size: int = 16,
        fill_value: float = 0.0,
    ) -> None:
        self.model = model
        self.patch_size = max(4, int(patch_size))
        self.stride = max(1, int(stride))
        self.batch_size = max(1, int(batch_size))
        self.fill_value = float(fill_value)

    @torch.no_grad()
    def generate(
        self,
        tensor: torch.Tensor,
        device: str | torch.device = "cpu",
        target_class: int = 0,
        output_size: Optional[int] = None,
        **_ignored,
    ) -> np.ndarray:
        image = tensor.detach().to(device)
        _, _, height, width = image.shape

        baseline = torch.softmax(self.model(image), dim=1)[0, target_class].item()

        ys = list(range(0, max(height - self.patch_size, 0) + 1, self.stride))
        xs = list(range(0, max(width - self.patch_size, 0) + 1, self.stride))
        if ys[-1] + self.patch_size < height:
            ys.append(height - self.patch_size)
        if xs[-1] + self.patch_size < width:
            xs.append(width - self.patch_size)

        positions = [(y, x) for y in ys for x in xs]
        logger.debug(
            "Occlusion: %d windows of %dpx (stride %d)", len(positions), self.patch_size, self.stride
        )

        heat = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)

        for start in range(0, len(positions), self.batch_size):
            chunk = positions[start : start + self.batch_size]
            batch = image.repeat(len(chunk), 1, 1, 1).clone()
            for index, (y, x) in enumerate(chunk):
                batch[index, :, y : y + self.patch_size, x : x + self.patch_size] = self.fill_value

            probs = torch.softmax(self.model(batch), dim=1)[:, target_class]
            drops = (baseline - probs).clamp(min=0.0).cpu().numpy()

            for index, (y, x) in enumerate(chunk):
                heat[y : y + self.patch_size, x : x + self.patch_size] += float(drops[index])
                counts[y : y + self.patch_size, x : x + self.patch_size] += 1.0

        heat = np.divide(heat, counts, out=np.zeros_like(heat), where=counts > 0)

        if output_size and output_size != height:
            cam = torch.from_numpy(heat)[None, None]
            cam = torch.nn.functional.interpolate(
                cam, size=(output_size, output_size), mode="bilinear", align_corners=False
            )
            heat = cam.squeeze().numpy()

        return normalize_map(heat)
