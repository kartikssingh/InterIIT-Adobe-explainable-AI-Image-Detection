"""Image preprocessing for the Stage 1 detector.

The transform mirrors the evaluation transform used during training
(resize to ``image_size`` square, to-tensor, ImageNet normalisation) so that
inference-time statistics match what the model saw while learning.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from ..config import DetectorConfig


def build_transform(cfg: DetectorConfig):
    """Return the deterministic evaluation transform."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(cfg.mean), std=tuple(cfg.std)),
        ]
    )


def preprocess(image, cfg: DetectorConfig) -> torch.Tensor:
    """PIL image -> ``(1, 3, H, W)`` normalised tensor."""
    transform = build_transform(cfg)
    return transform(image).unsqueeze(0)


def preprocess_batch(images: Sequence, cfg: DetectorConfig) -> torch.Tensor:
    """Stack several PIL images into one ``(B, 3, H, W)`` batch."""
    transform = build_transform(cfg)
    return torch.stack([transform(image) for image in images], dim=0)


def denormalize(tensor: torch.Tensor, cfg: DetectorConfig) -> torch.Tensor:
    """Undo normalisation, returning values in ``[0, 1]``."""
    mean = torch.tensor(cfg.mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.std, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor, cfg: DetectorConfig):
    """Convert a normalised ``(1, 3, H, W)`` tensor back into a PIL image."""
    import numpy as np
    from PIL import Image

    array = denormalize(tensor.detach().cpu(), cfg)[0].permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def resized_copy(image, cfg: DetectorConfig):
    """A model-resolution copy of the image, used for overlay rendering."""
    from PIL import Image

    return image.resize((cfg.image_size, cfg.image_size), Image.LANCZOS)


def tta_variants(tensor: torch.Tensor, hflip: bool = True) -> List[torch.Tensor]:
    """Build the test-time-augmentation views of a preprocessed batch."""
    views = [tensor]
    if hflip:
        views.append(torch.flip(tensor, dims=[3]))
    return views


def patch_grid(image_size: int, patch_size: int) -> Tuple[int, int]:
    """Number of patch tokens along (height, width) for a square input."""
    return image_size // patch_size, image_size // patch_size
