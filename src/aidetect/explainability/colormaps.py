"""Dependency-free colormaps.

matplotlib is a heavy import for what amounts to a 256-entry lookup table, so
the handful of colormaps we need are stored as control points and linearly
interpolated with numpy. Output matches matplotlib closely enough for
visualisation while keeping the runtime dependency list at numpy + Pillow.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

RGB = Tuple[int, int, int]

# Control points sampled evenly across [0, 1] for each colormap.
_CONTROL_POINTS: Dict[str, List[RGB]] = {
    "viridis": [
        (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
        (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
        (180, 222, 44), (253, 231, 37),
    ],
    "inferno": [
        (0, 0, 4), (22, 11, 57), (66, 10, 104), (106, 23, 110),
        (147, 38, 103), (188, 55, 84), (221, 81, 58), (243, 120, 25),
        (252, 165, 10), (252, 255, 164),
    ],
    "magma": [
        (0, 0, 4), (20, 14, 54), (59, 15, 112), (100, 26, 128),
        (140, 41, 129), (183, 55, 121), (222, 73, 104), (247, 112, 92),
        (254, 159, 109), (252, 253, 191),
    ],
    "turbo": [
        (48, 18, 59), (62, 73, 184), (69, 130, 236), (46, 183, 209),
        (44, 218, 148), (108, 240, 87), (183, 244, 48), (238, 205, 40),
        (250, 137, 24), (216, 45, 12),
    ],
    "jet": [
        (0, 0, 128), (0, 0, 255), (0, 128, 255), (0, 255, 255),
        (128, 255, 128), (255, 255, 0), (255, 128, 0), (255, 0, 0),
        (200, 0, 0), (128, 0, 0),
    ],
    "hot": [
        (0, 0, 0), (85, 0, 0), (170, 0, 0), (255, 0, 0),
        (255, 85, 0), (255, 170, 0), (255, 255, 0), (255, 255, 85),
        (255, 255, 170), (255, 255, 255),
    ],
    "gray": [(0, 0, 0), (255, 255, 255)],
    "coolwarm": [
        (59, 76, 192), (98, 130, 234), (141, 176, 254), (184, 208, 249),
        (221, 221, 221), (245, 196, 173), (244, 154, 123), (222, 96, 77),
        (180, 40, 42), (146, 0, 26),
    ],
}

AVAILABLE_COLORMAPS: Tuple[str, ...] = tuple(sorted(_CONTROL_POINTS))

_LUT_CACHE: Dict[str, np.ndarray] = {}


def build_lut(name: str, size: int = 256) -> np.ndarray:
    """Return an ``(size, 3)`` uint8 lookup table for ``name``."""
    key = f"{name}:{size}"
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached

    points = _CONTROL_POINTS.get(name.lower())
    if points is None:
        raise KeyError(
            f"Unknown colormap {name!r}. Available: {', '.join(AVAILABLE_COLORMAPS)}"
        )

    stops = np.linspace(0.0, 1.0, len(points))
    positions = np.linspace(0.0, 1.0, size)
    channels = [
        np.interp(positions, stops, [p[channel] for p in points]) for channel in range(3)
    ]
    lut = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
    _LUT_CACHE[key] = lut
    return lut


def apply_colormap(values: np.ndarray, name: str = "turbo", gamma: float = 1.0) -> np.ndarray:
    """Map a ``[0, 1]`` array to an ``(H, W, 3)`` uint8 RGB image.

    ``gamma < 1`` brightens weak activations, which helps when a saliency map is
    dominated by one very hot pixel.
    """
    array = np.asarray(values, dtype=np.float32)
    if gamma != 1.0:
        array = np.power(np.clip(array, 0.0, 1.0), max(gamma, 1e-6))
    indices = np.clip(array, 0.0, 1.0)
    lut = build_lut(name)
    positions = np.rint(indices * (lut.shape[0] - 1)).astype(np.int32)
    return lut[positions]


def colorbar_strip(name: str = "turbo", width: int = 256, height: int = 16) -> np.ndarray:
    """A horizontal colour ramp, used as a legend in reports."""
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    return np.repeat(apply_colormap(ramp, name)[None, :, :], height, axis=0)


def blend(base: np.ndarray, overlay: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    """Alpha-blend two uint8 RGB arrays; ``alpha`` may be a per-pixel map."""
    base_f = base.astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    if isinstance(alpha, np.ndarray):
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    else:
        alpha = float(np.clip(alpha, 0.0, 1.0))
    mixed = base_f * (1.0 - alpha) + overlay_f * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8)


def sequential_palette(count: int, name: str = "turbo") -> List[str]:
    """``count`` evenly-spaced hex colours from a colormap (used by reports)."""
    if count <= 0:
        return []
    values = np.linspace(0.15, 0.9, count, dtype=np.float32)
    colors = apply_colormap(values, name)
    return ["#%02x%02x%02x" % tuple(int(c) for c in color) for color in colors]


def ensure_colormap(name: str, fallback: str = "turbo") -> str:
    """Return ``name`` if known, otherwise ``fallback``."""
    return name if str(name).lower() in _CONTROL_POINTS else fallback
