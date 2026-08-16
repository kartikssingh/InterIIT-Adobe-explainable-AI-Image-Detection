"""Turn a saliency map into ranked, croppable regions.

The original pipeline took the bounding box of *all* hot pixels, so two artifacts
on opposite sides of an image produced one giant box covering everything between
them. Here the hot mask is split into connected components first, and each
component becomes its own ranked region.

Connected-component labelling runs on a coarse grid (64x64 by default) rather
than at full resolution, which keeps the pure-numpy/BFS implementation fast
without needing scipy or OpenCV.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..types import BBox, Region

Grid = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Mask construction
# --------------------------------------------------------------------------- #


def hot_mask(
    cam: np.ndarray,
    threshold: float = 0.5,
    fallback_percentile: float = 90.0,
) -> np.ndarray:
    """Binarise a saliency map, backing off to a percentile when needed."""
    cam = np.asarray(cam, dtype=np.float32)
    mask = cam > threshold
    if mask.any():
        return mask

    cutoff = float(np.percentile(cam, fallback_percentile))
    mask = cam > cutoff
    if mask.any():
        return mask

    # Degenerate map (all equal): keep only the single hottest pixel.
    mask = np.zeros_like(cam, dtype=bool)
    flat = int(np.argmax(cam))
    mask[flat // cam.shape[1], flat % cam.shape[1]] = True
    return mask


def downsample_mask(mask: np.ndarray, grid: int) -> np.ndarray:
    """Max-pool a boolean mask down to at most ``grid x grid`` cells."""
    height, width = mask.shape
    if max(height, width) <= grid:
        return mask

    ys = np.linspace(0, height, min(grid, height) + 1).astype(int)
    xs = np.linspace(0, width, min(grid, width) + 1).astype(int)
    coarse = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
    for row in range(len(ys) - 1):
        for col in range(len(xs) - 1):
            coarse[row, col] = mask[ys[row] : ys[row + 1], xs[col] : xs[col + 1]].any()
    return coarse


def label_components(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Label connected ``True`` cells. Returns ``(labels, count)``.

    Iterative BFS — no recursion limits, no scipy dependency.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    neighbours = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 4
        else [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    )

    current = 0
    for start_y, start_x in zip(*np.nonzero(mask)):
        if labels[start_y, start_x]:
            continue
        current += 1
        queue = deque([(int(start_y), int(start_x))])
        labels[start_y, start_x] = current
        while queue:
            y, x = queue.popleft()
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    queue.append((ny, nx))

    return labels, current


# --------------------------------------------------------------------------- #
# Region extraction
# --------------------------------------------------------------------------- #


def find_regions(
    cam: np.ndarray,
    image_size: Tuple[int, int],
    threshold: float = 0.5,
    fallback_percentile: float = 90.0,
    max_regions: int = 3,
    min_area_fraction: float = 0.002,
    padding: int = 20,
    component_grid: int = 64,
) -> List[Region]:
    """Extract ranked :class:`Region` objects from a saliency map.

    Args:
        cam: ``[0, 1]`` saliency map of any resolution.
        image_size: ``(width, height)`` of the image the boxes must index into.
        threshold: relative activation above which a pixel is "hot".
        max_regions: keep at most this many components, strongest first.
        min_area_fraction: discard specks smaller than this fraction of the map.
        padding: pixels added around each box, in image coordinates.
    """
    cam = np.asarray(cam, dtype=np.float32)
    cam_h, cam_w = cam.shape
    img_w, img_h = image_size

    mask = hot_mask(cam, threshold, fallback_percentile)
    coarse = downsample_mask(mask, component_grid)
    labels, count = label_components(coarse)
    if count == 0:
        return []

    scale_y = cam_h / coarse.shape[0]
    scale_x = cam_w / coarse.shape[1]
    total_cells = coarse.size

    candidates: List[Tuple[float, float, BBox, Tuple[int, int]]] = []
    for label in range(1, count + 1):
        cells = labels == label
        area_fraction = float(cells.sum()) / float(total_cells)
        if area_fraction < min_area_fraction and count > 1:
            continue

        rows, cols = np.nonzero(cells)
        y1 = int(np.floor(rows.min() * scale_y))
        y2 = int(np.ceil((rows.max() + 1) * scale_y))
        x1 = int(np.floor(cols.min() * scale_x))
        x2 = int(np.ceil((cols.max() + 1) * scale_x))

        window = cam[y1:y2, x1:x2]
        if window.size == 0:
            continue
        score = float(window.max())

        peak_flat = int(np.argmax(window))
        peak_y = y1 + peak_flat // max(window.shape[1], 1)
        peak_x = x1 + peak_flat % max(window.shape[1], 1)

        bbox = _to_image_coords((x1, y1, x2, y2), (cam_w, cam_h), (img_w, img_h), padding)
        peak_img = (
            int(peak_x * img_w / cam_w),
            int(peak_y * img_h / cam_h),
        )
        candidates.append((score, area_fraction, bbox, peak_img))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    regions: List[Region] = []
    for rank, (score, area_fraction, bbox, peak) in enumerate(candidates[:max_regions], start=1):
        regions.append(
            Region(
                bbox=bbox,
                score=score,
                rank=rank,
                area_fraction=round(area_fraction, 6),
                peak_xy=peak,
                peak_pct=(
                    round(peak[0] / img_w * 100.0, 1) if img_w else 0.0,
                    round(peak[1] / img_h * 100.0, 1) if img_h else 0.0,
                ),
            )
        )
    return regions


def _to_image_coords(
    bbox: BBox,
    cam_size: Grid,
    image_size: Grid,
    padding: int,
) -> BBox:
    """Scale a CAM-space box to image space and apply padding + clamping."""
    x1, y1, x2, y2 = bbox
    cam_w, cam_h = cam_size
    img_w, img_h = image_size

    sx = img_w / cam_w
    sy = img_h / cam_h

    nx1 = max(0, int(x1 * sx) - padding)
    ny1 = max(0, int(y1 * sy) - padding)
    nx2 = min(img_w, int(x2 * sx) + padding)
    ny2 = min(img_h, int(y2 * sy) + padding)

    # Guarantee a non-empty box even for a single-pixel component.
    if nx2 <= nx1:
        nx2 = min(img_w, nx1 + 1)
        nx1 = max(0, nx2 - 1)
    if ny2 <= ny1:
        ny2 = min(img_h, ny1 + 1)
        ny1 = max(0, ny2 - 1)

    return (nx1, ny1, nx2, ny2)


def whole_image_region(image_size: Tuple[int, int], score: float = 0.0) -> Region:
    """Fallback region covering the entire image."""
    width, height = image_size
    return Region(
        bbox=(0, 0, width, height),
        score=score,
        rank=1,
        area_fraction=1.0,
        peak_xy=(width // 2, height // 2),
        peak_pct=(50.0, 50.0),
    )


def merge_overlapping(regions: Sequence[Region], iou_threshold: float = 0.6) -> List[Region]:
    """Greedy NMS-style merge so near-duplicate boxes are not reported twice."""
    kept: List[Region] = []
    for region in sorted(regions, key=lambda r: r.score, reverse=True):
        if all(iou(region.bbox, other.bbox) < iou_threshold for other in kept):
            kept.append(region)
    for rank, region in enumerate(kept, start=1):
        region.rank = rank
    return kept


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def saliency_stats(cam: np.ndarray) -> dict:
    """Summary statistics used in reports and the rule-based description."""
    cam = np.asarray(cam, dtype=np.float32)
    flat = int(np.argmax(cam))
    return {
        "mean": float(cam.mean()),
        "max": float(cam.max()),
        "std": float(cam.std()),
        "hot_fraction": float((cam > 0.5).mean()),
        "peak_row_pct": round(flat // cam.shape[1] / cam.shape[0] * 100.0, 1),
        "peak_col_pct": round(flat % cam.shape[1] / cam.shape[1] * 100.0, 1),
        #: Concentration: high values mean one tight hotspot, low means diffuse.
        "concentration": float(cam.max() - cam.mean()),
    }
