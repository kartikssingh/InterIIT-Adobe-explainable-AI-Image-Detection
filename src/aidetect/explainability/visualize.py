"""Rendering helpers for saliency maps, boxes and summary panels.

Only numpy + Pillow are used, so visualisation works on machines without
matplotlib or OpenCV installed.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Region
from .colormaps import apply_colormap, blend, colorbar_strip, ensure_colormap

RGB = Tuple[int, int, int]

FAKE_COLOR: RGB = (229, 72, 77)
REAL_COLOR: RGB = (46, 184, 114)
INK: RGB = (23, 23, 28)
PAPER: RGB = (247, 247, 250)


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #


def _font(size: int = 16):
    """Load a truetype font when available, else Pillow's bitmap default."""
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _text_size(draw, text: str, font) -> Tuple[int, int]:
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:  # pragma: no cover - very old Pillow
        return draw.textsize(text, font=font)


# --------------------------------------------------------------------------- #
# Heatmaps
# --------------------------------------------------------------------------- #


def heatmap_image(cam: np.ndarray, size: Optional[Tuple[int, int]] = None, colormap: str = "turbo", gamma: float = 1.0):
    """Render a ``[0, 1]`` saliency map as a standalone colour image."""
    from PIL import Image

    rgb = apply_colormap(cam, ensure_colormap(colormap), gamma=gamma)
    image = Image.fromarray(rgb)
    if size and image.size != size:
        image = image.resize(size, Image.LANCZOS)
    return image


def resize_cam(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a saliency map to ``(width, height)`` keeping the ``[0, 1]`` range."""
    from PIL import Image

    array = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    resized = Image.fromarray((array * 255.0).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def overlay_heatmap(
    image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "turbo",
    gamma: float = 1.0,
    intensity_alpha: bool = True,
):
    """Blend a saliency map over the image.

    With ``intensity_alpha`` the blend strength follows the activation, so cold
    areas stay legible instead of being washed out by a flat colour wash — a
    noticeable readability upgrade over a constant-alpha blend.
    """
    from PIL import Image

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    cam_resized = resize_cam(cam, (base.shape[1], base.shape[0]))
    heat = apply_colormap(cam_resized, ensure_colormap(colormap), gamma=gamma)
    weights = cam_resized * alpha if intensity_alpha else alpha
    return Image.fromarray(blend(base, heat, weights))


def draw_regions(
    image,
    regions: Sequence[Region],
    color: RGB = FAKE_COLOR,
    thickness: int = 3,
    labels: bool = True,
):
    """Draw ranked bounding boxes with a small score badge on each."""
    from PIL import Image, ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(max(12, canvas.width // 55))

    for region in regions:
        x1, y1, x2, y2 = (int(v) for v in region.bbox)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        if not labels:
            continue
        text = f"#{region.rank} {region.score:.2f}"
        text_w, text_h = _text_size(draw, text, font)
        pad = 4
        badge_y1 = max(0, y1 - text_h - 2 * pad)
        draw.rectangle(
            [x1, badge_y1, x1 + text_w + 2 * pad, badge_y1 + text_h + 2 * pad],
            fill=color,
        )
        draw.text((x1 + pad, badge_y1 + pad), text, fill=(255, 255, 255), font=font)

    return canvas


def add_banner(image, text: str, accent: RGB = FAKE_COLOR, height: Optional[int] = None):
    """Add a title bar above the image (verdict + confidence)."""
    from PIL import Image, ImageDraw

    banner_h = height or max(36, image.height // 14)
    canvas = Image.new("RGB", (image.width, image.height + banner_h), INK)
    canvas.paste(image.convert("RGB"), (0, banner_h))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, 6, banner_h], fill=accent)
    font = _font(max(13, banner_h // 2))
    _, text_h = _text_size(draw, text, font)
    draw.text((16, max(2, (banner_h - text_h) // 2)), text, fill=(240, 240, 245), font=font)
    return canvas


def crop_region(image, region: Region, min_size: int = 96):
    """Crop a region, growing the box to ``min_size`` while staying in bounds.

    Downstream VLMs reject very small inputs (Qwen2-VL needs >= 28px). Growing
    the crop here removes the "crop too small, using the whole image" fallback
    that used to throw away the localisation entirely.
    """
    x1, y1, x2, y2 = (int(v) for v in region.bbox)
    width, height = image.size

    target = min(min_size, width, height)
    if x2 - x1 < target:
        deficit = target - (x2 - x1)
        x1 = max(0, x1 - deficit // 2)
        x2 = min(width, x1 + target)
        x1 = max(0, x2 - target)
    if y2 - y1 < target:
        deficit = target - (y2 - y1)
        y1 = max(0, y1 - deficit // 2)
        y2 = min(height, y1 + target)
        y1 = max(0, y2 - target)

    return image.crop((x1, y1, x2, y2))


# --------------------------------------------------------------------------- #
# Composite panels
# --------------------------------------------------------------------------- #


def side_by_side(images: Sequence, labels: Optional[Sequence[str]] = None, gap: int = 12, background: RGB = PAPER):
    """Lay images out in a row, scaled to a common height, with captions."""
    from PIL import Image, ImageDraw

    if not images:
        raise ValueError("side_by_side needs at least one image")

    target_h = max(img.height for img in images)
    scaled = [
        img.convert("RGB").resize(
            (max(1, int(img.width * target_h / img.height)), target_h), Image.LANCZOS
        )
        for img in images
    ]

    caption_h = 30 if labels else 0
    total_w = sum(img.width for img in scaled) + gap * (len(scaled) + 1)
    canvas = Image.new("RGB", (total_w, target_h + caption_h + 2 * gap), background)
    draw = ImageDraw.Draw(canvas)
    font = _font(15)

    x = gap
    for index, img in enumerate(scaled):
        canvas.paste(img, (x, gap))
        if labels and index < len(labels):
            text = labels[index]
            text_w, _ = _text_size(draw, text, font)
            draw.text(
                (x + max(0, (img.width - text_w) // 2), gap + target_h + 8),
                text,
                fill=INK,
                font=font,
            )
        x += img.width + gap

    return canvas


def summary_panel(
    original,
    overlay,
    crop=None,
    verdict: str = "FAKE",
    confidence: float = 0.0,
    artifacts: Optional[Iterable[str]] = None,
    colormap: str = "turbo",
):
    """A single shareable image: original | heatmap | crop, plus a caption bar."""
    from PIL import Image, ImageDraw

    accent = FAKE_COLOR if verdict.upper() == "FAKE" else REAL_COLOR
    panels = [original, overlay]
    labels = ["original", "saliency overlay"]
    if crop is not None:
        panels.append(crop)
        labels.append("suspicious region")

    row = side_by_side(panels, labels)

    artifact_list = list(artifacts or [])[:3]
    footer_h = 34 + (22 * len(artifact_list) if artifact_list else 0)
    canvas = Image.new("RGB", (row.width, row.height + footer_h + 26), PAPER)
    canvas.paste(row, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = row.height + 4
    draw.rectangle([12, y, row.width - 12, y + 3], fill=accent)
    y += 12

    title_font = _font(18)
    draw.text((14, y), f"{verdict}  ·  {confidence:.1f}% confidence", fill=accent, font=title_font)
    y += 26

    body_font = _font(14)
    for name in artifact_list:
        draw.text((18, y), f"• {name}", fill=INK, font=body_font)
        y += 20

    legend = Image.fromarray(colorbar_strip(ensure_colormap(colormap), width=180, height=10))
    canvas.paste(legend, (row.width - 200, row.height + 12))
    draw.text((row.width - 200, row.height + 24), "low → high activation", fill=(90, 90, 100), font=_font(11))

    return canvas


def contact_sheet(images: Sequence, columns: int = 4, thumb: int = 220, background: RGB = PAPER):
    """Grid montage of many images (used by ``aidetect report``)."""
    from PIL import Image

    if not images:
        raise ValueError("contact_sheet needs at least one image")

    columns = max(1, columns)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb, rows * thumb), background)

    for index, image in enumerate(images):
        tile = image.convert("RGB").copy()
        tile.thumbnail((thumb - 8, thumb - 8), Image.LANCZOS)
        x = (index % columns) * thumb + (thumb - tile.width) // 2
        y = (index // columns) * thumb + (thumb - tile.height) // 2
        canvas.paste(tile, (x, y))

    return canvas
