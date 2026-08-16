"""Connected-component region extraction (numpy only, no model needed)."""

from __future__ import annotations

import numpy as np

from aidetect.core import localization


def test_hot_mask_uses_threshold():
    cam = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    assert localization.hot_mask(cam, threshold=0.5).tolist() == [[False, True], [False, True]]


def test_hot_mask_falls_back_to_percentile():
    cam = np.full((10, 10), 0.2, dtype=np.float32)
    cam[3, 3] = 0.45  # below the 0.5 threshold but clearly the peak
    mask = localization.hot_mask(cam, threshold=0.5, fallback_percentile=90.0)
    assert mask.any()
    assert mask[3, 3]


def test_hot_mask_on_flat_map_keeps_one_pixel():
    cam = np.zeros((8, 8), dtype=np.float32)
    assert localization.hot_mask(cam, threshold=0.5).sum() == 1


def test_label_components_separates_islands():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[7:9, 7:9] = True
    _, count = localization.label_components(mask)
    assert count == 2


def test_label_components_merges_diagonal_touching_with_8_connectivity():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    assert localization.label_components(mask, connectivity=8)[1] == 1
    assert localization.label_components(mask, connectivity=4)[1] == 2


def test_two_hotspots_produce_two_regions():
    """The old implementation returned one box spanning both blobs."""
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[5:12, 5:12] = 1.0
    cam[50:58, 50:58] = 0.9

    regions = localization.find_regions(cam, image_size=(64, 64), threshold=0.5, padding=0)
    assert len(regions) == 2
    assert regions[0].score >= regions[1].score
    assert regions[0].rank == 1

    # No single box may swallow the whole frame.
    for region in regions:
        assert region.width < 60 and region.height < 60


def test_regions_scale_to_image_coordinates():
    cam = np.zeros((32, 32), dtype=np.float32)
    cam[8:16, 8:16] = 1.0
    regions = localization.find_regions(cam, image_size=(320, 320), threshold=0.5, padding=0)
    x1, y1, x2, y2 = regions[0].bbox
    assert 0 <= x1 < x2 <= 320
    assert 0 <= y1 < y2 <= 320
    assert (x2 - x1) > 40  # 8/32 of 320 = 80px, allowing for grid quantisation


def test_regions_are_clamped_and_non_empty():
    cam = np.zeros((16, 16), dtype=np.float32)
    cam[0, 0] = 1.0
    regions = localization.find_regions(cam, image_size=(50, 50), threshold=0.5, padding=40)
    x1, y1, x2, y2 = regions[0].bbox
    assert (x1, y1) == (0, 0)
    assert x2 <= 50 and y2 <= 50
    assert x2 > x1 and y2 > y1


def test_max_regions_limits_output():
    cam = np.zeros((64, 64), dtype=np.float32)
    for index, offset in enumerate((2, 20, 40, 56)):
        cam[offset : offset + 4, offset : offset + 4] = 1.0 - index * 0.1
    regions = localization.find_regions(cam, image_size=(64, 64), max_regions=2, threshold=0.5)
    assert len(regions) == 2


def test_iou_and_merge():
    assert localization.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert localization.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    from aidetect.types import Region

    duplicates = [
        Region(bbox=(0, 0, 10, 10), score=0.9),
        Region(bbox=(0, 0, 10, 10), score=0.5),
        Region(bbox=(50, 50, 60, 60), score=0.7),
    ]
    merged = localization.merge_overlapping(duplicates, iou_threshold=0.6)
    assert len(merged) == 2
    assert [r.rank for r in merged] == [1, 2]


def test_whole_image_fallback():
    region = localization.whole_image_region((200, 100), score=0.4)
    assert region.bbox == (0, 0, 200, 100)
    assert region.area_fraction == 1.0


def test_saliency_stats():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[9, 0] = 1.0
    stats = localization.saliency_stats(cam)
    assert stats["max"] == 1.0
    assert stats["peak_row_pct"] == 90.0
    assert stats["peak_col_pct"] == 0.0
    assert 0.0 < stats["concentration"] <= 1.0


def test_downsample_mask_preserves_signal():
    mask = np.zeros((128, 128), dtype=bool)
    mask[64:70, 64:70] = True
    coarse = localization.downsample_mask(mask, grid=32)
    assert coarse.shape == (32, 32)
    assert coarse.any()
