"""Saliency + rendering, exercised on a randomly-initialised tiny ViT.

No weights are downloaded (``pretrained=False``) and no project checkpoint is
needed, so these run anywhere torch and timm are installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from aidetect.config import DetectorConfig  # noqa: E402
from aidetect.core import localization  # noqa: E402
from aidetect.core.model import DINOv2Classifier, build_head  # noqa: E402
from aidetect.explainability import build_saliency  # noqa: E402
from aidetect.explainability.gradcam import ViTGradCAM, normalize_map  # noqa: E402
from aidetect.exceptions import ExplainabilityError  # noqa: E402

IMAGE_SIZE = 224


@pytest.fixture(scope="module")
def tiny_model():
    """A 5M-parameter ViT wired up exactly like the real classifier."""
    torch.manual_seed(0)
    cfg = DetectorConfig(image_size=IMAGE_SIZE)
    backbone = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=0, global_pool="token"
    )
    model = DINOv2Classifier(backbone, build_head(backbone.num_features, cfg)).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def sample_batch():
    torch.manual_seed(1)
    return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_model_forward_shape(tiny_model, sample_batch):
    assert tiny_model(sample_batch).shape == (1, 2)


@pytest.mark.parametrize("method", ["gradcam", "gradcam++", "rollout", "occlusion"])
def test_saliency_methods_produce_usable_maps(tiny_model, sample_batch, method):
    engine = build_saliency(method, tiny_model, patch_size=64, stride=64, batch_size=4)
    cam = engine.generate(sample_batch, device="cpu", target_class=0, output_size=IMAGE_SIZE)

    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert np.isfinite(cam).all()
    assert 0.0 <= cam.min() <= cam.max() <= 1.0
    # Regression guard: hooking the last block's *output* gave an all-zero map
    # because only the [CLS] token reaches the head under global_pool="token".
    assert cam.max() > 0.0, f"{method} produced a blank saliency map"
    assert cam.std() > 0.0, f"{method} produced a constant saliency map"


def test_gradcam_targets_a_layer_with_spatial_gradients(tiny_model):
    layer = ViTGradCAM._default_target_layer(tiny_model)
    assert layer is not tiny_model.backbone.blocks[-1]
    assert layer is tiny_model.backbone.blocks[-1].norm1


def test_gradcam_rejects_out_of_range_class(tiny_model, sample_batch):
    with pytest.raises(ExplainabilityError):
        ViTGradCAM(tiny_model).generate(sample_batch, target_class=99)


def test_gradcam_rejects_bad_tensor_rank(tiny_model):
    with pytest.raises(ExplainabilityError):
        ViTGradCAM(tiny_model).generate(torch.randn(3, IMAGE_SIZE, IMAGE_SIZE))


def test_unknown_method_is_rejected(tiny_model):
    with pytest.raises(ExplainabilityError):
        build_saliency("vibes", tiny_model)


def test_hooks_are_always_removed(tiny_model, sample_batch):
    cam_engine = ViTGradCAM(tiny_model)
    cam_engine.generate(sample_batch)
    assert cam_engine._handles == []


def test_saliency_feeds_localisation(tiny_model, sample_batch):
    cam = build_saliency("gradcam", tiny_model).generate(sample_batch, output_size=IMAGE_SIZE)
    regions = localization.find_regions(cam, image_size=(400, 300), max_regions=3)
    assert regions
    for region in regions:
        x1, y1, x2, y2 = region.bbox
        assert 0 <= x1 < x2 <= 400
        assert 0 <= y1 < y2 <= 300
    assert [r.rank for r in regions] == list(range(1, len(regions) + 1))


def test_normalize_map_handles_degenerate_input():
    assert normalize_map(np.full((4, 4), 3.0)).max() == 0.0
    assert normalize_map(np.array([[0.0, np.nan], [1.0, np.inf]])).max() <= 1.0


# ---- rendering ------------------------------------------------------------ #


def test_rendering_pipeline_produces_images(tiny_model, sample_batch):
    from PIL import Image

    from aidetect.explainability import visualize
    from aidetect.types import Region

    image = Image.fromarray(
        (np.random.RandomState(0).rand(300, 400, 3) * 255).astype(np.uint8)
    )
    cam = build_saliency("gradcam", tiny_model).generate(sample_batch, output_size=IMAGE_SIZE)
    regions = localization.find_regions(cam, image_size=image.size)

    overlay = visualize.overlay_heatmap(image, cam, alpha=0.45, colormap="turbo")
    assert overlay.size == image.size

    annotated = visualize.draw_regions(image, regions)
    assert annotated.size == image.size

    banner = visualize.add_banner(annotated, "FAKE · 93.1%")
    assert banner.height > image.height

    crop = visualize.crop_region(image, regions[0], min_size=96)
    assert crop.width >= 96 and crop.height >= 96

    panel = visualize.summary_panel(
        image, overlay, crop, verdict="FAKE", confidence=93.1, artifacts=["Artificial smoothness"]
    )
    assert panel.width > image.width

    sheet = visualize.contact_sheet([image, crop], columns=2, thumb=100)
    assert sheet.size == (200, 100)


def test_crop_region_grows_tiny_boxes():
    from PIL import Image

    from aidetect.explainability import visualize
    from aidetect.types import Region

    image = Image.new("RGB", (200, 200))
    crop = visualize.crop_region(image, Region(bbox=(10, 10, 14, 14), score=1.0), min_size=96)
    assert crop.size == (96, 96)


def test_crop_region_never_exceeds_the_image():
    from PIL import Image

    from aidetect.explainability import visualize
    from aidetect.types import Region

    image = Image.new("RGB", (40, 40))
    crop = visualize.crop_region(image, Region(bbox=(0, 0, 4, 4), score=1.0), min_size=96)
    assert crop.size == (40, 40)
