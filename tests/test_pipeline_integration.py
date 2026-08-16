"""End-to-end orchestration with a stand-in detector.

A randomly-initialised tiny ViT is injected into :class:`Detector`, and Stage 2
is stubbed, so the whole pipeline — preprocessing, saliency, localisation,
cropping, description, rendering, JSON/CSV/HTML output — is exercised without
any checkpoint, download or GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from aidetect.artifacts import taxonomy  # noqa: E402
from aidetect.config import AppConfig  # noqa: E402
from aidetect.core.detector import Detector  # noqa: E402
from aidetect.core.model import (  # noqa: E402
    CheckpointInfo,
    DINOv2Classifier,
    build_head,
    extract_state_dict,
    infer_checkpoint_kind,
    resolve_checkpoint_path,
    strip_prefixes,
)
from aidetect.exceptions import CheckpointError  # noqa: E402
from aidetect.pipeline import AnalysisPipeline  # noqa: E402
from aidetect.types import ArtifactMatch, ArtifactReport  # noqa: E402

IMAGE_SIZE = 224


def make_config(tmp_path) -> AppConfig:
    config = AppConfig()
    config.detector.image_size = IMAGE_SIZE
    config.detector.backbone = "vit_tiny_patch16_224"
    config.runtime.device = "cpu"
    config.runtime.output_dir = str(tmp_path / "outputs")
    config.explain.save_outputs = ["raw", "overlay", "crop", "annotated", "panel"]
    config.describe.backend = "rule_based"
    return config.validate()


def make_detector(config: AppConfig) -> Detector:
    """A Detector with weights injected instead of loaded from disk."""
    torch.manual_seed(0)
    detector = Detector.__new__(Detector)
    detector.config = config
    detector.cfg = config.detector
    detector.device = "cpu"
    detector.dtype = torch.float32
    detector.checkpoint_path = "<injected>"

    backbone = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=0, global_pool="token"
    )
    model = DINOv2Classifier(backbone, build_head(backbone.num_features, config.detector)).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    detector._model = model
    detector._info = CheckpointInfo(path="<injected>", kind="adversarial", epoch=3)
    return detector


class StubClassifier:
    """Stands in for SigLIP so Stage 2 needs no model download."""

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, image, top_k=None, return_all=False) -> ArtifactReport:
        self.calls += 1
        descriptors = taxonomy.OFFICIAL_ARTIFACTS[:3]
        matches = [
            ArtifactMatch(
                descriptor=name,
                score=0.9 - 0.1 * index,
                raw_score=0.5,
                category=taxonomy.category_of(name),
                artifact_id=taxonomy.artifact_id(name),
                rank=index + 1,
            )
            for index, name in enumerate(descriptors)
        ]
        return ArtifactReport(
            matches=matches,
            categories=list(dict.fromkeys(m.category for m in matches)),
            category_scores={matches[0].category: 0.9},
            model_name="stub",
            score_mode="minmax",
        )

    def unload(self) -> None:
        pass


@pytest.fixture
def sample_image(tmp_path):
    from PIL import Image

    path = tmp_path / "image_0001.png"
    Image.fromarray(
        (np.random.RandomState(0).rand(300, 400, 3) * 255).astype(np.uint8)
    ).save(path)
    return path


@pytest.fixture
def pipeline(tmp_path):
    config = make_config(tmp_path)
    pipe = AnalysisPipeline(config)
    pipe._detector = make_detector(config)
    pipe._classifier = StubClassifier()
    return pipe


# ---- detector ------------------------------------------------------------- #


def test_detector_predict_returns_calibrated_probabilities(tmp_path, sample_image):
    detector = make_detector(make_config(tmp_path))
    result = detector.predict(str(sample_image))

    assert result.prediction in ("FAKE", "REAL")
    assert 50.0 <= result.confidence <= 100.0
    assert pytest.approx(sum(result.probabilities.values()), abs=1e-5) == 1.0
    assert set(result.probabilities) == {"FAKE", "REAL"}
    assert result.model_type == "adversarial"
    assert result.latency_ms >= 0


def test_detector_threshold_changes_the_verdict(tmp_path, sample_image):
    config = make_config(tmp_path)
    detector = make_detector(config)
    fake_prob = detector.predict(str(sample_image)).fake_prob

    detector.cfg.decision_threshold = max(0.001, fake_prob - 0.01)
    assert detector.predict(str(sample_image)).prediction == "FAKE"

    detector.cfg.decision_threshold = min(0.999, fake_prob + 0.01)
    assert detector.predict(str(sample_image)).prediction == "REAL"


def test_temperature_softens_probabilities(tmp_path, sample_image):
    config = make_config(tmp_path)
    detector = make_detector(config)
    sharp = detector.predict(str(sample_image)).confidence
    detector.cfg.temperature = 10.0
    assert detector.predict(str(sample_image)).confidence <= sharp


def test_batch_matches_single_prediction(tmp_path, sample_image):
    detector = make_detector(make_config(tmp_path))
    single = detector.predict(str(sample_image))
    batched = detector.predict_batch([str(sample_image)] * 3)

    assert len(batched) == 3
    for result in batched:
        assert result.prediction == single.prediction
        assert pytest.approx(result.fake_prob, abs=1e-4) == single.fake_prob


def test_tta_runs_and_stays_normalised(tmp_path, sample_image):
    config = make_config(tmp_path)
    config.detector.tta_hflip = True
    result = make_detector(config).predict(str(sample_image))
    assert result.tta is True
    assert pytest.approx(sum(result.probabilities.values()), abs=1e-5) == 1.0


def test_localize_returns_ranked_regions(tmp_path, sample_image):
    from aidetect.io_utils import load_image

    detector = make_detector(make_config(tmp_path))
    image = load_image(sample_image)
    cam, regions = detector.localize(image)

    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert regions
    for region in regions:
        assert region.bbox[2] <= image.width and region.bbox[3] <= image.height


def test_detector_describe(tmp_path):
    info = make_detector(make_config(tmp_path)).describe()
    assert info["device"] == "cpu"
    assert info["parameters"]["total"] > 0


# ---- checkpoint helpers ---------------------------------------------------- #


def test_extract_state_dict_supports_every_layout():
    weights = {"head.0.weight": torch.zeros(2, 2)}
    assert extract_state_dict({"model_state": weights}) is weights
    assert extract_state_dict({"state_dict": weights}) is weights
    assert extract_state_dict(weights) is weights
    with pytest.raises(CheckpointError):
        extract_state_dict({"nothing": 1})


def test_strip_prefixes_removes_ddp_and_compile_wrappers():
    state = {"module._orig_mod.head.0.weight": torch.zeros(1)}
    assert list(strip_prefixes(state)) == ["head.0.weight"]


def test_infer_checkpoint_kind():
    assert infer_checkpoint_kind("checkpoints_adv/adv_best_model.pt") == "adversarial"
    assert infer_checkpoint_kind("checkpoints/best_model.pt") == "original"
    assert infer_checkpoint_kind("weights/foo.pt") == "unknown"


def test_resolve_checkpoint_path_uses_fallback(tmp_path):
    from aidetect.config import DetectorConfig

    fallback = tmp_path / "best_model.pt"
    fallback.write_bytes(b"stub")
    cfg = DetectorConfig(
        checkpoint=str(tmp_path / "missing.pt"), fallback_checkpoints=[str(fallback)]
    )
    assert resolve_checkpoint_path(cfg) == fallback


def test_resolve_checkpoint_path_reports_everything_tried(tmp_path):
    from aidetect.config import DetectorConfig

    cfg = DetectorConfig(checkpoint=str(tmp_path / "a.pt"), fallback_checkpoints=[str(tmp_path / "b.pt")])
    with pytest.raises(CheckpointError) as excinfo:
        resolve_checkpoint_path(cfg)
    assert "a.pt" in str(excinfo.value) and "b.pt" in str(excinfo.value)


# ---- full pipeline --------------------------------------------------------- #


def test_analyze_produces_a_complete_result(pipeline, sample_image, tmp_path):
    pipeline.config.runtime.explain_real_images = True  # force stages 2/3 regardless
    result = pipeline.analyze(sample_image)

    assert result.ok
    assert result.detection.prediction in ("FAKE", "REAL")
    assert result.regions
    assert result.artifacts is not None and len(result.artifacts.matches) == 3
    assert result.explanation is not None and result.explanation.backend == "rule_based"
    assert len(result.description.split()) <= pipeline.config.describe.max_words
    assert set(result.saved_files) == {"raw", "overlay", "crop", "annotated", "panel"}
    for path in result.saved_files.values():
        assert (tmp_path / "outputs").exists() and path.endswith(".png")
    assert result.timings["total_ms"] > 0
    assert result.regions[0].crop_path == result.saved_files["crop"]


def test_real_images_skip_stage_2_and_3(pipeline, sample_image):
    pipeline.config.detector.decision_threshold = 0.999  # force a REAL verdict
    result = pipeline.analyze(sample_image, save_visuals=False)

    assert result.detection.prediction == "REAL"
    assert result.artifacts is None
    assert result.explanation is None
    assert pipeline._classifier.calls == 0


def test_result_json_roundtrips_through_disk(pipeline, sample_image, tmp_path):
    from aidetect.types import AnalysisResult

    pipeline.config.runtime.explain_real_images = True
    result = pipeline.analyze(sample_image, save_visuals=False)

    path = tmp_path / "result.json"
    path.write_text(result.to_json())
    restored = AnalysisResult.from_dict(json.loads(path.read_text()))

    assert restored.detection.prediction == result.detection.prediction
    assert restored.description == result.description


def test_analyze_many_isolates_failures(pipeline, sample_image, tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    results = list(pipeline.analyze_many([sample_image, broken], save_visuals=False))
    assert len(results) == 2
    assert results[0].ok
    assert not results[1].ok and "broken.png" in results[1].image_path


def test_batch_runner_writes_every_output(pipeline, sample_image, tmp_path):
    from aidetect.batch import run_batch
    from aidetect.io_utils import read_jsonl

    out_dir = tmp_path / "run"
    results, summary = run_batch(
        pipeline, [sample_image, sample_image], out_dir, save_visuals=False
    )

    assert summary.total == 2 and summary.succeeded == 2 and summary.failed == 0
    assert len(list(read_jsonl(out_dir / "results.jsonl"))) == 2
    assert (out_dir / "summary.csv").is_file()
    assert json.loads((out_dir / "summary.json").read_text())["succeeded"] == 2
    assert len(results) == 2


def test_batch_resume_skips_finished_images(pipeline, sample_image, tmp_path):
    from aidetect.batch import run_batch

    out_dir = tmp_path / "run"
    run_batch(pipeline, [sample_image], out_dir, save_visuals=False)
    _, second = run_batch(pipeline, [sample_image], out_dir, save_visuals=False, resume=True)
    assert second.total == 0


def test_report_and_submission_from_a_real_run(pipeline, sample_image, tmp_path):
    from aidetect.report import write_report
    from aidetect.submission import task1_records, task2_records, validate_task2

    pipeline.config.runtime.explain_real_images = True
    pipeline.config.detector.decision_threshold = 0.0001  # force FAKE
    result = pipeline.analyze(sample_image, save_visuals=True)

    html = write_report([result], tmp_path / "report.html").read_text()
    assert "FAKE" in html
    assert "data:image/png;base64," in html  # visuals inlined

    assert task1_records([result])[0]["index"] == 1
    records = task2_records([result])
    assert validate_task2(records) == []
