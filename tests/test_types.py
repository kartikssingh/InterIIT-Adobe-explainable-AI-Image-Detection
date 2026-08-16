"""Result dataclasses: derived properties and JSON round-tripping."""

from __future__ import annotations

from aidetect.types import (
    AnalysisResult,
    ArtifactMatch,
    ArtifactReport,
    BatchSummary,
    DetectionResult,
    Explanation,
    Region,
    results_to_records,
)


def make_detection(fake_prob: float = 0.93) -> DetectionResult:
    return DetectionResult(
        image_path="img.png",
        prediction="FAKE" if fake_prob >= 0.5 else "REAL",
        confidence=max(fake_prob, 1 - fake_prob) * 100,
        probabilities={"FAKE": fake_prob, "REAL": 1 - fake_prob},
        threshold=0.5,
    )


def test_detection_properties():
    detection = make_detection(0.93)
    assert detection.is_fake
    assert detection.fake_prob == 0.93
    assert round(detection.real_prob, 2) == 0.07
    assert round(detection.margin, 2) == 0.43
    assert detection.certainty_label() == "very high"


def test_borderline_certainty():
    assert make_detection(0.52).certainty_label() == "borderline"


def test_region_position_phrases():
    assert Region(bbox=(0, 0, 10, 10), score=1.0, peak_pct=(10.0, 10.0)).position_phrase() == "upper-left"
    assert Region(bbox=(0, 0, 10, 10), score=1.0, peak_pct=(50.0, 50.0)).position_phrase() == "centre"
    assert Region(bbox=(0, 0, 10, 10), score=1.0, peak_pct=(90.0, 90.0)).position_phrase() == "lower-right"


def test_region_geometry():
    region = Region(bbox=(10, 20, 110, 220), score=0.8)
    assert region.width == 100
    assert region.height == 200
    assert region.center == (60.0, 120.0)


def build_result() -> AnalysisResult:
    report = ArtifactReport(
        matches=[
            ArtifactMatch("Artificial smoothness", 0.91, "Texture & Material", 1, 0.4, "artificial_smoothness"),
            ArtifactMatch("Inconsistent shadow directions", 0.75, "Lighting & Shadows", 2, 0.3, "x"),
        ],
        categories=["Texture & Material", "Lighting & Shadows"],
        category_scores={"Texture & Material": 0.91},
        model_name="google/siglip-base-patch16-224",
        score_mode="minmax",
    )
    return AnalysisResult(
        image_path="img.png",
        detection=make_detection(),
        artifacts=report,
        explanation=Explanation(text="Surfaces look plastic.", backend="rule_based", word_count=4),
        regions=[Region(bbox=(4, 8, 100, 120), score=0.9, rank=1, peak_pct=(20.0, 30.0))],
        saved_files={"overlay": "outputs/img/overlay.png"},
        timings={"detect": 12.0, "total_ms": 30.0},
    )


def test_analysis_result_roundtrip():
    original = build_result()
    restored = AnalysisResult.from_dict(original.to_dict())

    assert restored.image_path == original.image_path
    assert restored.detection.prediction == "FAKE"
    assert restored.artifacts is not None
    assert restored.artifacts.matches[0].descriptor == "Artificial smoothness"
    assert restored.artifacts.categories == original.artifacts.categories
    assert restored.description == "Surfaces look plastic."
    assert restored.regions[0].bbox == (4, 8, 100, 120)
    assert restored.saved_files == original.saved_files


def test_result_dict_includes_derived_fields():
    data = build_result().to_dict()
    assert data["detection"]["is_fake"] is True
    assert data["detection"]["certainty"] == "very high"
    assert data["regions"][0]["position"] == "upper-left"
    assert data["schema_version"] == "2.0"


def test_error_result_is_not_ok():
    result = AnalysisResult(
        image_path="bad.png",
        detection=DetectionResult(image_path="bad.png", prediction="ERROR", confidence=0.0),
        error="ImageLoadError: broken",
    )
    assert not result.ok
    assert AnalysisResult.from_dict(result.to_dict()).error == "ImageLoadError: broken"


def test_records_flatten_for_csv():
    rows = results_to_records([build_result()])
    assert rows[0]["prediction"] == "FAKE"
    assert rows[0]["top_artifact"] == "Artificial smoothness"
    assert rows[0]["description"] == "Surfaces look plastic."


def test_batch_summary_rates():
    summary = BatchSummary(total=10, succeeded=8, failed=2, fake=6, real=2, elapsed_s=4.0)
    assert summary.fake_rate == 0.75
    assert summary.throughput == 2.5
    assert summary.to_dict()["images_per_second"] == 2.5
