"""Submission writers, CLI parsing and the colormap LUTs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aidetect import cli, submission
from aidetect.explainability import colormaps
from aidetect.types import AnalysisResult, ArtifactMatch, ArtifactReport, DetectionResult, Explanation


def build_result(name: str = "image_0042.png", fake: bool = True) -> AnalysisResult:
    detection = DetectionResult(
        image_path=name,
        prediction="FAKE" if fake else "REAL",
        confidence=92.0,
        probabilities={"FAKE": 0.92 if fake else 0.08, "REAL": 0.08 if fake else 0.92},
    )
    report = ArtifactReport(
        matches=[
            ArtifactMatch("Artificial smoothness", 0.9, "Texture & Material", 1),
            ArtifactMatch("Inconsistent shadow directions", 0.8, "Lighting & Shadows", 2),
        ],
        categories=["Texture & Material"],
    )
    return AnalysisResult(
        image_path=name,
        detection=detection,
        artifacts=report,
        explanation=Explanation(text="Surfaces are unnaturally smooth.", backend="rule_based"),
    )


# ---- submission ----------------------------------------------------------- #


def test_index_extracted_from_filename():
    assert submission.index_from_path("data/image_0042.png") == 42
    assert submission.index_from_path("no_digits.png", fallback=7) == 7


def test_task1_records():
    records = submission.task1_records([build_result(), build_result("image_7.png", fake=False)])
    assert records[0] == {
        "index": 42,
        "image": "image_0042.png",
        "prediction": "FAKE",
        "label": 1,
        "confidence": 92.0,
        "fake_probability": 0.92,
    }
    assert records[1]["label"] == 0


def test_task2_records_shape_and_budget():
    long_text = " ".join(["word"] * 90)
    result = build_result()
    result.explanation = Explanation(text=long_text, backend="qwen2vl")

    records = submission.task2_records([result], max_artifacts=2, max_words=50)
    assert records[0]["index"] == 42
    assert set(records[0]["explanation"]) == {
        "Artificial smoothness",
        "Inconsistent shadow directions",
    }
    for text in records[0]["explanation"].values():
        assert len(text.split()) <= 50


def test_task2_accepts_per_artifact_text():
    result = build_result()
    custom = {result.image_path: {"Artificial smoothness": "Plastic-looking skin."}}
    records = submission.task2_records([result], per_artifact=custom)
    assert records[0]["explanation"]["Artificial smoothness"] == "Plastic-looking skin."


def test_validate_task2_detects_problems():
    records = [
        {"index": 1, "explanation": {"A": " ".join(["w"] * 60)}},
        {"index": 1, "explanation": {}},
        {"explanation": {}},
    ]
    problems = submission.validate_task2(records, max_words=50)
    assert any("exceeds" in p for p in problems)
    assert any("duplicate" in p for p in problems)
    assert any("missing 'index'" in p for p in problems)


def test_validate_task2_accepts_good_submission():
    records = submission.task2_records([build_result()])
    assert submission.validate_task2(records) == []


def test_write_submissions(tmp_path):
    results = [build_result()]
    task1 = submission.write_task1(results, tmp_path / "t1.json")
    task2 = submission.write_task2(results, tmp_path / "t2.json")
    assert json.loads(task1.read_text())[0]["prediction"] == "FAKE"
    assert "explanation" in json.loads(task2.read_text())[0]


# ---- CLI ------------------------------------------------------------------ #


def test_parser_builds_and_lists_commands():
    parser = cli.build_parser()
    args = parser.parse_args(["analyze", "a.png", "--backend", "qwen2vl", "--top-k", "3"])
    assert args.command == "analyze"
    assert args.images == ["a.png"]
    assert args.backend == "qwen2vl"
    assert args.top_k == 3


def test_cli_flags_map_onto_config():
    args = cli.build_parser().parse_args(
        [
            "analyze",
            "a.png",
            "--threshold",
            "0.7",
            "--backend",
            "moondream",
            "--device",
            "cpu",
            "--tta",
            "--explain-method",
            "gradcam++",
            "--set",
            "artifacts.top_k=9",
        ]
    )
    config = cli.config_from_args(args)
    assert config.detector.decision_threshold == 0.7
    assert config.describe.backend == "moondream"
    assert config.runtime.device == "cpu"
    assert config.detector.tta_hflip is True
    assert config.explain.method == "gradcam++"
    assert config.artifacts.top_k == 9


def test_save_outputs_flag_parsing():
    args = cli.build_parser().parse_args(
        ["analyze", "a.png", "--save-outputs", "overlay,panel"]
    )
    assert cli.config_from_args(args).explain.save_outputs == ["overlay", "panel"]


def test_no_command_prints_help(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    assert "usage" in capsys.readouterr().out.lower()


def test_taxonomy_command_json(capsys):
    assert cli.main(["taxonomy", "--format", "json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 70


def test_taxonomy_command_filtered(capsys):
    assert cli.main(["taxonomy", "--category", "lighting", "--format", "csv"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Inconsistent shadow directions" in out
    assert "Incorrect wheel geometry" not in out


def test_unknown_flag_exits_with_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["analyze", "a.png", "--nope"])
    assert excinfo.value.code == 2


def test_formatters_handle_error_results():
    result = AnalysisResult(
        image_path="bad.png",
        detection=DetectionResult(image_path="bad.png", prediction="ERROR", confidence=0.0),
        error="boom",
    )
    assert "ERROR: boom" in cli.format_result(result)


# ---- colormaps ------------------------------------------------------------ #


@pytest.mark.parametrize("name", colormaps.AVAILABLE_COLORMAPS)
def test_colormaps_produce_valid_rgb(name):
    values = np.linspace(0, 1, 32, dtype=np.float32)
    rgb = colormaps.apply_colormap(values, name)
    assert rgb.shape == (32, 3)
    assert rgb.dtype == np.uint8


def test_colormap_is_monotone_in_brightness_at_the_ends():
    ramp = colormaps.apply_colormap(np.array([0.0, 1.0], dtype=np.float32), "viridis")
    assert ramp[0].sum() != ramp[1].sum()


def test_apply_colormap_clips_out_of_range_values():
    rgb = colormaps.apply_colormap(np.array([-5.0, 5.0], dtype=np.float32), "turbo")
    assert rgb.shape == (2, 3)


def test_unknown_colormap_falls_back():
    assert colormaps.ensure_colormap("not-a-map") == "turbo"
    with pytest.raises(KeyError):
        colormaps.build_lut("not-a-map")


def test_blend_respects_alpha():
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    white = np.full((2, 2, 3), 255, dtype=np.uint8)
    assert colormaps.blend(black, white, 0.0).max() == 0
    assert colormaps.blend(black, white, 1.0).min() == 255
    assert 100 < colormaps.blend(black, white, 0.5).mean() < 155


def test_sequential_palette_hex():
    palette = colormaps.sequential_palette(4)
    assert len(palette) == 4
    assert all(color.startswith("#") and len(color) == 7 for color in palette)
