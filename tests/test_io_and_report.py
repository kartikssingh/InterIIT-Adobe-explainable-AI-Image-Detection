"""IO helpers, the HTML report writer and batch bookkeeping."""

from __future__ import annotations

import json

import pytest

from aidetect import io_utils, report
from aidetect.batch import aggregate_stats, completed_images
from aidetect.exceptions import ImageLoadError
from aidetect.types import AnalysisResult, ArtifactMatch, ArtifactReport, DetectionResult, Explanation


def make_result(name="a.png", fake=True, error=None) -> AnalysisResult:
    return AnalysisResult(
        image_path=name,
        detection=DetectionResult(
            image_path=name,
            prediction="FAKE" if fake else "REAL",
            confidence=91.0,
            probabilities={"FAKE": 0.91 if fake else 0.09, "REAL": 0.09 if fake else 0.91},
        ),
        artifacts=ArtifactReport(
            matches=[ArtifactMatch("Artificial smoothness", 0.9, "Texture & Material", 1)],
            categories=["Texture & Material"],
        ),
        explanation=Explanation(text="Too smooth to be real.", backend="rule_based"),
        error=error,
    )


# ---- io ------------------------------------------------------------------- #


def test_slugify():
    assert io_utils.slugify("My Image (1).png") == "my-image-1-.png"
    assert " " not in io_utils.slugify("weird name here")
    assert io_utils.slugify("///") == "item"
    assert len(io_utils.slugify("x" * 200)) <= 60


def test_unique_path(tmp_path):
    target = tmp_path / "a.txt"
    assert io_utils.unique_path(target) == target
    target.write_text("x")
    assert io_utils.unique_path(target).name == "a-1.txt"


def test_atomic_write_and_json_roundtrip(tmp_path):
    path = io_utils.save_json(tmp_path / "nested" / "data.json", {"a": 1})
    assert io_utils.load_json(path) == {"a": 1}
    assert not list(path.parent.glob("*.tmp"))


def test_jsonl_append_and_read(tmp_path):
    path = tmp_path / "records.jsonl"
    io_utils.append_jsonl(path, {"image": "a.png"})
    io_utils.append_jsonl(path, {"image": "b.png"})
    assert [r["image"] for r in io_utils.read_jsonl(path)] == ["a.png", "b.png"]


def test_read_jsonl_skips_bad_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"image": "a.png"}\nnot json\n{"image": "b.png"}\n')
    assert len(list(io_utils.read_jsonl(path))) == 2


def test_csv_roundtrip(tmp_path):
    rows = [{"image": "a.png", "prediction": "FAKE"}]
    path = io_utils.save_csv(tmp_path / "s.csv", rows)
    assert io_utils.read_csv(path) == rows


def test_iter_image_paths_expands_directories(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "sub" / "b.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignored")

    found = io_utils.iter_image_paths([tmp_path])
    assert [p.name for p in found] == ["a.png", "b.jpg"]


def test_iter_image_paths_deduplicates(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    assert len(io_utils.iter_image_paths([image, image, tmp_path])) == 1


def test_load_image_errors_are_wrapped(tmp_path):
    with pytest.raises(ImageLoadError):
        io_utils.load_image(tmp_path / "missing.png")

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"definitely not a png")
    with pytest.raises(ImageLoadError):
        io_utils.load_image(broken)


def test_chunked():
    assert list(io_utils.chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError):
        list(io_utils.chunked([1], 0))


def test_human_bytes():
    assert io_utils.human_bytes(512) == "512.0B"
    assert io_utils.human_bytes(1536) == "1.5KB"


# ---- report --------------------------------------------------------------- #


def test_report_html_is_self_contained(tmp_path):
    path = report.write_report([make_result(), make_result("b.png", fake=False)], tmp_path / "r.html")
    html = path.read_text()

    assert html.startswith("<!doctype html>")
    assert "FAKE" in html and "REAL" in html
    assert "Too smooth to be real." in html
    assert "Artificial smoothness" in html
    # No external assets: everything inline.
    assert "http://" not in html and "https://" not in html
    assert "prefers-color-scheme" in html


def test_report_escapes_untrusted_text(tmp_path):
    """Model-generated text is untrusted input and must never render as markup."""
    result = make_result("evil.png")
    result.explanation = Explanation(text="<script>alert(1)</script>", backend="qwen2vl")
    result.artifacts.matches[0].descriptor = "<img src=x onerror=alert(1)>"

    html = report.write_report([result], tmp_path / "r.html").read_text()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "onerror=alert(1)>" not in html


def test_report_renders_error_results(tmp_path):
    html = report.write_report([make_result(error="boom")], tmp_path / "r.html").read_text()
    assert "ERROR" in html and "boom" in html


def test_load_results_from_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"
    io_utils.append_jsonl(path, make_result().to_dict(include_config=False))
    loaded = report.load_results([path])
    assert len(loaded) == 1
    assert loaded[0].detection.prediction == "FAKE"


def test_load_results_from_json_file(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(make_result().to_dict()))
    assert report.load_results([path])[0].description == "Too smooth to be real."


# ---- batch bookkeeping ---------------------------------------------------- #


def test_completed_images(tmp_path):
    path = tmp_path / "results.jsonl"
    assert completed_images(path) == set()
    io_utils.append_jsonl(path, {"image": "a.png"})
    assert completed_images(path) == {"a.png"}


def test_aggregate_stats():
    stats = aggregate_stats([make_result(), make_result("b.png", fake=False), make_result(error="x")])
    assert stats["verdicts"]["FAKE"] == 1
    assert stats["verdicts"]["REAL"] == 1
    assert stats["verdicts"]["ERROR"] == 1
    assert stats["top_artifacts"]["Artificial smoothness"] == 2
