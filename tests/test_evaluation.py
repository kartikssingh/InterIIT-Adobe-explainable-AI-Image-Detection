"""Metric implementations (checked against known closed-form answers)."""

from __future__ import annotations

import math

from aidetect.evaluation import detection as det
from aidetect.evaluation import description as desc


# ---- detection ------------------------------------------------------------ #


def test_confusion_matrix():
    counts = det.confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0])
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_classification_metrics_perfect():
    metrics = det.classification_metrics([1, 1, 0, 0], [1, 1, 0, 0])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mcc"] == 1.0


def test_classification_metrics_values():
    metrics = det.classification_metrics([1, 1, 1, 0], [1, 1, 0, 1])
    assert math.isclose(metrics["precision"], 2 / 3)
    assert math.isclose(metrics["recall"], 2 / 3)
    assert math.isclose(metrics["f1"], 2 / 3)
    assert math.isclose(metrics["accuracy"], 0.5)


def test_roc_auc_perfect_and_inverted():
    assert det.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert det.roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_roc_auc_with_ties_is_one_half():
    assert det.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_roc_auc_single_class_is_nan():
    assert math.isnan(det.roc_auc([1, 1, 1], [0.2, 0.5, 0.9]))


def test_average_precision_perfect():
    assert math.isclose(det.average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)


def test_threshold_sweep_finds_a_better_cutoff():
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40]  # separable below 0.5
    best_threshold, metrics = det.threshold_sweep(y_true, scores)
    assert 0.20 < best_threshold <= 0.30
    assert metrics["f1"] == 1.0
    # The default 0.5 threshold would call everything REAL.
    assert det.classification_metrics(y_true, [int(s >= 0.5) for s in scores])["f1"] == 0.0


def test_calibration_error_range():
    result = det.expected_calibration_error([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert 0.0 <= result["ece"] <= 1.0


def test_evaluate_detection_report_shape():
    report = det.evaluate_detection([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert report["metrics"]["accuracy"] == 1.0
    assert report["roc_auc"] == 1.0
    assert "best_threshold" in report
    assert report["positives"] == 2 and report["negatives"] == 2


def test_normalise_label_variants():
    for value in (1, "1", "FAKE", "fake", "true", "synthetic"):
        assert det.normalise_label(value) == 1
    for value in (0, "0", "REAL", "real", "false", "authentic"):
        assert det.normalise_label(value) == 0


def test_load_labels_from_csv(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("image,label\na.png,fake\nb.png,real\n")
    assert det.load_labels(str(path)) == {"a.png": 1, "b.png": 0}


def test_load_labels_from_json_mapping(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text('{"a.png": 1, "b.png": 0}')
    assert det.load_labels(str(path)) == {"a.png": 1, "b.png": 0}


# ---- descriptions --------------------------------------------------------- #


def test_lcs_and_rouge_identity():
    assert desc.lcs_length(["a", "b", "c"], ["a", "x", "c"]) == 2
    scores = desc.rouge_l("the cat sat", "the cat sat")
    assert scores["f1"] == 1.0


def test_rouge_l_no_overlap():
    assert desc.rouge_l("alpha beta", "gamma delta")["f1"] == 0.0


def test_bleu_ranges():
    assert desc.bleu("the cat sat on the mat", "the cat sat on the mat") > 0.5
    assert desc.bleu("", "reference text") == 0.0


def test_distinct_n_detects_templated_output():
    templated = ["the same phrase here"] * 5
    varied = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    assert desc.distinct_n(varied, 2) > desc.distinct_n(templated, 2)


def test_artifact_grounding():
    text = "Surfaces show artificial smoothness and shadows point in conflicting directions."
    grounded = desc.artifact_grounding(text, ["Artificial smoothness", "Inconsistent shadow directions"])
    assert grounded == 1.0
    assert desc.artifact_grounding(text, ["Incorrect wheel geometry"]) == 0.0


def test_length_stats():
    stats = desc.length_stats(["one two three", "one two"])
    assert stats["max"] == 3.0 and stats["min"] == 2.0


def test_evaluate_descriptions_flags_budget_violations():
    long_text = " ".join(["word"] * 60)
    report = desc.evaluate_descriptions([long_text, "short one"], max_words=50)
    assert report["count"] == 2
    assert report["over_word_budget"] == 1
