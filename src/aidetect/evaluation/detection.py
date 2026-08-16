"""Detection metrics — accuracy, ROC-AUC, calibration, threshold sweep.

Implemented directly on numpy so evaluation runs without scikit-learn. The
AUC uses the rank (Mann–Whitney) formulation with proper tie handling, which
matches ``sklearn.metrics.roc_auc_score`` exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    """Binary confusion counts with 1 = FAKE (the positive class)."""
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    return {
        "tp": int(np.sum((truth == 1) & (pred == 1))),
        "fp": int(np.sum((truth == 0) & (pred == 1))),
        "tn": int(np.sum((truth == 0) & (pred == 0))),
        "fn": int(np.sum((truth == 1) & (pred == 0))),
    }


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Accuracy, precision, recall, F1, specificity, balanced accuracy, MCC."""
    counts = confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    denominator = float(
        np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    )
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator > 0 else 0.0

    return {
        **{k: float(v) for k, v in counts.items()},
        "support": float(total),
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": mcc,
    }


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based ROC-AUC with tie correction."""
    truth = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    positives = int(np.sum(truth == 1))
    negatives = int(np.sum(truth == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    # Average the ranks inside each group of tied scores.
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start
        while end + 1 < len(sorted_values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        if end > start:
            ranks[order[start : end + 1]] = np.mean(ranks[order[start : end + 1]])
        start = end + 1

    rank_sum = float(np.sum(ranks[truth == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    truth = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if truth.sum() == 0:
        return float("nan")

    order = np.argsort(-values, kind="mergesort")
    truth = truth[order]
    tp_cumulative = np.cumsum(truth)
    precision = tp_cumulative / np.arange(1, len(truth) + 1)
    recall = tp_cumulative / truth.sum()

    previous_recall = 0.0
    area = 0.0
    for prec, rec in zip(precision, recall):
        area += prec * (rec - previous_recall)
        previous_recall = rec
    return float(area)


def roc_curve(y_true: Sequence[int], scores: Sequence[float], points: int = 101) -> Dict[str, List[float]]:
    """Sampled ROC curve, handy for plotting in the HTML report."""
    thresholds = np.linspace(0.0, 1.0, points)
    fpr: List[float] = []
    tpr: List[float] = []
    for threshold in thresholds:
        predictions = (np.asarray(scores, dtype=float) >= threshold).astype(int)
        metrics = classification_metrics(y_true, predictions)
        tpr.append(metrics["recall"])
        fpr.append(1.0 - metrics["specificity"])
    return {"thresholds": thresholds.tolist(), "fpr": fpr, "tpr": tpr}


def threshold_sweep(
    y_true: Sequence[int],
    scores: Sequence[float],
    metric: str = "f1",
    points: int = 199,
) -> Tuple[float, Dict[str, float]]:
    """Find the decision threshold maximising ``metric``.

    Useful because the detector's default 0.5 cut-off is rarely optimal on a
    class-imbalanced validation split.
    """
    values = np.asarray(scores, dtype=float)
    best_threshold, best_metrics, best_score = 0.5, {}, -1.0

    for threshold in np.linspace(0.005, 0.995, points):
        predictions = (values >= threshold).astype(int)
        metrics = classification_metrics(y_true, predictions)
        score = metrics.get(metric, 0.0)
        if score > best_score:
            best_threshold, best_metrics, best_score = float(threshold), metrics, score

    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


def expected_calibration_error(
    y_true: Sequence[int], probabilities: Sequence[float], bins: int = 10
) -> Dict[str, float]:
    """ECE plus max calibration error — is a 90% confidence really 90% right?"""
    truth = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)

    ece = 0.0
    max_error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probs > low) & (probs <= high)
        if not mask.any():
            continue
        accuracy = float(np.mean(truth[mask] == (probs[mask] >= 0.5)))
        confidence = float(np.mean(np.maximum(probs[mask], 1.0 - probs[mask])))
        gap = abs(accuracy - confidence)
        weight = float(mask.mean())
        ece += weight * gap
        max_error = max(max_error, gap)

    return {"ece": ece, "max_calibration_error": max_error, "bins": float(bins)}


def evaluate_detection(
    y_true: Sequence[int],
    fake_probabilities: Sequence[float],
    threshold: float = 0.5,
    optimise: bool = True,
) -> Dict[str, object]:
    """Full detection report at a fixed threshold, plus the optimal one."""
    probs = np.asarray(fake_probabilities, dtype=float)
    predictions = (probs >= threshold).astype(int)

    report: Dict[str, object] = {
        "threshold": threshold,
        "metrics": classification_metrics(y_true, predictions),
        "roc_auc": roc_auc(y_true, probs),
        "average_precision": average_precision(y_true, probs),
        "calibration": expected_calibration_error(y_true, probs),
        "positives": int(np.sum(np.asarray(y_true) == 1)),
        "negatives": int(np.sum(np.asarray(y_true) == 0)),
    }

    if optimise:
        best_threshold, best_metrics = threshold_sweep(y_true, probs)
        report["best_threshold"] = best_threshold
        report["best_metrics"] = best_metrics

    return report


def load_labels(path: str, image_key: str = "image", label_key: str = "label") -> Dict[str, int]:
    """Read ground-truth labels from CSV or JSON.

    Accepted label spellings: ``1/0``, ``fake/real`` (any case), ``true/false``.
    """
    from .. import io_utils

    if str(path).lower().endswith(".json"):
        raw = io_utils.load_json(path)
        rows = raw if isinstance(raw, list) else [
            {image_key: k, label_key: v} for k, v in raw.items()
        ]
    else:
        rows = io_utils.read_csv(path)

    labels: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(image_key, "")).strip()
        if not key:
            continue
        labels[key] = normalise_label(row.get(label_key))
    return labels


def normalise_label(value: object) -> int:
    """Map assorted label spellings onto ``1`` (FAKE) / ``0`` (REAL)."""
    text = str(value).strip().lower()
    if text in ("1", "fake", "true", "ai", "generated", "synthetic"):
        return 1
    if text in ("0", "real", "false", "authentic", "natural"):
        return 0
    try:
        return 1 if float(text) >= 0.5 else 0
    except ValueError as exc:
        raise ValueError(f"Unrecognised label value: {value!r}") from exc
