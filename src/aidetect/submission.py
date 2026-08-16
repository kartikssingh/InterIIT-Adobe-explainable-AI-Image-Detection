"""Competition submission writers.

Task 1 wants one verdict per image; Task 2 wants, per image, a mapping of
artifact name to a short explanation (50 words or fewer). Both formats are
produced from the same :class:`~aidetect.types.AnalysisResult` list so they can
never drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .describe import postprocess
from .io_utils import save_csv, save_json
from .logging_utils import get_logger
from .types import AnalysisResult

logger = get_logger(__name__)

_INDEX_RE = re.compile(r"(\d+)")


def index_from_path(path: str | Path, fallback: int = 0) -> int:
    """Extract a numeric index from a filename such as ``image_0042.png``."""
    matches = _INDEX_RE.findall(Path(path).stem)
    return int(matches[-1]) if matches else fallback


def task1_records(
    results: Sequence[AnalysisResult],
    start_index: int = 0,
    use_filename_index: bool = True,
) -> List[Dict[str, Any]]:
    """``[{index, prediction, confidence, fake_probability}, ...]``"""
    records: List[Dict[str, Any]] = []
    for offset, result in enumerate(results):
        index = (
            index_from_path(result.image_path, start_index + offset)
            if use_filename_index
            else start_index + offset
        )
        records.append(
            {
                "index": index,
                "image": Path(result.image_path).name,
                "prediction": result.detection.prediction,
                "label": 1 if result.detection.is_fake else 0,
                "confidence": round(result.detection.confidence, 2),
                "fake_probability": round(result.detection.fake_prob, 6),
            }
        )
    return records


def task2_records(
    results: Sequence[AnalysisResult],
    max_artifacts: int = 3,
    max_words: int = 50,
    start_index: int = 0,
    use_filename_index: bool = True,
    per_artifact: Optional[Dict[str, Dict[str, str]]] = None,
    include_real: bool = False,
) -> List[Dict[str, Any]]:
    """``[{index, explanation: {artifact: text}}, ...]``

    ``per_artifact`` optionally supplies VLM-written text keyed by image path,
    otherwise the image-level description is reused for each artifact (trimmed
    to the word budget).

    Task 2 asks for artifact explanations, which only exist for images flagged
    FAKE. Images classified REAL are therefore skipped unless ``include_real``
    is set — emitting empty explanation objects for them would look like missing
    data to a grader.
    """
    records: List[Dict[str, Any]] = []

    for offset, result in enumerate(results):
        index = (
            index_from_path(result.image_path, start_index + offset)
            if use_filename_index
            else start_index + offset
        )
        if not include_real and not result.detection.is_fake:
            continue
        explanation: Dict[str, str] = {}

        custom = (per_artifact or {}).get(result.image_path, {})
        matches = result.artifacts.matches[:max_artifacts] if result.artifacts else []

        for match in matches:
            text = custom.get(match.descriptor) or result.description
            if not text:
                text = (
                    f"The region shows {match.descriptor.lower()}, inconsistent with a "
                    "natural photograph."
                )
            trimmed, _ = postprocess.limit_words(text, max_words)
            explanation[match.descriptor] = postprocess.ensure_terminal_punctuation(trimmed)

        records.append({"index": index, "explanation": explanation})

    return records


def write_task1(
    results: Sequence[AnalysisResult],
    output_path: str | Path,
    fmt: str = "json",
    **kwargs: Any,
) -> Path:
    records = task1_records(results, **kwargs)
    path = Path(output_path)
    if fmt == "csv":
        save_csv(path, records)
    else:
        save_json(path, records)
    logger.info("Wrote Task 1 submission (%d rows) -> %s", len(records), path)
    return path


def write_task2(
    results: Sequence[AnalysisResult],
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    records = task2_records(results, **kwargs)
    path = save_json(output_path, records)
    logger.info("Wrote Task 2 submission (%d rows) -> %s", len(records), path)
    return path


def validate_task2(records: Sequence[Dict[str, Any]], max_words: int = 50) -> List[str]:
    """Return a list of format problems (empty means the submission is valid)."""
    problems: List[str] = []
    seen: set[int] = set()

    for position, record in enumerate(records):
        if "index" not in record:
            problems.append(f"row {position}: missing 'index'")
            continue
        index = record["index"]
        if index in seen:
            problems.append(f"row {position}: duplicate index {index}")
        seen.add(index)

        explanation = record.get("explanation")
        if not isinstance(explanation, dict):
            problems.append(f"index {index}: 'explanation' must be an object")
            continue
        if not explanation:
            problems.append(f"index {index}: no artifacts explained")
        for artifact, text in explanation.items():
            if not isinstance(text, str) or not text.strip():
                problems.append(f"index {index}: empty explanation for {artifact!r}")
            elif len(text.split()) > max_words:
                problems.append(
                    f"index {index}: explanation for {artifact!r} exceeds "
                    f"{max_words} words ({len(text.split())})"
                )
    return problems
