"""Resumable batch processing over folders of images.

Features the old scripts lacked:

* **Resume** — results already present in the JSONL are skipped, so a run
  interrupted after 900 of 1000 images restarts at image 901.
* **Failure isolation** — one unreadable file no longer aborts the run; it is
  recorded and the batch continues.
* **Live progress** with throughput and ETA.
* **Multiple output formats** written as it goes: JSONL (streamed), per-image
  JSON, a CSV summary and an aggregate ``summary.json``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Set

from .io_utils import append_jsonl, ensure_dir, read_jsonl, save_csv, save_json, slugify
from .logging_utils import get_logger
from .types import AnalysisResult, BatchSummary, results_to_records

if TYPE_CHECKING:  # avoids importing torch just to read a summary file
    from .pipeline import AnalysisPipeline

logger = get_logger(__name__)

ProgressHook = Callable[[int, int, AnalysisResult], None]


def completed_images(jsonl_path: str | Path) -> Set[str]:
    """Image paths already present in a results JSONL."""
    path = Path(jsonl_path)
    if not path.is_file():
        return set()
    done = {str(record.get("image", "")) for record in read_jsonl(path)}
    done.discard("")
    return done


def run_batch(
    pipeline: "AnalysisPipeline",
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    save_visuals: bool = True,
    save_individual_json: bool = True,
    resume: bool = False,
    progress: Optional[ProgressHook] = None,
    log_every: int = 1,
) -> tuple[List[AnalysisResult], BatchSummary]:
    """Run the pipeline over many images, streaming results to disk."""
    out_dir = ensure_dir(output_dir)
    jsonl_path = out_dir / "results.jsonl"

    pending = [Path(p) for p in image_paths]
    if resume:
        done = completed_images(jsonl_path)
        before = len(pending)
        pending = [p for p in pending if str(p) not in done]
        if before != len(pending):
            logger.info("Resuming: %d of %d images already done", before - len(pending), before)

    summary = BatchSummary(total=len(pending))
    results: List[AnalysisResult] = []
    started = time.perf_counter()

    for index, path in enumerate(pending, start=1):
        result = _analyze_one(pipeline, path, save_visuals, out_dir)
        results.append(result)

        if result.ok:
            summary.succeeded += 1
            if result.detection.is_fake:
                summary.fake += 1
            else:
                summary.real += 1
        else:
            summary.failed += 1
            summary.failures.append({"image": result.image_path, "error": result.error or ""})

        append_jsonl(jsonl_path, result.to_dict(include_config=False))
        if save_individual_json and result.ok:
            save_json(out_dir / slugify(path.stem) / "result.json", result.to_dict())

        if progress:
            progress(index, len(pending), result)
        elif log_every and index % log_every == 0:
            _log_progress(index, len(pending), started, result)

    summary.elapsed_s = round(time.perf_counter() - started, 2)

    save_csv(out_dir / "summary.csv", results_to_records(results))
    save_json(out_dir / "summary.json", summary.to_dict())

    logger.info(
        "Batch finished: %d ok (%d FAKE / %d REAL), %d failed in %.1fs (%.2f img/s)",
        summary.succeeded,
        summary.fake,
        summary.real,
        summary.failed,
        summary.elapsed_s,
        summary.throughput,
    )
    return results, summary


def _analyze_one(
    pipeline: "AnalysisPipeline",
    path: Path,
    save_visuals: bool,
    output_dir: Path,
) -> AnalysisResult:
    from .exceptions import AidetectError
    from .types import DetectionResult

    try:
        return pipeline.analyze(path, save_visuals=save_visuals, output_dir=output_dir)
    except (AidetectError, OSError, ValueError) as exc:
        logger.error("Failed on %s: %s", path, exc)
        return AnalysisResult(
            image_path=str(path),
            detection=DetectionResult(image_path=str(path), prediction="ERROR", confidence=0.0),
            error=f"{type(exc).__name__}: {exc}",
        )


def _log_progress(index: int, total: int, started: float, result: AnalysisResult) -> None:
    elapsed = time.perf_counter() - started
    rate = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else 0.0
    logger.info(
        "[%d/%d] %s — %s (%.1f%%) | %.2f img/s | ETA %s",
        index,
        total,
        Path(result.image_path).name,
        result.detection.prediction,
        result.detection.confidence,
        rate,
        _format_eta(remaining),
    )


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def aggregate_stats(results: Iterable[AnalysisResult]) -> Dict[str, object]:
    """Corpus-level statistics: verdict mix, artifact and category frequencies."""
    verdicts: Dict[str, int] = {}
    artifacts: Dict[str, int] = {}
    categories: Dict[str, int] = {}
    confidences: List[float] = []

    for result in results:
        if not result.ok:
            verdicts["ERROR"] = verdicts.get("ERROR", 0) + 1
            continue
        verdicts[result.detection.prediction] = verdicts.get(result.detection.prediction, 0) + 1
        confidences.append(result.detection.confidence)
        if result.artifacts:
            for match in result.artifacts.matches[:3]:
                artifacts[match.descriptor] = artifacts.get(match.descriptor, 0) + 1
            for category in result.artifacts.categories:
                categories[category] = categories.get(category, 0) + 1

    return {
        "verdicts": verdicts,
        "mean_confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
        "top_artifacts": dict(sorted(artifacts.items(), key=lambda kv: kv[1], reverse=True)[:15]),
        "category_frequency": dict(sorted(categories.items(), key=lambda kv: kv[1], reverse=True)),
    }
