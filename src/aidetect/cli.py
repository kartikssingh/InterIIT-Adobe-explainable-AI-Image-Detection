"""``aidetect`` — one command for the whole project.

    aidetect analyze image.jpg --backend qwen2vl
    aidetect detect data/test --json
    aidetect batch data/test -o runs/test --resume --submission
    aidetect artifacts crop.png --top-k 10
    aidetect compare image.jpg
    aidetect evaluate runs/test/results.jsonl --labels labels.csv
    aidetect report runs/test -o report.html
    aidetect serve --port 8000
    aidetect taxonomy --format table
    aidetect info

Heavy imports live inside the command handlers, so ``aidetect --help`` is
instant and never touches torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import AppConfig, discover_config_file
from .exceptions import AidetectError
from .logging_utils import configure_logging, get_logger, verbosity_to_level
from .version import __version__

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", help="Path to a YAML/JSON config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="Override any config key, e.g. --set detector.decision_threshold=0.6",
    )
    parser.add_argument("--device", help="auto | cpu | cuda | cuda:0 | mps")
    parser.add_argument("-o", "--output-dir", help="Where to write outputs")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    parser.add_argument("-q", "--quiet", action="store_true", help="Errors only")
    parser.add_argument("--log-file", help="Also write logs to this file")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", "--model", dest="checkpoint", help="Detector checkpoint (.pt)")
    parser.add_argument("--threshold", type=float, help="FAKE probability cut-off (default 0.5)")
    parser.add_argument("--tta", action="store_true", help="Average with the horizontal mirror")
    parser.add_argument(
        "--explain-method",
        choices=["gradcam", "gradcam++", "rollout", "occlusion"],
        help="Saliency method",
    )


def _add_stage23_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["rule_based", "moondream", "qwen2vl"],
        help="Stage 3 description backend",
    )
    parser.add_argument("--top-k", type=int, help="Artifacts to report (Stage 2)")
    parser.add_argument("--max-words", type=int, help="Word budget for the description")
    parser.add_argument(
        "--explain-real",
        action="store_true",
        help="Run Stages 2/3 even when the image is classified REAL",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aidetect",
        description="Detect AI-generated images, localise their artifacts and explain them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"aidetect {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # analyze -------------------------------------------------------------- #
    analyze = subparsers.add_parser("analyze", help="Full pipeline on one or more images")
    analyze.add_argument("images", nargs="+", help="Image files, directories or globs")
    analyze.add_argument("--no-visuals", action="store_true", help="Skip writing PNG outputs")
    analyze.add_argument(
        "--save-outputs",
        help="Comma-separated subset of raw,overlay,crop,annotated,panel",
    )
    _add_model_flags(analyze)
    _add_stage23_flags(analyze)
    _add_common(analyze)

    # detect --------------------------------------------------------------- #
    detect = subparsers.add_parser("detect", help="Stage 1 only — fast REAL/FAKE verdict")
    detect.add_argument("images", nargs="+")
    detect.add_argument("--batch-size", type=int, help="Images per forward pass")
    _add_model_flags(detect)
    _add_common(detect)

    # batch ---------------------------------------------------------------- #
    batch = subparsers.add_parser("batch", help="Full pipeline over a folder, resumable")
    batch.add_argument("images", nargs="+")
    batch.add_argument("--resume", action="store_true", help="Skip images already in results.jsonl")
    batch.add_argument("--no-visuals", action="store_true")
    batch.add_argument("--submission", action="store_true", help="Also write Task 1/2 submissions")
    batch.add_argument("--report", action="store_true", help="Also write an HTML report")
    batch.add_argument("--limit", type=int, help="Only process the first N images")
    _add_model_flags(batch)
    _add_stage23_flags(batch)
    _add_common(batch)

    # artifacts ------------------------------------------------------------ #
    artifacts = subparsers.add_parser("artifacts", help="Stage 2 only — zero-shot artifact scores")
    artifacts.add_argument("images", nargs="+")
    artifacts.add_argument("--top-k", type=int, help="How many artifacts to show")
    artifacts.add_argument("--all", action="store_true", help="Show all 70 scores")
    artifacts.add_argument(
        "--score-mode", choices=["raw", "minmax", "softmax", "zscore"], help="Score normalisation"
    )
    _add_common(artifacts)

    # describe ------------------------------------------------------------- #
    describe = subparsers.add_parser("describe", help="Stage 3 — explain a crop")
    describe.add_argument("images", nargs="+")
    describe.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip Stage 2 (no SigLIP download); describe from the image alone",
    )
    _add_stage23_flags(describe)
    _add_common(describe)

    # compare -------------------------------------------------------------- #
    compare = subparsers.add_parser("compare", help="Compare two checkpoints on the same image")
    compare.add_argument("images", nargs="+")
    compare.add_argument("--checkpoints", nargs="+", help="Checkpoints to compare (default: adv + original)")
    _add_common(compare)

    # evaluate ------------------------------------------------------------- #
    evaluate = subparsers.add_parser("evaluate", help="Metrics for a finished run")
    evaluate.add_argument("results", nargs="+", help="results.jsonl / result.json / run directory")
    evaluate.add_argument("--labels", help="CSV or JSON of ground-truth labels")
    evaluate.add_argument("--image-key", default="image")
    evaluate.add_argument("--label-key", default="label")
    evaluate.add_argument("--threshold", type=float, default=0.5)
    _add_common(evaluate)

    # report --------------------------------------------------------------- #
    report = subparsers.add_parser("report", help="Build a standalone HTML report")
    report.add_argument("results", nargs="+", help="Run directories or result JSON/JSONL files")
    report.add_argument("--title", default="AI Image Detection Report")
    _add_common(report)

    # submission ----------------------------------------------------------- #
    submission = subparsers.add_parser("submission", help="Write Task 1/2 submission files")
    submission.add_argument("results", nargs="+")
    submission.add_argument("--max-artifacts", type=int, default=3)
    submission.add_argument("--max-words", type=int, default=50)
    _add_common(submission)

    # serve ---------------------------------------------------------------- #
    serve = subparsers.add_parser("serve", help="Run the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    _add_model_flags(serve)
    _add_stage23_flags(serve)
    _add_common(serve)

    # taxonomy ------------------------------------------------------------- #
    taxonomy_cmd = subparsers.add_parser("taxonomy", help="List the 70 artifacts and their families")
    taxonomy_cmd.add_argument("--format", choices=["table", "json", "csv"], default="table")
    taxonomy_cmd.add_argument("--category", help="Filter by category name or key")
    _add_common(taxonomy_cmd)

    # info ----------------------------------------------------------------- #
    info = subparsers.add_parser("info", help="Environment, config and model diagnostics")
    info.add_argument("--check-models", action="store_true", help="Try loading the detector")
    _add_model_flags(info)
    _add_common(info)

    return parser


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #


def config_from_args(args: argparse.Namespace) -> AppConfig:
    """Layer CLI flags on top of file/env configuration."""
    config_path = getattr(args, "config", None) or discover_config_file()
    config = AppConfig.load(config_path, getattr(args, "overrides", None) or [])

    mapping = {
        "device": "runtime.device",
        "output_dir": "runtime.output_dir",
        "threshold": "detector.decision_threshold",
        "backend": "describe.backend",
        "top_k": "artifacts.top_k",
        "max_words": "describe.max_words",
        "batch_size": "detector.batch_size",
        "explain_method": "explain.method",
        "score_mode": "artifacts.score_mode",
    }
    from .config import set_by_path

    for attribute, dotted in mapping.items():
        value = getattr(args, attribute, None)
        if value is not None:
            set_by_path(config, dotted, value)

    if getattr(args, "tta", False):
        config.detector.tta_hflip = True
    if getattr(args, "explain_real", False):
        config.runtime.explain_real_images = True
    if getattr(args, "save_outputs", None):
        config.explain.save_outputs = [
            part.strip() for part in args.save_outputs.split(",") if part.strip()
        ]

    config.runtime.log_level = verbosity_to_level(args.verbose, args.quiet)
    if getattr(args, "log_file", None):
        config.runtime.log_file = args.log_file

    return config.validate()


def _emit(payload: Any, as_json: bool, text: str = "") -> None:
    """Print JSON or human text depending on ``--json``."""
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    elif text:
        print(text)


def _expand_images(patterns: Sequence[str]) -> List[Path]:
    from .io_utils import iter_image_paths

    paths = iter_image_paths(patterns)
    if not paths:
        raise AidetectError(f"No images found for: {' '.join(patterns)}")
    return paths


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_analyze(args: argparse.Namespace) -> int:
    from .pipeline import AnalysisPipeline

    config = config_from_args(args)
    images = _expand_images(args.images)

    with AnalysisPipeline(config, checkpoint=args.checkpoint) as pipeline:
        results = list(
            pipeline.analyze_many(
                images,
                save_visuals=not args.no_visuals,
                output_dir=config.runtime.output_dir,
            )
        )

    from .io_utils import save_json, slugify

    if args.json:
        print(json.dumps([r.to_dict(include_config=False) for r in results], indent=2, default=str))
    else:
        for result in results:
            print(format_result(result))

    # Same folder the visual outputs went to, so a result travels with its images.
    out_dir = Path(config.runtime.output_dir)
    for result in results:
        if result.ok:
            stem = slugify(Path(result.image_path).stem)
            save_json(out_dir / stem / "result.json", result.to_dict())

    return EXIT_OK if all(r.ok for r in results) else EXIT_ERROR


def cmd_detect(args: argparse.Namespace) -> int:
    from .core.detector import Detector

    config = config_from_args(args)
    images = _expand_images(args.images)

    detector = Detector(config, checkpoint=args.checkpoint)
    results = detector.predict_batch([str(p) for p in images], [str(p) for p in images])

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
    else:
        width = max((len(Path(r.image_path).name) for r in results), default=10)
        for result in results:
            print(
                f"{Path(result.image_path).name:<{width}}  {result.prediction:<5} "
                f"{result.confidence:6.2f}%  (p_fake={result.fake_prob:.4f})"
            )
        fake = sum(1 for r in results if r.is_fake)
        print(f"\n{len(results)} image(s): {fake} FAKE, {len(results) - fake} REAL")

    return EXIT_OK


def cmd_batch(args: argparse.Namespace) -> int:
    from .batch import aggregate_stats, run_batch
    from .pipeline import AnalysisPipeline

    config = config_from_args(args)
    images = _expand_images(args.images)
    if args.limit:
        images = images[: args.limit]

    out_dir = Path(config.runtime.output_dir)
    with AnalysisPipeline(config, checkpoint=args.checkpoint) as pipeline:
        results, summary = run_batch(
            pipeline,
            images,
            output_dir=out_dir,
            save_visuals=not args.no_visuals,
            resume=args.resume,
        )

    stats = aggregate_stats(results)

    if args.submission:
        from .submission import write_task1, write_task2

        write_task1(results, out_dir / "task1_submission.json")
        write_task2(results, out_dir / "task2_submission.json", max_words=config.describe.max_words)

    if args.report:
        from .report import write_report

        write_report(results, out_dir / "report.html", subtitle=f"{len(results)} images")

    payload = {"summary": summary.to_dict(), "stats": stats, "output_dir": str(out_dir)}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_batch_summary(summary, stats, out_dir))

    return EXIT_OK if summary.failed == 0 else EXIT_ERROR


def cmd_artifacts(args: argparse.Namespace) -> int:
    from .artifacts.classifier import ArtifactClassifier

    config = config_from_args(args)
    images = _expand_images(args.images)

    classifier = ArtifactClassifier(config)
    payload: List[Dict[str, Any]] = []

    for path in images:
        report = classifier.classify(str(path), return_all=args.all)
        payload.append({"image": str(path), **report.to_dict()})
        if not args.json:
            print(f"\n{path.name}")
            print(format_artifact_report(report))

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    return EXIT_OK


def cmd_describe(args: argparse.Namespace) -> int:
    from .describe.generator import DescriptionGenerator
    from .io_utils import load_image

    config = config_from_args(args)
    images = _expand_images(args.images)

    classifier = None
    if not args.no_artifacts:
        from .artifacts.classifier import ArtifactClassifier

        classifier = ArtifactClassifier(config)

    generator = DescriptionGenerator(config)
    payload: List[Dict[str, Any]] = []

    for path in images:
        image = load_image(path)
        report = classifier.classify(image) if classifier else None
        explanation = generator.generate(image=image, report=report)
        payload.append({"image": str(path), **explanation.to_dict()})
        if not args.json:
            print(f"\n{path.name} [{explanation.backend}]\n  {explanation.text}")

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    from .core.detector import Detector

    config = config_from_args(args)
    images = _expand_images(args.images)
    checkpoints = args.checkpoints or [
        config.detector.checkpoint,
        *config.detector.fallback_checkpoints,
    ]

    rows: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        if not Path(checkpoint).is_file():
            logger.warning("Skipping missing checkpoint %s", checkpoint)
            continue
        detector = Detector(config, checkpoint=checkpoint)
        for result in detector.predict_batch([str(p) for p in images], [str(p) for p in images]):
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "kind": detector.checkpoint_info.kind,
                    "image": Path(result.image_path).name,
                    "prediction": result.prediction,
                    "confidence": round(result.confidence, 2),
                    "fake_prob": round(result.fake_prob, 6),
                }
            )
        detector.unload()

    if not rows:
        raise AidetectError("No usable checkpoints found to compare")

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        print(format_comparison(rows))
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_descriptions, evaluate_detection, load_labels
    from .report import load_results

    results = load_results(args.results)
    if not results:
        raise AidetectError(f"No results found in: {' '.join(args.results)}")

    payload: Dict[str, Any] = {"count": len(results)}

    descriptions = [r.description for r in results if r.description]
    descriptor_lists = [
        r.artifacts.descriptors(3) if r.artifacts else [] for r in results if r.description
    ]
    if descriptions:
        payload["descriptions"] = evaluate_descriptions(
            descriptions, descriptor_lists=descriptor_lists
        )

    if args.labels:
        labels = load_labels(args.labels, args.image_key, args.label_key)
        y_true: List[int] = []
        scores: List[float] = []
        missing = 0
        for result in results:
            name = Path(result.image_path).name
            label = labels.get(name, labels.get(result.image_path))
            if label is None:
                missing += 1
                continue
            y_true.append(label)
            scores.append(result.detection.fake_prob)
        if not y_true:
            raise AidetectError("No result filenames matched the label file")
        payload["detection"] = evaluate_detection(y_true, scores, threshold=args.threshold)
        payload["unlabelled_images"] = missing

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_evaluation(payload))
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    from .report import load_results, write_report

    config = config_from_args(args)
    results = load_results(args.results)
    if not results:
        raise AidetectError(f"No results found in: {' '.join(args.results)}")

    destination = Path(args.output_dir) if args.output_dir else Path(config.runtime.output_dir) / "report.html"
    if destination.is_dir():
        destination = destination / "report.html"

    path = write_report(results, destination, title=args.title)
    _emit({"report": str(path), "images": len(results)}, args.json, f"Report written to {path}")
    return EXIT_OK


def cmd_submission(args: argparse.Namespace) -> int:
    from .report import load_results
    from .submission import validate_task2, task2_records, write_task1, write_task2

    config = config_from_args(args)
    results = load_results(args.results)
    if not results:
        raise AidetectError(f"No results found in: {' '.join(args.results)}")

    out_dir = Path(config.runtime.output_dir)
    task1 = write_task1(results, out_dir / "task1_submission.json")
    task2 = write_task2(
        results,
        out_dir / "task2_submission.json",
        max_artifacts=args.max_artifacts,
        max_words=args.max_words,
    )
    problems = validate_task2(
        task2_records(results, max_artifacts=args.max_artifacts, max_words=args.max_words),
        max_words=args.max_words,
    )

    payload = {"task1": str(task1), "task2": str(task2), "problems": problems}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"Task 1 -> {task1}\nTask 2 -> {task2}")
        if problems:
            print(f"\n{len(problems)} validation problem(s):")
            for problem in problems[:20]:
                print(f"  - {problem}")
        else:
            print("Validation: OK")
    return EXIT_OK if not problems else EXIT_ERROR


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    config = config_from_args(args)
    serve(config, host=args.host, port=args.port, checkpoint=args.checkpoint, reload=args.reload)
    return EXIT_OK


def cmd_taxonomy(args: argparse.Namespace) -> int:
    from .artifacts import taxonomy

    rows = taxonomy.taxonomy_table()
    if args.category:
        needle = args.category.strip().lower()
        rows = [
            row
            for row in rows
            if needle in row["category"].lower()
            or needle == taxonomy.category_key_of(row["descriptor"])
        ]

    if args.format == "json" or args.json:
        print(json.dumps(rows, indent=2))
    elif args.format == "csv":
        print("id,descriptor,category")
        for row in rows:
            print(f'{row["id"]},"{row["descriptor"]}","{row["category"]}"')
    else:
        by_category: Dict[str, List[str]] = {}
        for row in rows:
            by_category.setdefault(row["category"], []).append(row["descriptor"])
        for category, descriptors in by_category.items():
            print(f"\n{category} ({len(descriptors)})")
            print("-" * (len(category) + 6))
            for descriptor in descriptors:
                print(f"  • {descriptor}")
        print(f"\n{len(rows)} artifact(s) across {len(by_category)} categor(ies)")
    return EXIT_OK


def cmd_info(args: argparse.Namespace) -> int:
    from .describe.base import available_backends
    from .device import device_report
    from .artifacts import taxonomy

    config = config_from_args(args)
    payload: Dict[str, Any] = {
        "version": __version__,
        "environment": device_report(),
        "description_backends": available_backends(),
        "artifacts": len(taxonomy.OFFICIAL_ARTIFACTS),
        "categories": len(taxonomy.CATEGORIES),
        "config": config.to_dict(),
    }

    checkpoints: Dict[str, Any] = {}
    for candidate in [config.detector.checkpoint, *config.detector.fallback_checkpoints]:
        path = Path(candidate)
        checkpoints[candidate] = (
            {"exists": True, "size_mb": round(path.stat().st_size / 1024**2, 1)}
            if path.is_file()
            else {"exists": False}
        )
    payload["checkpoints"] = checkpoints

    if args.check_models:
        from .core.detector import Detector

        try:
            detector = Detector(config, checkpoint=args.checkpoint)
            payload["detector"] = detector.describe()
            detector.unload()
        except AidetectError as exc:
            payload["detector_error"] = str(exc)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_info(payload))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Human-readable formatting
# --------------------------------------------------------------------------- #


def _bar(value: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return "█" * filled + "░" * (width - filled)


def format_result(result) -> str:
    detection = result.detection
    lines = [
        "",
        "─" * 62,
        f"  {Path(result.image_path).name}",
        "─" * 62,
    ]
    if not result.ok:
        lines.append(f"  ERROR: {result.error}")
        return "\n".join(lines)

    lines += [
        f"  Verdict     : {detection.prediction}  ({detection.certainty_label()} certainty)",
        f"  Confidence  : {detection.confidence:.2f}%",
        f"  P(FAKE)     : {detection.fake_prob:.4f}   P(REAL): {detection.real_prob:.4f}",
        f"  Checkpoint  : {detection.model_type} — {Path(detection.model_path).name}",
    ]

    if result.regions:
        lines.append("  Regions     :")
        for region in result.regions:
            lines.append(
                f"    #{region.rank} {region.position_phrase():<14} "
                f"{region.width}×{region.height}px  score {region.score:.2f}"
            )

    if result.artifacts and result.artifacts.matches:
        lines.append("  Artifacts   :")
        for match in result.artifacts.matches:
            lines.append(
                f"    {_bar(match.score)}  {match.score:.3f}  {match.descriptor}"
            )
        lines.append("  Families    :")
        for category, score in list(result.artifacts.category_scores.items())[:5]:
            lines.append(f"    {_bar(score)}  {score:.3f}  {category}")

    if result.description:
        lines += ["", f"  Explanation ({result.explanation.backend}):", f"    {result.description}"]

    if result.saved_files:
        lines.append("  Files       :")
        for label, path in result.saved_files.items():
            lines.append(f"    {label:<10} {path}")

    if result.warnings:
        lines.append("  Warnings    :")
        lines.extend(f"    ! {warning}" for warning in result.warnings)

    lines.append(f"  Timing      : {result.timings.get('total_ms', 0):.0f} ms total")
    return "\n".join(lines)


def format_artifact_report(report) -> str:
    lines = []
    for match in report.matches:
        lines.append(
            f"  {match.rank:>2}. {_bar(match.score)} {match.score:.3f}  "
            f"{match.descriptor}  [{match.category}]"
        )
    if report.category_scores:
        lines.append("  Families:")
        for category, score in list(report.category_scores.items())[:8]:
            lines.append(f"      {_bar(score)} {score:.3f}  {category}")
    return "\n".join(lines)


def format_batch_summary(summary, stats: Dict[str, Any], out_dir: Path) -> str:
    lines = [
        "",
        "═" * 62,
        "  BATCH SUMMARY",
        "═" * 62,
        f"  Processed   : {summary.total}",
        f"  Succeeded   : {summary.succeeded}  (FAKE {summary.fake} / REAL {summary.real})",
        f"  Failed      : {summary.failed}",
        f"  Elapsed     : {summary.elapsed_s:.1f}s  ({summary.throughput:.2f} img/s)",
        f"  Mean conf.  : {stats.get('mean_confidence', 0)}%",
        f"  Output dir  : {out_dir}",
    ]
    top = stats.get("top_artifacts") or {}
    if top:
        lines.append("  Most frequent artifacts:")
        for name, count in list(top.items())[:8]:
            lines.append(f"    {count:>4}×  {name}")
    if summary.failures:
        lines.append("  Failures:")
        for failure in summary.failures[:10]:
            lines.append(f"    ! {failure['image']}: {failure['error']}")
    return "\n".join(lines)


def format_comparison(rows: List[Dict[str, Any]]) -> str:
    lines = ["", f"{'image':<28}{'checkpoint':<14}{'verdict':<8}{'conf':>8}{'p_fake':>10}", "-" * 68]
    for row in rows:
        lines.append(
            f"{row['image'][:27]:<28}{row['kind']:<14}{row['prediction']:<8}"
            f"{row['confidence']:>7.2f}%{row['fake_prob']:>10.4f}"
        )
    return "\n".join(lines)


def format_evaluation(payload: Dict[str, Any]) -> str:
    lines = ["", "═" * 62, "  EVALUATION", "═" * 62, f"  Results     : {payload['count']}"]

    detection = payload.get("detection")
    if detection:
        metrics = detection["metrics"]
        lines += [
            "",
            f"  Threshold   : {detection['threshold']:.3f}",
            f"  Accuracy    : {metrics['accuracy']:.4f}",
            f"  Precision   : {metrics['precision']:.4f}   Recall: {metrics['recall']:.4f}",
            f"  F1          : {metrics['f1']:.4f}   Balanced acc: {metrics['balanced_accuracy']:.4f}",
            f"  MCC         : {metrics['mcc']:.4f}",
            f"  ROC-AUC     : {detection['roc_auc']:.4f}   AP: {detection['average_precision']:.4f}",
            f"  ECE         : {detection['calibration']['ece']:.4f}",
            f"  Confusion   : TP={int(metrics['tp'])} FP={int(metrics['fp'])} "
            f"TN={int(metrics['tn'])} FN={int(metrics['fn'])}",
        ]
        best = detection.get("best_metrics")
        if best:
            lines.append(
                f"  Best thresh : {best['threshold']:.3f} -> F1 {best['f1']:.4f} "
                f"(accuracy {best['accuracy']:.4f})"
            )

    descriptions = payload.get("descriptions")
    if descriptions:
        length = descriptions["length"]
        lines += [
            "",
            f"  Descriptions: {descriptions['count']}",
            f"  Words       : mean {length['mean']:.1f}, median {length['median']:.0f}, max {length['max']:.0f}",
            f"  Distinct-2  : {descriptions['distinct_2']:.3f}  (higher = less templated)",
            f"  Over budget : {descriptions['over_word_budget']}",
        ]
        if "artifact_grounding" in descriptions:
            lines.append(f"  Grounding   : {descriptions['artifact_grounding']:.3f}")
    return "\n".join(lines)


def format_info(payload: Dict[str, Any]) -> str:
    env = payload["environment"]
    lines = [
        "",
        "═" * 62,
        f"  aidetect {payload['version']}",
        "═" * 62,
        f"  Python      : {env.get('python')}",
        f"  Platform    : {env.get('platform')}",
        f"  torch       : {env.get('torch')}   timm: {env.get('timm')}",
        f"  transformers: {env.get('transformers')}   numpy: {env.get('numpy')}",
        f"  CUDA        : {env.get('cuda_available')} ({env.get('cuda_version', 'n/a')})",
    ]
    for gpu in env.get("gpus", []) or []:
        lines.append(f"    GPU {gpu['index']}: {gpu['name']} — {gpu['total_memory_gb']} GB")

    lines.append("  Checkpoints :")
    for path, status in payload["checkpoints"].items():
        state = f"{status['size_mb']} MB" if status["exists"] else "missing"
        lines.append(f"    {'✓' if status['exists'] else '✗'} {path} ({state})")

    lines.append("  Stage 3 backends:")
    for name, available in payload["description_backends"].items():
        lines.append(f"    {'✓' if available else '✗'} {name}")

    lines.append(
        f"  Taxonomy    : {payload['artifacts']} artifacts / {payload['categories']} families"
    )
    if "detector" in payload:
        detector = payload["detector"]
        lines += [
            "  Detector    :",
            f"    device {detector['device']} · dtype {detector['dtype']}",
            f"    params {detector['parameters']['total']:,}",
        ]
    if "detector_error" in payload:
        lines.append(f"  Detector    : ERROR — {payload['detector_error']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

_COMMANDS = {
    "analyze": cmd_analyze,
    "detect": cmd_detect,
    "batch": cmd_batch,
    "artifacts": cmd_artifacts,
    "describe": cmd_describe,
    "compare": cmd_compare,
    "evaluate": cmd_evaluate,
    "report": cmd_report,
    "submission": cmd_submission,
    "serve": cmd_serve,
    "taxonomy": cmd_taxonomy,
    "info": cmd_info,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return EXIT_USAGE

    configure_logging(
        level=verbosity_to_level(args.verbose, args.quiet),
        log_file=getattr(args, "log_file", None),
        show_name=args.verbose >= 2,
    )

    handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except AidetectError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.warning("Interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level guard
        logger.exception("Unexpected failure: %s", exc)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
