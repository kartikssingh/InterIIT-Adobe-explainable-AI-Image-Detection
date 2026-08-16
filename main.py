#!/usr/bin/env python3
"""Backwards-compatible wrapper for the old ``python main.py`` entry point.

The full pipeline now lives in ``aidetect.pipeline.AnalysisPipeline`` and is
driven by the ``aidetect`` CLI. This shim maps the historical flags onto it:

    python main.py --image image.jpg --backend qwen2vl
    python main.py --image image.jpg --model_type both     # -> aidetect compare

Modern equivalents::

    aidetect analyze image.jpg --backend qwen2vl
    aidetect compare image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aidetect.cli import main as cli_main  # noqa: E402

DEFAULT_ADVERSARIAL = "checkpoints_adv/adv_best_model.pt"
DEFAULT_ORIGINAL = "checkpoints/best_model.pt"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Full AI image detection pipeline (compatibility wrapper)")
    parser.add_argument("--image", default="image.jpg")
    parser.add_argument("--model", "--checkpoint", dest="model", default=None)
    parser.add_argument(
        "--backend", default="rule_based", choices=["moondream", "qwen2vl", "rule_based"]
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--model_type", default="adversarial", choices=["adversarial", "original", "both"]
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.model_type == "both":
        forwarded = [
            "compare",
            args.image,
            "--checkpoints",
            DEFAULT_ADVERSARIAL,
            DEFAULT_ORIGINAL,
            "--output-dir",
            args.output_dir,
        ]
    else:
        checkpoint = args.model or (
            DEFAULT_ORIGINAL if args.model_type == "original" else DEFAULT_ADVERSARIAL
        )
        forwarded = [
            "analyze",
            args.image,
            "--backend",
            args.backend,
            "--top-k",
            str(args.top_k),
            "--checkpoint",
            checkpoint,
            "--output-dir",
            args.output_dir,
            "--set",
            "explain.save_outputs=raw,overlay,crop,annotated,panel",
        ]

    if args.json:
        forwarded.append("--json")

    print(
        "note: main.py is a compatibility wrapper — "
        f"the modern command is:  aidetect {' '.join(forwarded)}",
        file=sys.stderr,
    )
    return cli_main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
