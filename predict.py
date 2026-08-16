#!/usr/bin/env python3
"""Backwards-compatible wrapper for the old ``python predict.py`` entry point.

The implementation now lives in the ``aidetect`` package
(``aidetect.core.detector`` + ``aidetect.explainability``). This shim keeps the
original command line working:

    python predict.py --image image.jpg
    python predict.py --image image.jpg --model checkpoints_adv/adv_best_model.pt

The equivalent modern command is::

    aidetect analyze image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aidetect.cli import main as cli_main  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict REAL/FAKE with saliency localisation (compatibility wrapper)"
    )
    parser.add_argument("--image", required=True, help="Path to the image to analyse")
    parser.add_argument("--model", "--checkpoint", dest="model", help="Detector checkpoint (.pt)")
    parser.add_argument(
        "--model_type",
        choices=["adversarial", "original"],
        default="adversarial",
        help="Which bundled checkpoint to prefer when --model is not given",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = args.model
    if not checkpoint and args.model_type == "original":
        checkpoint = "checkpoints/best_model.pt"

    forwarded = [
        "analyze",
        args.image,
        "--output-dir",
        args.output_dir,
        "--set",
        "explain.save_outputs=raw,overlay,crop,annotated,panel",
    ]
    if checkpoint:
        forwarded += ["--checkpoint", checkpoint]
    if args.json:
        forwarded.append("--json")

    print(
        "note: predict.py is a compatibility wrapper — "
        f"the modern command is:  aidetect {' '.join(forwarded)}",
        file=sys.stderr,
    )
    return cli_main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
