"""
Batch runner for mid-term Task 2 submission.

This intentionally uses the full image as the Stage 2/3 input, because Stage 1
localization is being ignored for the current mid-term submission path.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from pipeline import Task2Pipeline, to_task2_submission_item


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_index(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if not match:
        raise ValueError(f"Cannot infer numeric index from filename: {path.name}")
    return int(match.group(0))


def collect_images(input_dir: Path) -> list[Path]:
    images = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=image_index)


def main():
    parser = argparse.ArgumentParser(description="Create Task 2 JSON using full images.")
    parser.add_argument("--input-dir", required=True, help="Folder containing fake images")
    parser.add_argument("--output", required=True, help="Output JSON path, e.g. team_11_task2.json")
    parser.add_argument("--raw-output", default=None, help="Optional raw predictions JSON for evaluate.py")
    parser.add_argument(
        "--backend",
        default="rule_based",
        choices=["rule_based", "moondream", "qwen2vl"],
        help="Stage 3 backend",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of artifacts to include per image")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional small test limit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    images = collect_images(input_dir)
    if args.limit:
        images = images[:args.limit]

    pipeline = Task2Pipeline(
        vlm_backend=args.backend,
        top_k=args.top_k,
        device=args.device,
    )

    submission = []
    raw_predictions = []
    for i, image_path in enumerate(images, start=1):
        logging.info("Processing %s/%s: %s", i, len(images), image_path.name)
        result = pipeline.run(str(image_path), return_stage2_details=False)
        submission.append(
            to_task2_submission_item(
                index=image_index(image_path),
                result=result,
                max_artifacts=args.top_k,
            )
        )
        raw_predictions.append(
            {
                "image_id": image_path.name,
                "detected_categories": result.get("detected_categories", []),
                "description": result.get("description", ""),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(submission, f, indent=2)

    print(f"Saved {len(submission)} Task 2 items to {output_path}")

    if args.raw_output:
        raw_output_path = Path(args.raw_output)
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_output_path.open("w") as f:
            json.dump(raw_predictions, f, indent=2)
        print(f"Saved {len(raw_predictions)} raw predictions to {raw_output_path}")


if __name__ == "__main__":
    main()
