"""
Task 2 Pipeline — Complete Runner
=====================================
Combines Stage 2 (artifact classification) and Stage 3 (description generation).
Accepts the crop image from Stage 1 (localization) as input.

Can also be run standalone on a full image (skips localization, uses full image as crop).

Usage:
    # As a module:
    from pipeline import Task2Pipeline
    pipeline = Task2Pipeline()
    result = pipeline.run(crop_image)

    # CLI:
    python pipeline.py --image path/to/crop.jpg --backend moondream
"""

import os
import json
import logging
import argparse
from PIL import Image
from typing import Union

from stage2.artifact_classifier import ArtifactClassifier
from stage3.description_generator import DescriptionGenerator

logger = logging.getLogger(__name__)


def limit_words(text: str, max_words: int = 50) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def artifact_explanation(artifact_name: str, description: str) -> str:
    base = description.strip()

    if not base:
        base = (
            f"The image shows signs of {artifact_name.lower()}, "
            f"with visual details that appear inconsistent with a natural photograph."
        )

    return limit_words(base, 50)


def to_task2_submission_item(
    index: int,
    result: dict,
    max_artifacts: int = 3,
) -> dict:
    explanation = {}

    for artifact in result.get("top_artifacts", [])[:max_artifacts]:
        artifact_name = artifact["descriptor"]

        explanation[artifact_name] = artifact_explanation(
            artifact_name,
            result.get("description", ""),
        )

    return {
        "index": int(index),
        "explanation": explanation,
    }


class Task2Pipeline:
    """
    End-to-end Task 2 pipeline.

    Stage 2:
        SigLIP zero-shot artifact classification

    Stage 3:
        VLM description generation
        (Moondream2 / Qwen2-VL / rule-based)
    """

    def __init__(
        self,
        vlm_backend: str = "moondream",
        top_k: int = 5,
        device: str = None,
        siglip_model: str = "google/siglip-base-patch16-224",
    ):
        self.top_k = top_k
        self.device = device

        self.classifier = ArtifactClassifier(
            model_name=siglip_model,
            device=device,
            top_k=top_k,
        )

        self.generator = DescriptionGenerator(
            backend=vlm_backend,
            device=device,
        )

    def run(
        self,
        crop_image: Union[Image.Image, str],
        return_stage2_details: bool = True,
    ) -> dict:
        """
        Run Stage 2 + Stage 3 on a cropped image region.

        Returns:
        {
            "description": str,
            "detected_categories": list,
            "top_artifacts": list,
            "category_scores": dict,
            "stage2_full": list (optional)
        }
        """

        if isinstance(crop_image, str):
            crop_image = Image.open(crop_image).convert("RGB")

        # Stage 2
        logger.info("Running Stage 2: Artifact Classification...")

        stage2_output = self.classifier.classify_with_category_summary(
            crop_image,
            top_k=self.top_k,
        )

        logger.info(
            f"Stage 2 done. Top category: "
            f"{stage2_output['detected_categories'][0] if stage2_output['detected_categories'] else 'none'}"
        )

        # Stage 3
        logger.info("Running Stage 3: Description Generation...")

        final_output = self.generator.generate_full_output(
            crop_image,
            stage2_output,
        )

        logger.info("Stage 3 done.")

        if return_stage2_details:
            final_output["stage2_full"] = stage2_output

        return final_output

    def run_batch(
        self,
        image_paths: list[str],
        output_dir: str = None,
    ) -> list[dict]:
        """
        Run pipeline on multiple images.
        """

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results = []

        for i, path in enumerate(image_paths):
            logger.info(
                f"Processing {i + 1}/{len(image_paths)}: {path}"
            )

            try:
                img = Image.open(path).convert("RGB")

                result = self.run(img)
                result["image_path"] = path

                results.append(result)

                if output_dir:
                    fname = (
                        os.path.splitext(
                            os.path.basename(path)
                        )[0]
                        + ".json"
                    )

                    out_path = os.path.join(
                        output_dir,
                        fname,
                    )

                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)

                    logger.info(f"Saved: {out_path}")

            except Exception as e:
                logger.error(f"Failed on {path}: {e}")

                results.append(
                    {
                        "image_path": path,
                        "error": str(e),
                    }
                )

        return results


# --------------------------------------------------
# CLI Entry Point
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Task 2 Pipeline: Stage 2 + Stage 3"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to crop image (Stage 1 output)",
    )

    parser.add_argument(
        "--backend",
        default="moondream",
        choices=[
            "moondream",
            "qwen2vl",
            "rule_based",
        ],
        help="VLM backend for Stage 3",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k artifacts from Stage 2",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="Image index for Task 2 submission JSON",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Save result to JSON file",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="cuda or cpu",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline = Task2Pipeline(
        vlm_backend=args.backend,
        top_k=args.top_k,
        device=args.device,
    )

    result = pipeline.run(args.image)

    submission_item = to_task2_submission_item(
        index=args.index,
        result=result,
    )

    print("\n" + "=" * 60)
    print("TASK 2 OUTPUT")
    print("=" * 60)

    print(
        f"\nDetected Categories: "
        f"{', '.join(result['detected_categories'])}"
    )

    print("\nTop Artifacts:")

    for art in result["top_artifacts"]:
        print(
            f"   [{art['score']:.3f}] "
            f"{art['descriptor']}"
        )
        print(
            f"          → Category: "
            f"{art['category']}"
        )

    print(f"\nDescription:\n{result['description']}")

    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                [submission_item],
                f,
                indent=2,
            )

        print(
            f"\nTask 2 submission result saved to: "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()