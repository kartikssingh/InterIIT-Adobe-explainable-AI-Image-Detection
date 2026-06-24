"""
Integration Example
====================
Shows exactly how Stage 1's output feeds into Stage 2 + Stage 3.
Your teammates handle Stage 1 (localization). This is the handoff point.

Stage 1 output format (what you'll receive):
    {
        "image": PIL.Image,          # Original image
        "crop": PIL.Image,           # Cropped suspicious region
        "bbox": [x1, y1, x2, y2],   # Bounding box of the crop
        "heatmap": np.ndarray        # GradCAM heatmap (optional, for viz)
    }

Your Stage 2 + Stage 3 output format:
    {
        "description": str,
        "detected_categories": list[str],
        "top_artifacts": list[dict],
        "category_scores": dict
    }
"""

from PIL import Image
import numpy as np


# ─────────────────────────────────────────────
# Simulated Stage 1 output (your teammates produce this)
# ─────────────────────────────────────────────
def simulate_stage1_output(image_path: str) -> dict:
    """
    Mimics what Stage 1 produces. Replace this with the actual
    Stage 1 module call once your teammates finish it.
    """
    original = Image.open(image_path).convert("RGB")
    w, h = original.size

    # Stage 1 would localize the artifact — we simulate a center crop
    x1, y1 = int(w * 0.25), int(h * 0.25)
    x2, y2 = int(w * 0.75), int(h * 0.75)
    crop = original.crop((x1, y1, x2, y2))

    return {
        "image": original,
        "crop": crop,
        "bbox": [x1, y1, x2, y2],
        "is_fake": True,   # Task 1 classification result
    }


# ─────────────────────────────────────────────
# Full end-to-end call
# ─────────────────────────────────────────────
def run_full_task2(image_path: str, vlm_backend: str = "moondream") -> dict:
    """
    Full Task 2 flow. Call this from the evaluation harness.

    Args:
        image_path:  Path to the input image
        vlm_backend: "moondream" | "qwen2vl" | "rule_based"

    Returns:
        Task 2 output dict
    """
    from pipeline import Task2Pipeline

    # Step 0: Get Stage 1 output (crop from localization)
    stage1 = simulate_stage1_output(image_path)

    # If Task 1 says it's real, no explanation needed
    if not stage1.get("is_fake", True):
        return {"description": "Image classified as real. No artifacts to explain.", "detected_categories": []}

    # Step 1+2: Stage 2 + Stage 3
    pipeline = Task2Pipeline(vlm_backend=vlm_backend)
    result = pipeline.run(stage1["crop"])

    # Add bbox for reference
    result["localized_region"] = stage1["bbox"]

    return result


# ─────────────────────────────────────────────
# Expected output JSON schema
# ─────────────────────────────────────────────
EXPECTED_OUTPUT_SCHEMA = {
    "description": "string — 2-3 sentences explaining visible AI artifacts",
    "detected_categories": ["list of Adobe category names, e.g. 'Unnatural Blending'"],
    "top_artifacts": [
        {
            "descriptor": "string — specific artifact descriptor",
            "category": "string — one of Adobe's 5 categories",
            "score": "float — SigLIP similarity score 0-1",
        }
    ],
    "category_scores": {
        "Unnatural Blending": "float",
        "Anatomical Inconsistencies": "float",
        "Color Treatment": "float",
        "Resolution and Detail": "float",
        "Motion Effects": "float",
    },
    "localized_region": [0, 0, 224, 224],  # [x1, y1, x2, y2] from Stage 1
}


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python integration_example.py <image_path> [backend]")
        print("Example: python integration_example.py test.jpg moondream")
        sys.exit(1)

    path = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "rule_based"
    result = run_full_task2(path, vlm_backend=backend)
    print(json.dumps(result, indent=2))
