"""
main.py  —  Full Pipeline: Task 1 (Predict) + Task 2 (Artifact Classification + Description)
=============================================================================================
Takes an image, runs:
    1. predict.py           → REAL/FAKE + GradCAM crop
    2. stage_2_3/pipeline.py → Artifact classification + natural language explanation

Usage:
    python main.py                                     # uses image.jpg by default
    python main.py --image myimage.jpg
    python main.py --image myimage.jpg --backend moondream
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    "image":       "image.jpg",
    "model_path":  "checkpoints/best_model.pt",
    "backend":     "rule_based",   # rule_based | moondream | qwen2vl
    "top_k":       5,
    "stage23_dir": "stage_2_3",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Full AI image detection pipeline")
    parser.add_argument("--image",   type=str, default=CFG["image"])
    parser.add_argument("--model",   type=str, default=CFG["model_path"])
    parser.add_argument("--backend", type=str, default=CFG["backend"],
                        choices=["moondream", "qwen2vl", "rule_based"])
    parser.add_argument("--top_k",   type=int, default=CFG["top_k"])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def validate_setup(args):
    errors = []
    if not os.path.exists(args.image):
        errors.append(f"Image not found       : {args.image}")
    if not os.path.exists(args.model):
        errors.append(f"Model not found       : {args.model}")
    if not os.path.exists("predict.py"):
        errors.append("predict.py not found in current directory")
    if not os.path.exists(CFG["stage23_dir"]):
        errors.append(f"stage_2_3 dir missing : {CFG['stage23_dir']}")
    else:
        for f in ["pipeline.py", "stage2/artifact_classifier.py",
                  "stage3/description_generator.py"]:
            if not os.path.exists(os.path.join(CFG["stage23_dir"], f)):
                errors.append(f"Missing               : {CFG['stage23_dir']}/{f}")
    if errors:
        print("\n❌  Setup errors:")
        for e in errors:
            print(f"    • {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. STEP 1 — TASK 1  (predict.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_task1(args) -> dict:
    print("\n" + "="*60)
    print("  STEP 1 — Task 1: Classification + GradCAM")
    print("="*60)

    # Dynamically import predict.py
    spec = importlib.util.spec_from_file_location("predict", "predict.py")
    pred = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pred)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = pred.load_model(args.model, device)
    result = pred.predict(args.image, model, device)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. STEP 2 — TASK 2  (stage_2_3/pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_task2(task1_result: dict, args) -> dict:
    print("\n" + "="*60)
    print("  STEP 2 — Task 2: Artifact Classification + Description")
    print("="*60)

    if task1_result["prediction"] == "REAL":
        print("\n  Image is REAL — Task 2 skipped.")
        return {}

    # ── Decide which image to pass to Stage 2/3 ──
    # Try to use GradCAM crop first, but check if it's big enough
    crop_path = task1_result.get("task2", {}).get("crop_path", None)
    
    # Check if crop is valid (bigger than 28x28 — minimum for Qwen2-VL)
    crop_is_valid = False
    crop_width = 0
    crop_height = 0
    
    if crop_path and os.path.exists(str(crop_path)):
        from PIL import Image as PILImage
        try:
            w, h = PILImage.open(crop_path).size
            crop_width, crop_height = w, h
            if w >= 28 and h >= 28:
                crop_is_valid = True
        except Exception as e:
            print(f"  ⚠️  Could not read crop: {e}")

    # Use crop if valid, otherwise use original image
    if crop_is_valid:
        print(f"  Using GradCAM crop : {crop_path} ({crop_width}×{crop_height}px)")
        image_to_use = crop_path
    else:
        if crop_path:
            print(f"  ⚠️  Crop too small ({crop_width}×{crop_height}px) — using original image")
        else:
            print(f"  ⚠️  No crop found — using original image")
        image_to_use = args.image
        print(f"  Using original image: {image_to_use}")

    print(f"  Backend            : {args.backend}")
    print(f"  Top-K artifacts    : {args.top_k}")

    # ── Add stage_2_3 to sys.path so its internal imports work ──────────────
    # pipeline.py does: from stage2.artifact_classifier import ArtifactClassifier
    # This only works if stage_2_3/ is in sys.path
    stage23_abs = os.path.abspath(CFG["stage23_dir"])
    if stage23_abs not in sys.path:
        sys.path.insert(0, stage23_abs)

    # ── Dynamically import stage_2_3/pipeline.py ────────────────────────────
    pipeline_path = os.path.join(CFG["stage23_dir"], "pipeline.py")
    spec = importlib.util.spec_from_file_location("stage23_pipeline", pipeline_path)
    pipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipe)

    # ── Initialise Task2Pipeline ─────────────────────────────────────────────
    # Matches exact __init__ signature:
    #   def __init__(self, vlm_backend, top_k, device, siglip_model)
    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = pipe.Task2Pipeline(
        vlm_backend=args.backend,
        top_k=args.top_k,
        device=device_str,
    )

    # ── Run — matches exact .run() signature: ────────────────────────────────
    #   def run(self, crop_image: Union[Image.Image, str], return_stage2_details: bool)
    result = pipeline.run(
        crop_image=image_to_use,  # ← Using original image or crop
        return_stage2_details=True,
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRINT COMBINED OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_final_output(task1_result: dict, task2_result: dict) -> None:
    print("\n" + "="*60)
    print("  FINAL RESULT")
    print("="*60)

    pred = task1_result["prediction"]
    conf = task1_result["confidence"]

    icon = "🔴" if pred == "FAKE" else "🟢"
    print(f"\n  {icon}  Verdict     : {pred}")
    print(f"      Confidence  : {conf:.2f}%")
    print(f"      FAKE prob   : {task1_result['fake_prob']*100:.2f}%")
    print(f"      REAL prob   : {task1_result['real_prob']*100:.2f}%")

    if task2_result:
        print(f"\n  ── Artifact Analysis ──────────────────────────")

        # Description (Stage 3 output)
        desc = task2_result.get("description", "N/A")
        print(f"\n  Description:\n  {desc}")

        # Adobe categories
        cats = task2_result.get("detected_categories", [])
        if cats:
            print(f"\n  Adobe Categories Detected:")
            for c in cats:
                print(f"    • {c}")

        # Top artifacts with scores
        top_arts = task2_result.get("top_artifacts", [])
        if top_arts:
            print(f"\n  Top Artifacts:")
            for art in top_arts:
                if isinstance(art, dict):
                    name  = art.get("name",  art.get("artifact", str(art)))
                    score = art.get("score", art.get("confidence", 0.0))
                    print(f"    • {name:<50} {score*100:.1f}%")
                else:
                    print(f"    • {art}")

        # Per-category scores with bar chart
        cat_scores = task2_result.get("category_scores", {})
        if cat_scores:
            print(f"\n  Per-Category Scores:")
            for cat, score in cat_scores.items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"    {cat:<32} [{bar}] {score*100:.1f}%")

    # GradCAM files saved
    saved = task1_result.get("saved_files", [])
    if saved:
        print(f"\n  ── Saved GradCAM Files ────────────────────────")
        for f in saved:
            print(f"    📁 {f}")

    print("\n" + "="*60)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE RESULTS TO JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_results(task1_result: dict, task2_result: dict, image_path: str) -> None:
    stem        = Path(image_path).stem
    output_path = f"result_{stem}.json"

    combined = {
        "image": image_path,
        "task1": {
            "prediction":  task1_result["prediction"],
            "confidence":  task1_result["confidence"],
            "fake_prob":   task1_result["fake_prob"],
            "real_prob":   task1_result["real_prob"],
            "saved_files": task1_result.get("saved_files", []),
            "gradcam_info": task1_result.get("task2", {}),
        },
        "task2": task2_result if task2_result else "skipped — image is REAL",
    }

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"  Full results saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "="*60)
    print("  AI Image Detection — Full Pipeline")
    print("="*60)
    print(f"  Image   : {args.image}")
    print(f"  Model   : {args.model}")
    print(f"  Backend : {args.backend}")
    print(f"  Top-K   : {args.top_k}")

    validate_setup(args)

    task1_result = run_task1(args)
    task2_result = run_task2(task1_result, args)

    print_final_output(task1_result, task2_result)
    save_results(task1_result, task2_result, args.image)