"""
Evaluation — Stage 2 & Stage 3
================================
Stage 2 metrics: Category-level accuracy, top-k hit rate, IoU (if bounding boxes available)
Stage 3 metrics: BERT score, BLEU, ROUGE-L (content similarity — matches Adobe's eval criterion)

Run:
    python evaluate.py --pred predictions.json --gt ground_truth.json
    python evaluate.py --demo   # runs on a small synthetic test
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Stage 2 Evaluation
# ─────────────────────────────────────────────
def evaluate_stage2(
    predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    """
    Evaluate Stage 2 artifact category predictions.

    Args:
        predictions: List of {"image_id": str, "detected_categories": list[str]}
        ground_truth: List of {"image_id": str, "true_categories": list[str]}

    Returns:
        {
            "category_accuracy": float,   # exact match of primary category
            "top1_hit_rate": float,       # true category appears as top prediction
            "topk_hit_rate": float,       # true category in any of top-k predictions
            "per_category_recall": dict
        }
    """
    gt_map = {g["image_id"]: g["true_categories"] for g in ground_truth}

    top1_hits = 0
    topk_hits = 0
    exact_matches = 0
    per_cat_total = {}
    per_cat_hit = {}

    total = 0
    for pred in predictions:
        img_id = pred["image_id"]
        if img_id not in gt_map:
            continue
        true_cats = set(gt_map[img_id])
        pred_cats = pred.get("detected_categories", [])

        for c in true_cats:
            per_cat_total[c] = per_cat_total.get(c, 0) + 1

        if pred_cats:
            if pred_cats[0] in true_cats:
                top1_hits += 1
                for c in true_cats:
                    per_cat_hit[c] = per_cat_hit.get(c, 0) + 1

        if any(c in true_cats for c in pred_cats):
            topk_hits += 1

        if set(pred_cats) == true_cats:
            exact_matches += 1

        total += 1

    per_category_recall = {
        cat: per_cat_hit.get(cat, 0) / per_cat_total[cat]
        for cat in per_cat_total
    }

    return {
        "total_samples": total,
        "top1_hit_rate": top1_hits / total if total else 0.0,
        "topk_hit_rate": topk_hits / total if total else 0.0,
        "exact_set_match": exact_matches / total if total else 0.0,
        "per_category_recall": per_category_recall,
    }


# ─────────────────────────────────────────────
# Stage 3 Evaluation (Content Similarity)
# ─────────────────────────────────────────────
def evaluate_stage3_bleu_rouge(
    predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    """
    Compute BLEU and ROUGE-L between predicted and reference descriptions.
    These match Adobe's "content similarity" metric.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer as rouge_lib
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except ImportError:
        logger.warning("nltk or rouge_score not installed. Run: pip install nltk rouge-score")
        return {"error": "dependencies missing"}

    gt_map = {g["image_id"]: g["reference_description"] for g in ground_truth}
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    bleu_scores = []
    rouge_scores = []

    for pred in predictions:
        img_id = pred["image_id"]
        if img_id not in gt_map:
            continue
        pred_text = pred.get("description", "")
        ref_text = gt_map[img_id]

        # BLEU
        ref_tokens = [ref_text.lower().split()]
        pred_tokens = pred_text.lower().split()
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)

        # ROUGE-L
        scores = scorer.score(ref_text, pred_text)
        rouge_scores.append(scores["rougeL"].fmeasure)

    return {
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "num_evaluated": len(bleu_scores),
    }


def evaluate_stage3_bertscore(
    predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    """
    Compute BERTScore — semantic similarity, more meaningful than BLEU.
    Requires: pip install bert-score
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        logger.warning("bert-score not installed. Run: pip install bert-score")
        return {"error": "bert-score not installed"}

    gt_map = {g["image_id"]: g["reference_description"] for g in ground_truth}

    pred_texts = []
    ref_texts = []
    for pred in predictions:
        img_id = pred["image_id"]
        if img_id not in gt_map:
            continue
        pred_texts.append(pred.get("description", ""))
        ref_texts.append(gt_map[img_id])

    if not pred_texts:
        return {"error": "no matching IDs"}

    P, R, F1 = bert_score(pred_texts, ref_texts, lang="en", verbose=False)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
        "num_evaluated": len(pred_texts),
    }


# ─────────────────────────────────────────────
# Combined Evaluation
# ─────────────────────────────────────────────
def run_full_evaluation(pred_path: str, gt_path: str) -> dict:
    with open(pred_path) as f:
        predictions = json.load(f)
    with open(gt_path) as f:
        ground_truth = json.load(f)

    results = {}

    # Stage 2
    print("Evaluating Stage 2 (artifact classification)...")
    s2_metrics = evaluate_stage2(predictions, ground_truth)
    results["stage2"] = s2_metrics
    print(f"  Top-1 Hit Rate: {s2_metrics['top1_hit_rate']:.3f}")
    print(f"  Top-K Hit Rate: {s2_metrics['topk_hit_rate']:.3f}")

    # Stage 3
    print("\nEvaluating Stage 3 (description quality)...")
    s3_bleu_rouge = evaluate_stage3_bleu_rouge(predictions, ground_truth)
    s3_bert = evaluate_stage3_bertscore(predictions, ground_truth)
    results["stage3"] = {**s3_bleu_rouge, **s3_bert}
    if "bleu" in s3_bleu_rouge:
        print(f"  BLEU:      {s3_bleu_rouge['bleu']:.3f}")
        print(f"  ROUGE-L:   {s3_bleu_rouge['rouge_l']:.3f}")
    if "bertscore_f1" in s3_bert:
        print(f"  BERTScore F1: {s3_bert['bertscore_f1']:.3f}")

    return results


# ─────────────────────────────────────────────
# Demo mode — no ground truth needed
# ─────────────────────────────────────────────
def run_demo():
    """
    Quick demo to verify Stage 2 + Stage 3 work end-to-end
    using a generated blank test image (no real image needed).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from PIL import Image
    from pipeline import Task2Pipeline

    print("Creating dummy test image...")
    # Solid color image as a placeholder
    img = Image.new("RGB", (224, 224), color=(128, 100, 80))

    print("\nRunning pipeline (rule_based mode, no model downloads needed)...")
    pipeline = Task2Pipeline(vlm_backend="rule_based")
    result = pipeline.run(img)

    print("\n" + "=" * 60)
    print("DEMO RESULT")
    print("=" * 60)
    print(f"Categories:  {result['detected_categories']}")
    print(f"Description: {result['description']}")
    print("=" * 60)
    print("\n✓ Pipeline ran successfully in rule-based mode.")
    print("  Use --backend moondream or --backend qwen2vl for full VLM output.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 & Stage 3")
    parser.add_argument("--pred", help="Path to predictions JSON")
    parser.add_argument("--gt", help="Path to ground truth JSON")
    parser.add_argument("--demo", action="store_true", help="Run demo without GT")
    parser.add_argument("--output", help="Save metrics to JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.demo:
        run_demo()
        return

    if not args.pred or not args.gt:
        parser.error("Provide --pred and --gt, or use --demo")

    results = run_full_evaluation(args.pred, args.gt)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
