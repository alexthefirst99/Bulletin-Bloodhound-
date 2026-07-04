"""Set B metrics: Groundedness, Factual Accuracy (+-1% tolerance via the
official Databricks reward.py), and Hallucination Rate."""
import argparse
import json

from common import RESULTS_DIR
from reward import score_answer


def evaluate(mode, tolerance=0.01):
    records = json.loads((RESULTS_DIR / f"generation_{mode}.json").read_text())

    correct = 0
    total_claims = supported = fabricated = 0
    per_question = []

    for rec in records:
        try:
            acc = score_answer(str(rec["gt_answer"]), rec["generated_answer"], tolerance)
        except (ValueError, TypeError):
            acc = 0.0
        correct += acc

        claims = rec.get("claims") or []
        n_sup = sum(1 for c in claims if c.get("verdict") == "supported")
        n_fab = sum(1 for c in claims if c.get("verdict") == "fabricated")
        total_claims += len(claims)
        supported += n_sup
        fabricated += n_fab

        per_question.append({
            "uid": rec["uid"], "factual_accuracy": acc,
            "n_claims": len(claims), "n_supported": n_sup, "n_fabricated": n_fab,
        })

    n = len(records)
    metrics = {
        "mode": mode,
        "n_questions": n,
        "factual_accuracy": correct / n,
        "groundedness": (supported / total_claims) if total_claims else 0.0,
        "hallucination_rate": (fabricated / total_claims) if total_claims else 0.0,
        "total_claims": total_claims,
    }
    print(json.dumps(metrics, indent=2))

    out_path = RESULTS_DIR / f"generation_metrics_{mode}.json"
    out_path.write_text(json.dumps({"summary": metrics, "per_question": per_question}, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "engineered"], required=True)
    p.add_argument("--tolerance", type=float, default=0.01)
    args = p.parse_args()
    evaluate(args.mode, args.tolerance)
