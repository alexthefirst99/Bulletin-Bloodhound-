import json

from common import RESULTS_DIR


def load(mode):
    ret = json.loads((RESULTS_DIR / f"retrieval_metrics_{mode}.json").read_text())
    gen = json.loads((RESULTS_DIR / f"generation_metrics_{mode}.json").read_text())["summary"]
    return ret, gen


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    b_ret, b_gen = load("baseline")
    e_ret, e_gen = load("engineered")

    lines = [
        "| Metric | Baseline (Simple) | Engineered (Improved) |",
        "|---|---|---|",
        f"| Hit Rate (K={b_ret['k']}) | {pct(b_ret['hit_rate_at_k'])} | {pct(e_ret['hit_rate_at_k'])} |",
        f"| MRR | {b_ret['mrr_at_k']:.2f} | {e_ret['mrr_at_k']:.2f} |",
        f"| Recall@{b_ret['k']} | {pct(b_ret['recall_at_k'])} | {pct(e_ret['recall_at_k'])} |",
        f"| Groundedness | {pct(b_gen['groundedness'])} | {pct(e_gen['groundedness'])} |",
        f"| Factual Accuracy | {pct(b_gen['factual_accuracy'])} | {pct(e_gen['factual_accuracy'])} |",
        f"| Hallucination Rate | {pct(b_gen['hallucination_rate'])} | {pct(e_gen['hallucination_rate'])} |",
    ]
    table = "\n".join(lines)
    print(table)
    (RESULTS_DIR / "scorecard.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
