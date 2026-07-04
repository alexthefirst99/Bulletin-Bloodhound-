"""
Set A metrics: Hit Rate@5, MRR@5, Recall@5.

The benchmark gives document-level ground truth (`source_files` per question),
not chunk-level relevance judgments. We treat a retrieved chunk as "relevant"
if its source_file is one of the question's ground-truth source_files, and
define Recall@5 at the document level: of the N distinct ground-truth source
files a question needs, how many are represented anywhere in the top-5
retrieved chunks.
"""
import argparse
import json

from common import RESULTS_DIR, source_files_list


def evaluate(mode, k=5):
    records = json.loads((RESULTS_DIR / f"retrieval_{mode}.json").read_text())

    hits, rr_sum, recall_sum = 0, 0.0, 0.0
    n = len(records)

    for rec in records:
        gt_files = set(source_files_list(rec["source_files_gt"]))
        retrieved_files = [r["source_file"] for r in rec["retrieved"][:k]]

        first_rank = None
        matched = set()
        for rank, sf in enumerate(retrieved_files, start=1):
            if sf in gt_files:
                if first_rank is None:
                    first_rank = rank
                matched.add(sf)

        if first_rank is not None:
            hits += 1
            rr_sum += 1.0 / first_rank

        recall_sum += (len(matched) / len(gt_files)) if gt_files else 0.0

    metrics = {
        "mode": mode,
        "k": k,
        "n_questions": n,
        "hit_rate_at_k": hits / n,
        "mrr_at_k": rr_sum / n,
        "recall_at_k": recall_sum / n,
    }
    print(json.dumps(metrics, indent=2))

    out_path = RESULTS_DIR / f"retrieval_metrics_{mode}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "engineered"], required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    evaluate(args.mode, args.k)
