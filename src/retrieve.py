"""Run retrieval for every benchmark question against a built Chroma index,
optionally applying the Year metadata filter (engineered mode only)."""
import argparse
import json
import re

import chromadb
import pandas as pd
from tqdm import tqdm

from common import CHROMA_DIR, CSV_DIR, RESULTS_DIR, embed_texts

YEAR_TOKEN_RE = re.compile(r"(?:19|20)\d{2}")
# "2016 to 2024", "2010 and 2015", "2012-2019", "FY2016 through FY2024", etc.
YEAR_RANGE_RE = re.compile(
    rf"({YEAR_TOKEN_RE.pattern})\s*(?:-|–|to|through|and)\s*({YEAR_TOKEN_RE.pattern})",
    re.IGNORECASE,
)


def detect_years_in_question(question):
    """Pull every year the question references, expanding inclusive ranges
    ("FY2016 to FY2024") to every year in between rather than just the two
    endpoints. Not clipped to the corpus's primary window - a question
    comparing CY 2010 vs CY 2015 needs the 2010 document even though 2010
    falls outside the 2015-2025 focus years."""
    found = set()
    for lo, hi in YEAR_RANGE_RE.findall(question):
        lo, hi = int(lo), int(hi)
        if lo <= hi and hi - lo < 50:
            found.update(range(lo, hi + 1))
    found.update(int(m) for m in YEAR_TOKEN_RE.findall(question))
    return sorted(found) if found else None


def run(mode, k=5):
    assert mode in ("baseline", "engineered")
    df = pd.read_csv(CSV_DIR / "officeqa_filtered.csv")

    ch_client = chromadb.PersistentClient(path=str(CHROMA_DIR / mode))
    collection = ch_client.get_collection(mode)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        [embedding] = embed_texts([question])

        where = None
        detected_years = None
        n_results = k
        if mode == "engineered":
            detected_years = detect_years_in_question(question)
            if detected_years:
                where = {"year": {"$in": detected_years}}
                # The metadata filter already narrows the search space to the
                # relevant year(s), so pull a wider context window for the
                # generator while still scoring Hit Rate/MRR/Recall at the
                # required cutoff (evaluate_retrieval slices to [:k]).
                n_results = max(k, 10)

        result = collection.query(
            query_embeddings=[embedding], n_results=n_results, where=where,
            include=["documents", "metadatas", "distances"],
        )

        # If a metadata filter over-narrows and returns nothing, fall back to
        # an unfiltered search rather than returning zero results.
        if where and not result["ids"][0]:
            result = collection.query(
                query_embeddings=[embedding], n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            detected_years = None

        retrieved = [
            {
                "source_file": meta["source_file"],
                "chunk_id": cid,
                "distance": dist,
                "text": doc,
            }
            for cid, meta, dist, doc in zip(
                result["ids"][0], result["metadatas"][0], result["distances"][0],
                result["documents"][0],
            )
        ]

        records.append({
            "uid": row.get("uid"),
            "question": question,
            "answer": row.get("answer"),
            "source_files_gt": row.get("source_files"),
            "detected_years": detected_years,
            "retrieved": retrieved,
        })

    out_path = RESULTS_DIR / f"retrieval_{mode}.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"[{mode}] wrote {len(records)} retrieval records -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "engineered"], required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    run(args.mode, args.k)
