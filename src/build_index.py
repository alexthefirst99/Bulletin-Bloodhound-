"""Build a persistent Chroma collection for either the baseline or engineered
chunking/metadata strategy."""
import argparse
import shutil

import chromadb
from tqdm import tqdm

from chunking import naive_chunk, smart_chunk
from common import CHROMA_DIR, embed_texts, load_raw_docs, parse_year_month


def build(mode):
    assert mode in ("baseline", "engineered")
    docs = load_raw_docs()
    print(f"[{mode}] loaded {len(docs)} source documents")

    chunk_fn = naive_chunk if mode == "baseline" else smart_chunk

    ids, texts, metadatas = [], [], []
    for stem, text in docs.items():
        year, month = parse_year_month(stem)
        pieces = chunk_fn(text)
        for idx, piece in enumerate(pieces):
            ids.append(f"{stem}__{idx}")
            texts.append(piece)
            meta = {"source_file": stem}
            if mode == "engineered":
                # Required metadata tags used later to pre-filter search.
                meta["year"] = year or 0
                meta["month"] = month or 0
            metadatas.append(meta)

    print(f"[{mode}] {len(texts)} chunks; embedding ...")

    persist_path = CHROMA_DIR / mode
    if persist_path.exists():
        shutil.rmtree(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    ch_client = chromadb.PersistentClient(path=str(persist_path))
    collection = ch_client.create_collection(mode)

    batch = 100
    for i in tqdm(range(0, len(texts), batch)):
        b_texts = texts[i:i + batch]
        b_ids = ids[i:i + batch]
        b_meta = metadatas[i:i + batch]
        b_embeds = embed_texts(b_texts)
        collection.add(ids=b_ids, documents=b_texts, metadatas=b_meta, embeddings=b_embeds)

    print(f"[{mode}] index built at {persist_path} ({len(texts)} chunks)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "engineered"], required=True)
    args = p.parse_args()
    build(args.mode)
