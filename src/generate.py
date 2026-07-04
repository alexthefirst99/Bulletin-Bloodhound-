"""Generate answers from retrieved context (Set B input), plus an LLM-judge
pass that decomposes each answer into atomic claims for Groundedness /
Hallucination scoring."""
import argparse
import json

from tqdm import tqdm

from common import CHAT_MODEL, RESULTS_DIR, client

BASELINE_SYSTEM = (
    "Answer the user's question using the provided excerpts. "
    "Respond with only the direct final answer (a number, date, or short "
    "phrase) and nothing else."
)

ENGINEERED_SYSTEM = (
    "You are a careful financial research assistant. Answer the question "
    "using ONLY the provided excerpts, each labeled with its source year and "
    "month. If the excerpts do not contain the answer, respond exactly with "
    "'Not found in provided context'. "
    "Respond with only the direct final answer (a number, date, or short "
    "phrase) and nothing else - no explanation, no citation in the answer "
    "itself."
)

JUDGE_SYSTEM = (
    "You are an evaluation judge. Given a CONTEXT and an ANSWER, break the "
    "ANSWER into its atomic factual claims (usually just one for a short "
    "answer). "
    "If the ANSWER is a refusal or non-answer (e.g. 'Not found in provided "
    "context', 'I don't know', 'cannot determine') it makes ZERO factual "
    "claims - return {\"claims\": []} for it. Declining to answer is not "
    "itself a claim to be judged. "
    "Otherwise, for each claim, decide: "
    "'supported' (the claim is directly backed by the CONTEXT), "
    "'fabricated' (the claim states a specific fact/number not present in "
    "the CONTEXT at all), or "
    "'unsupported' (related to the CONTEXT but not precisely verifiable from "
    "it). "
    "Return strict JSON: {\"claims\": [{\"text\": str, \"verdict\": "
    "\"supported\"|\"fabricated\"|\"unsupported\"}]}"
)


def build_context(retrieved, engineered):
    parts = []
    for r in retrieved:
        if engineered:
            parts.append(f"[Source: {r['source_file']}]\n{r['text']}")
        else:
            parts.append(r["text"])
    return "\n\n---\n\n".join(parts)


def generate_answer(c, question, context, engineered):
    system = ENGINEERED_SYSTEM if engineered else BASELINE_SYSTEM
    resp = c.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def judge_answer(c, context, answer):
    resp = c.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER: {answer}"},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content)["claims"]
    except (json.JSONDecodeError, KeyError):
        return []


def run(mode):
    engineered = mode == "engineered"
    records = json.loads((RESULTS_DIR / f"retrieval_{mode}.json").read_text())
    c = client()

    out = []
    for rec in tqdm(records):
        context = build_context(rec["retrieved"], engineered)
        answer = generate_answer(c, rec["question"], context, engineered)
        claims = judge_answer(c, context, answer)
        out.append({
            "uid": rec["uid"],
            "question": rec["question"],
            "gt_answer": rec["answer"],
            "generated_answer": answer,
            "claims": claims,
        })

    out_path = RESULTS_DIR / f"generation_{mode}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{mode}] wrote {len(out)} generation records -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "engineered"], required=True)
    args = p.parse_args()
    run(args.mode)
