# Self Discovery Lab: The Financial RAG Challenge

**Name:** Alex Tran **Recent Years Used:** 2015-2025 (11 years - see "Why 11 years" below)

Data source: [databricks/officeqa](https://huggingface.co/datasets/databricks/officeqa) (Hugging Face, gated) - U.S.
Treasury Bulletins, 1939-2025, transformed-to-Markdown TXT format.

## Why 11 years, not 4

The literal 2022-2025 window only matches **8** of the benchmark's 246 questions (it's an 87-year
archive, so ~2.8 questions/year on average). With 8 questions, any single wrong answer swings a
percentage metric by 12.5 points - too noisy to draw conclusions from. Widening to 2015-2025 yields
**24** questions while still being a "recent years" slice, giving materially more stable metrics.
Documents outside that window are still pulled in when a question needs them for a cross-year
comparison (e.g. a question comparing CY 2010 vs CY 2015 pulls both bulletins even though 2010 is
outside the primary window).

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN (needs approved access to databricks/officeqa) and OPENAI_API_KEY
python src/download_data.py         # pulls officeqa_full.csv, filters to target years, downloads needed TXTs
python src/run_pipeline.py --mode baseline
python src/run_pipeline.py --mode engineered
python src/make_scorecard.py
```

## Architecture

**Vector database:** ChromaDB (`PersistentClient`), one collection per mode (`chroma_store/baseline`,
`chroma_store/engineered`).

**Embeddings:** OpenAI `text-embedding-3-small` for both systems, so the only variables under test
are chunking strategy, metadata, and retrieval/generation logic - not embedding quality.

**Generation:** `gpt-4o-mini`, temperature 0.

### Chunking strategy

- **Baseline:** fixed 2000-character windows, zero overlap, no structural awareness. Will
  frequently cut a markdown table in half mid-row.
- **Engineered:** blocks are split on blank lines first, with any contiguous markdown table
  (`|`-prefixed lines) kept as one atomic block so a chunk boundary can never land inside a table.
  Blocks are then packed into ~512-token chunks with an ~80-token trailing overlap carried into the
  next chunk, so context isn't lost across a boundary. (`src/chunking.py`)

### Metadata (required)

Every chunk in the **engineered** index is tagged with `year` and `month`, parsed from the source
filename (`treasury_bulletin_<YYYY>_<MM>.txt`). At query time (`src/retrieve.py`):

1. The question text is scanned for explicit years (`YEAR_TOKEN_RE`) **and** inclusive year ranges
   ("FY2016 to FY2024", "between CY 2010 and 2015") are expanded to every year in between, not just
   the two endpoints mentioned.
2. If any years are detected, the Chroma query is filtered with `where={"year": {"$in": years}}`
   before similarity ranking runs - i.e. metadata narrows the candidate pool, then vector
   similarity ranks within it.
3. If the filter would return nothing (e.g. a year mismatch), the code falls back to an unfiltered
   search rather than returning zero results.
4. Because the year filter already narrows the search space, the engineered system pulls a wider
   context window (10 chunks) for the generator while Hit Rate/MRR/Recall are still scored at the
   required K=5 cutoff.

The **baseline** index stores `source_file` only (needed for bookkeeping/evaluation) and never
filters by year - every query searches the entire corpus.

## Part 1: The Scorecard (K=5)

| Metric | Baseline (Simple) | Engineered (Improved) |
|---|---|---|
| Hit Rate (K=5) | 45.8% | 75.0% |
| MRR | 0.20 | 0.56 |
| Recall@5 | 25.1% | 54.7% |
| Groundedness | 14.3% (3/21 claims) | 100.0% (2/2 claims) |
| Factual Accuracy | 0.0% | 0.0% |
| Hallucination Rate | 33.3% (7/21 claims) | 0.0% (0/2 claims) |

Factual Accuracy is scored with the **official Databricks `reward.py`** (vendored from the
`databricks/officeqa` GitHub repo, Apache 2.0), tolerance ±1%, so it's numerically comparable to the
benchmark's own grading.

**Read the Groundedness/Hallucination numbers with their denominators, not just the percentage.**
22 of 24 engineered answers were "Not found in provided context" - a deliberate refusal the
engineered system prompt encourages when the retrieved excerpts don't contain the answer. Refusals
aren't factual claims, so the judge (an LLM-as-judge pass, `src/generate.py::judge_answer`) doesn't
score them as supported/fabricated - it excludes them from the denominator entirely. That leaves
only 2 actual asserted claims to judge in the engineered system, vs. 21 in the baseline (which
almost never refuses and instead guesses). 100% groundedness on an N=2 sample is not a claim that
the engineered system is reliably grounded - it's a report that of the only two things it actually
asserted, both numbers were traceable to retrieved text (even though, per Factual Accuracy, both
final answers were still wrong - see the Bottleneck discussion below for why).

Retrieval and generation results, and the underlying per-question data, are in `results/`.

## Part 2: Engineering Reflection

**The Bottleneck.** The baseline's failure is squarely in the Retriever, not the Generator. Hit
Rate@5 was only 45.8% - more than half the time, none of the top-5 chunks even came from a correct
source document, so the Generator was never given a chance to answer correctly regardless of how
good it was. That upstream failure cascades into Hallucination Rate: with nothing relevant in
context, the baseline's ungrounded generator still had to say *something*, so it guessed
plausible-looking numbers (e.g. answering a Pareto-tail-exponent question with "1.000", a
textbook-typical value, instead of the actual computed 1.967) - a Hallucination Rate of 33.3% on
the claims it did assert. The Librarian failed first; the Student's fabrication was a downstream
symptom of that, not an independent failure.

**The Metadata Fix.** Adding Year/Month metadata + range-aware filtering was a clear net positive
for retrieval and helped one dimension of generation, but not the one you'd expect. Retrieval
metrics improved substantially: Hit Rate 45.8%→75.0%, MRR 0.20→0.56, Recall 25.1%→54.7%. On the
generation side, it helped **Hallucination**, not **Factual Accuracy**: paired with a stricter
"answer only from context, otherwise say not found" prompt, Hallucination Rate dropped from 33.3%
to 0% because the system now refuses instead of fabricating. But Factual Accuracy stayed at 0.0% in
both systems. That's the more important finding here: better retrieval got the *right pages* in
front of the model far more often, but most of these questions (even the ones labeled "easy") ask
for a derived statistic - a Pareto tail exponent via the Hill estimator, a Tukey-hinge quartile, a
20%-trimmed mean of log values, a multi-year CAGR - computed from data points scattered across many
separate monthly/yearly Bulletin editions. Finding the right five *pages* doesn't help if the answer
requires aggregating and correctly computing over dozens of data points spread across the whole
corpus in a single retrieve-then-generate pass. Metadata filtering measurably fixed the Librarian's
job; it could not fix a Generator task that plain top-K RAG isn't architected to do at all.

**Scaling Insight.** Scaling from this 11-year, 24-question, 28-document subset to the full 80+ year
archive (1939-2025, 697 documents), the first thing to break isn't infrastructure - Chroma and
OpenAI embeddings scale roughly linearly to ~135k chunks without issue, and even the gated HF
download just takes longer. What breaks first is the **retrieval architecture itself**: this
project's year-filtering already leans on explicit year mentions/ranges in the question text, and
that heuristic gets much weaker over 80 years, where cross-decade comparisons ("compare 1950 to
2020"), rolling windows ("the last 20 years"), and implicit date references (referring to an era by
event rather than year) become far more common than in an 11-year slice. More fundamentally, the
questions this benchmark asks - multi-document statistical aggregation, not single-fact lookup - are
already the wrong shape for single-pass "embed query, fetch top-K, generate" RAG, and that mismatch
only gets worse as the number of documents a single question might need to fuse across grows from a
handful to dozens. Fixing it requires a different architecture entirely: agentic multi-hop retrieval
that can issue several targeted queries per question, plus a code-execution/calculator tool so the
model computes statistics from extracted numbers instead of eyeballing them in free-text generation
- not just a bigger vector index.

## Repo layout

```
src/
  download_data.py        # HF download + year filtering
  inspect_repo.py         # one-off: discover the gated repo's real file layout
  common.py                # shared config, embedding, year/month parsing
  chunking.py              # naive_chunk (baseline) / smart_chunk (engineered)
  build_index.py           # builds a Chroma collection for a given mode
  retrieve.py               # per-question retrieval, engineered = metadata-filtered
  generate.py               # answer generation + LLM-judge claim scoring
  evaluate_retrieval.py    # Hit Rate@5 / MRR@5 / Recall@5
  evaluate_generation.py   # Factual Accuracy (official reward.py) / Groundedness / Hallucination
  reward.py                 # vendored from databricks/officeqa (Apache 2.0)
  run_pipeline.py           # orchestrates one mode end-to-end
  make_scorecard.py         # renders results/scorecard.md
data/                       # gitignored - CSV + TXT corpus (gated, not redistributed)
results/                    # retrieval/generation records + metrics + scorecard.md
```

## Licensing note

`src/reward.py` and `THIRD_PARTY_LICENSE-APACHE` are vendored unmodified from
[databricks/officeqa](https://github.com/databricks/officeqa) (Apache 2.0). The dataset itself
(CC-BY-SA 4.0) is not redistributed here - `data/` is gitignored and must be re-downloaded per the
Setup steps above with your own approved HF access.
