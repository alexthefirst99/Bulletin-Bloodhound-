"""
Download the OfficeQA benchmark (Databricks) from Hugging Face and filter it
down to the years this project targets.

Requires HF_TOKEN in .env with *approved* access to databricks/officeqa
(listing files works before approval; downloading content does not).

Repo layout (confirmed via src/inspect_repo.py):
  officeqa_full.csv
  treasury_bulletin_pdfs/treasury_bulletin_<YYYY>_<MM>.pdf
  treasury_bulletins_parsed/jsons/treasury_bulletin_<YYYY>_<MM>.json
  treasury_bulletins_parsed/transformed/treasury_bulletin_<YYYY>_<MM>.txt
"""
import os
import re
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

REPO_ID = "databricks/officeqa"
YEARS = set(range(2015, 2026))

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data" / "csv"
RAW_DIR = ROOT / "data" / "raw"

FNAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")


def parse_year_month(filename):
    m = FNAME_RE.search(filename)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN not set. Fill it in in .env first.")

    print(f"Downloading {REPO_ID} CSV metadata ...")
    csv_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["officeqa_full.csv"],
        token=hf_token,
    )
    src_csv = Path(csv_dir) / "officeqa_full.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_csv, CSV_DIR / "officeqa_full.csv")

    df = pd.read_csv(CSV_DIR / "officeqa_full.csv")
    print(f"Full benchmark: {len(df)} questions. Columns: {list(df.columns)}")

    # source_files separates multiple filenames with \r\n, not commas -
    # match the filename pattern directly rather than splitting on a delimiter.
    fname_re = re.compile(r"treasury_bulletin_\d{4}_\d{2}")

    def extract_stems(source_files):
        return fname_re.findall(str(source_files))

    df["_stems"] = df["source_files"].apply(extract_stems)
    df["_source_years"] = df["_stems"].apply(
        lambda stems: {parse_year_month(s)[0] for s in stems}
    )
    mask = df["_source_years"].apply(lambda ys: bool(ys & YEARS))
    filtered = df[mask].drop(columns=["_source_years", "_stems"]).reset_index(drop=True)
    filtered.to_csv(CSV_DIR / "officeqa_filtered.csv", index=False)
    print(f"Filtered to years {sorted(YEARS)}: {len(filtered)} questions "
          f"-> data/csv/officeqa_filtered.csv")

    needed_files = set()
    for source_files in filtered["source_files"]:
        needed_files.update(extract_stems(source_files))
    print(f"Need {len(needed_files)} distinct source documents.")

    print("Downloading transformed TXT corpus (filtered to needed files) ...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    patterns = [f"treasury_bulletins_parsed/transformed/{stem}.txt" for stem in needed_files]

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        token=hf_token,
    )
    src_txt_dir = Path(local_dir) / "treasury_bulletins_parsed" / "transformed"
    count = 0
    if src_txt_dir.exists():
        for f in src_txt_dir.glob("*.txt"):
            shutil.copy(f, RAW_DIR / f.name)
            count += 1
    print(f"Copied {count} TXT files -> data/raw/")

    if count < len(needed_files):
        print(f"WARNING: expected {len(needed_files)} files, got {count}. "
              f"Some source_files entries may not match the "
              f"treasury_bulletin_<YYYY>_<MM> naming convention.")


if __name__ == "__main__":
    main()
