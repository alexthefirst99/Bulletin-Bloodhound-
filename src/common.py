import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CSV_DIR = ROOT / "data" / "csv"
RESULTS_DIR = ROOT / "results"
CHROMA_DIR = ROOT / "chroma_store"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
YEARS = set(range(2015, 2026))

FNAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def parse_year_month(filename):
    m = FNAME_RE.search(filename)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_raw_docs():
    """Return {source_file_stem: text} for every downloaded TXT document."""
    docs = {}
    for path in sorted(RAW_DIR.glob("*.txt")):
        docs[path.stem] = path.read_text(errors="ignore")
    return docs


def embed_texts(texts, batch_size=100):
    """Batched OpenAI embedding calls. Returns list[list[float]] aligned to texts."""
    out = []
    c = client()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = c.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


def source_files_list(cell):
    """officeqa's source_files column separates multiple filenames with \\r\\n
    (not commas) - match the filename pattern directly instead of splitting."""
    return FNAME_RE_FULL.findall(str(cell))


FNAME_RE_FULL = re.compile(r"treasury_bulletin_\d{4}_\d{2}")
