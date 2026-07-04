"""
Two chunking strategies used to isolate the effect of "smart" chunking +
metadata in the Engineered system vs. the naive Baseline.
"""
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text):
    return len(_ENC.encode(text))


def naive_chunk(text, chunk_chars=2000):
    """Baseline: fixed character windows, no overlap, no table awareness.
    Will happily cut a markdown table in half."""
    chunks = []
    for i in range(0, len(text), chunk_chars):
        piece = text[i:i + chunk_chars].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _split_blocks(text):
    """Split text into blocks along blank lines, keeping each markdown table
    (contiguous lines starting with '|') as a single atomic block so a chunk
    boundary never lands inside a table."""
    lines = text.split("\n")
    blocks, cur, in_table = [], [], False
    for line in lines:
        is_table_line = line.strip().startswith("|")
        if is_table_line and not in_table and cur and not all(not l.strip() for l in cur[-1:]):
            blocks.append("\n".join(cur))
            cur = []
        in_table = is_table_line
        cur.append(line)
        if not line.strip() and not in_table:
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def smart_chunk(text, target_tokens=512, overlap_tokens=80):
    """Engineered: token-budgeted chunks built from table/paragraph-aware
    blocks, with a token overlap between consecutive chunks so context isn't
    lost at a boundary. Oversized single blocks (e.g. a huge table) are kept
    whole rather than split mid-table."""
    blocks = _split_blocks(text)
    chunks = []
    cur_blocks, cur_tokens = [], 0

    def flush():
        if cur_blocks:
            chunks.append("\n".join(cur_blocks).strip())

    for block in blocks:
        bt = n_tokens(block)
        if cur_tokens + bt > target_tokens and cur_blocks:
            flush()
            # carry overlap: keep trailing blocks worth ~overlap_tokens
            overlap, ot = [], 0
            for b in reversed(cur_blocks):
                bt2 = n_tokens(b)
                if ot + bt2 > overlap_tokens:
                    break
                overlap.insert(0, b)
                ot += bt2
            cur_blocks, cur_tokens = overlap, ot
        cur_blocks.append(block)
        cur_tokens += bt
    flush()
    return [c for c in chunks if c.strip()]
