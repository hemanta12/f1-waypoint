"""Search index: chunks, their source and their vectors in one SQLite file.

uv run python -m core.index      # rebuild modules/opt_stem/index.sqlite
"""

import json
import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.chunk import embed_input, structure_chunks
from core.registry import hash_text
from core.report import corpus_hash
from core.snapshot import extract_sections, find_raw

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_WORDS = 150

Embed = Callable[[list[str]], np.ndarray]

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    section TEXT,
    source_url TEXT NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX chunks_doc ON chunks (doc_id);
"""


def build_index(corpus_dir: Path, db_path: Path, embed: Embed, model_name: str) -> int:
    """Rebuild the whole index from the registry's active documents. Returns the chunk
    count. Written to a temp file and swapped in, so a failed build leaves the old one."""
    entries = json.loads((corpus_dir / "registry.json").read_text())
    rows = []
    for e in entries:
        if e["status"] != "active":
            continue
        raw = find_raw(corpus_dir / "raw", e["doc_id"])
        if raw is None:
            raise FileNotFoundError(f"no raw file for active document {e['doc_id']}")
        for c in structure_chunks(e["doc_id"], extract_sections(raw), MAX_WORDS):
            rows.append((c, e["source_url"]))
    if not rows:
        raise ValueError("no active documents to index")
    vectors = np.asarray(embed([embed_input(c) for c, _ in rows]), dtype=np.float32)

    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as db:
        db.executescript(_SCHEMA)
        db.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (c.chunk_id, c.doc_id, c.section, url, c.text, hash_text(c.text), v.tobytes())
                for (c, url), v in zip(rows, vectors, strict=True)
            ],
        )
        meta = {
            "embedding_model": model_name,
            "dim": str(vectors.shape[1]),
            "max_words": str(MAX_WORDS),
            "corpus_hash": corpus_hash(corpus_dir),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        db.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    db.close()
    tmp.replace(db_path)
    return len(rows)


class Index:
    """All vectors held in memory; search is one matrix product. Fine at a few
    thousand chunks, which is why there is no vector extension."""

    def __init__(self, db_path: Path) -> None:
        with sqlite3.connect(db_path) as db:
            db.row_factory = sqlite3.Row
            self.meta = {r["key"]: r["value"] for r in db.execute("SELECT * FROM meta")}
            rows = db.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
        db.close()
        dim = int(self.meta["dim"])
        self.chunks = [
            {**{k: r[k] for k in r.keys() if k != "vector"}, "pos": i} for i, r in enumerate(rows)
        ]
        self.vectors = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32)
        self.vectors = self.vectors.reshape(len(rows), dim)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[dict, float]]:
        # Vectors are unit length, so the dot product is cosine similarity.
        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def retrieve(self, query_vector: np.ndarray, k: int = 5, fetch: int = 10) -> list[dict]:
        """Top k passages for the generator: search `fetch` chunks, then clean them."""
        return clean_results(self.search(query_vector, fetch), k)


NEAR_COPY = 0.7


def _phrases(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + 5]) for i in range(max(1, len(words) - 4))}


def _overlap(a: set, b: set) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def clean_results(results: list[tuple[dict, float]], k: int = 5) -> list[dict]:
    """Ranked chunks in, ranked passages out.

    1. A chunk that shares NEAR_COPY or more of its 5-word phrases with a better-ranked
       chunk from another document is a copy and is dropped.
    2. Kept chunks from the same document and section join into one passage, in page
       order, ranked where its best piece ranked.
    """
    kept: list[tuple[dict, float, set]] = []
    for chunk, score in results:
        phrases = _phrases(chunk["text"])
        if any(
            c["doc_id"] != chunk["doc_id"] and _overlap(phrases, p) >= NEAR_COPY for c, _, p in kept
        ):
            continue
        kept.append((chunk, score, phrases))

    groups: dict[tuple, list[tuple[dict, float]]] = {}
    for chunk, score, _ in kept:
        groups.setdefault((chunk["doc_id"], chunk["section"]), []).append((chunk, score))
    passages = []
    for members in list(groups.values())[:k]:
        members.sort(key=lambda m: m[0]["pos"])
        first = members[0][0]
        passages.append(
            {
                "doc_id": first["doc_id"],
                "section": first["section"],
                "source_url": first["source_url"],
                "chunk_ids": [m[0]["chunk_id"] for m in members],
                "text": "\n".join(m[0]["text"] for m in members),
                "score": max(m[1] for m in members),
            }
        )
    return passages


if __name__ == "__main__":
    from fastembed import TextEmbedding

    root = Path(__file__).resolve().parent.parent / "modules" / "opt_stem"
    model = TextEmbedding(EMBEDDING_MODEL)
    n = build_index(
        root / "corpus",
        root / "index.sqlite",
        lambda texts: np.array(list(model.passage_embed(texts))),
        EMBEDDING_MODEL,
    )
    print(f"indexed {n} chunks into {root / 'index.sqlite'}")
