"""Chunking experiment: same embedding model and retriever, only chunking varies.

uv run python -m core.experiment
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from core.chunk import Chunk, embed_input, fixed_size_chunks, structure_chunks
from core.evalset import load_eval_set
from core.index import EMBEDDING_MODEL
from core.recall import is_hit
from core.report import run_record, score_cases, write_run
from core.snapshot import extract_sections, find_raw

K = 5
SIZES = (150, 200, 300)


def build_chunks(corpus_dir: Path, strategy: str, size: int) -> list[Chunk]:
    entries = json.loads((corpus_dir / "registry.json").read_text())
    chunks: list[Chunk] = []
    for e in entries:
        if e["status"] != "active":
            continue
        doc_id = e["doc_id"]
        if strategy == "fixed":
            text = (corpus_dir / "text" / f"{doc_id}.txt").read_text(encoding="utf-8")
            chunks += fixed_size_chunks(doc_id, text, size=size, overlap=size // 5)
        else:
            chunks += structure_chunks(
                doc_id, extract_sections(find_raw(corpus_dir / "raw", doc_id)), size
            )
    return chunks


def main() -> None:
    from fastembed import TextEmbedding

    repo = Path(__file__).resolve().parent.parent
    module = repo / "modules" / "opt_stem"
    corpus = module / "corpus"
    cases = [c for c in load_eval_set(module / "eval") if c.gold]
    model = TextEmbedding(EMBEDDING_MODEL)
    queries = np.array(list(model.query_embed([c.question for c in cases])))

    out_root = repo / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    for strategy in ("fixed", "structure"):
        for size in SIZES:
            name = f"{strategy}-{size}"
            chunks = build_chunks(corpus, strategy, size)
            vectors = np.array(list(model.passage_embed([embed_input(c) for c in chunks])))
            # fastembed returns unit vectors, so the dot product is cosine similarity.
            top = np.argsort(-(queries @ vectors.T), axis=1)[:, :K]
            retrieved = {
                c.id: [(chunks[j].doc_id, chunks[j].text) for j in top[i]]
                for i, c in enumerate(cases)
            }
            results = score_cases(cases, retrieved, K)
            gold = [g for c in cases for g in c.gold]
            whole = sum(any(is_hit(g, (ch.doc_id, ch.text)) for ch in chunks) for g in gold)
            retriever = {
                "chunking": strategy,
                "max_words": size,
                "overlap": size // 5 if strategy == "fixed" else 0,
                "embedding_model": EMBEDDING_MODEL,
                "retriever": "dense cosine",
                "k": K,
            }
            write_run(
                out_root / name, results, run_record(repo, corpus, repo / "llm.yaml", retriever), K
            )
            recall = sum(r["recall_at_k"] for r in results) / len(results)
            rows.append(f"| {name} | {len(chunks)} | {whole}/{len(gold)} | {recall:.3f} |")
            print(rows[-1], flush=True)

    table = "| Strategy | Chunks | Gold quotes whole in a chunk | Recall@5 |\n|---|---|---|---|\n"
    table += "\n".join(rows) + "\n"
    (out_root / "comparison.md").write_text(table)
    print(f"\n{table}\nwritten to {out_root}")


if __name__ == "__main__":
    main()
