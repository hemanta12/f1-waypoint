"""Retrieval ablation: every method scored on the same index, eval set and cleanup.

uv run python -m core.ablation   # needs a current index: uv run python -m core.index
"""

import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from core.bm25 import BM25Index
from core.evalset import load_eval_set
from core.index import RRF_K, Index, build_index, clean_results, rrf
from core.recall import recall_at_k
from core.report import corpus_hash, run_record, score_cases, write_run
from core.rerank import Reranker

K = 5
FETCH = 10
# The first model is the one in index.sqlite; the others get a cached index under runs/.
EMBEDDING_MODELS = ("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5")
RERANKERS = ("Xenova/ms-marco-MiniLM-L-6-v2", "Xenova/ms-marco-MiniLM-L-12-v2")


def _short(model: str) -> str:
    return model.split("/")[-1].removeprefix("ms-marco-").removesuffix("-en-v1.5")


def _index_for(model: str, repo: Path, module: Path) -> Index:
    """The index built with `model`, rebuilt only when missing or older than the corpus."""
    corpus = module / "corpus"
    path = module / "index.sqlite"
    if model != EMBEDDING_MODELS[0]:
        path = repo / "runs" / "index-cache" / f"{_short(model)}.sqlite"
    if path.exists():
        index = Index(path)
        if (
            index.meta["corpus_hash"] == corpus_hash(corpus)
            and index.meta["embedding_model"] == model
        ):
            return index
    if path.name == "index.sqlite":
        raise SystemExit("index.sqlite is stale or missing; run: uv run python -m core.index")
    from fastembed import TextEmbedding

    print(f"building index for {model} ...", flush=True)
    embedder = TextEmbedding(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_index(corpus, path, lambda t: np.array(list(embedder.passage_embed(t))), model)
    return Index(path)


def main() -> None:
    from fastembed import TextEmbedding

    repo = Path(__file__).resolve().parent.parent
    module = repo / "modules" / "opt_stem"
    corpus = module / "corpus"
    cases = [c for c in load_eval_set(module / "eval") if c.gold]
    rerankers = [Reranker(m) for m in RERANKERS]

    indexes = {m: _index_for(m, repo, module) for m in EMBEDDING_MODELS}
    bm25 = BM25Index(indexes[EMBEDDING_MODELS[0]].chunks)
    b = bm25.bm25
    bm25_settings = {"bm25": f"bm25s {version('bm25s')} {b.method} k1={b.k1} b={b.b}"}

    def keyword(i, c):
        return bm25.search(c.question, FETCH)

    # Each method returns its candidate pool; clean_results turns it into the top K.
    methods = {"bm25": (keyword, bm25_settings)}
    for model, index in indexes.items():
        queries = list(TextEmbedding(model).query_embed([c.question for c in cases]))
        vector_settings = {"embedding_model": model}
        hybrid_settings = {**vector_settings, **bm25_settings, "fusion": f"rrf k={RRF_K}"}

        def vector(i, c, index=index, queries=queries):
            return index.search(queries[i], FETCH)

        def hybrid(i, c, vector=vector):
            return rrf(vector(i, c), keyword(i, c))

        e = _short(model)
        methods[f"{e} vector"] = (vector, vector_settings)
        methods[f"{e} hybrid"] = (hybrid, hybrid_settings)
        for r in rerankers:
            for pool_name, pool_fn, settings in (
                ("vector", vector, vector_settings),
                ("hybrid", hybrid, hybrid_settings),
            ):
                methods[f"{e} {pool_name}+{_short(r.model_name)}"] = (
                    lambda i, c, r=r, f=pool_fn: r.rerank(c.question, f(i, c)),
                    {**settings, "reranker": r.model_name},
                )

    out_root = repo / "runs" / f"ablation-{datetime.now():%Y%m%d-%H%M%S}"
    per_method = {}
    rows = []
    for name, (candidates, settings) in methods.items():
        retrieved, pool_recall = {}, []
        start = time.perf_counter()
        for i, c in enumerate(cases):
            pool = candidates(i, c)
            pool_recall.append(
                recall_at_k(c.gold, [(ch["doc_id"], ch["text"]) for ch, _ in pool], len(pool))
            )
            retrieved[c.id] = [(p["doc_id"], p["text"]) for p in clean_results(pool, K)]
        ms = (time.perf_counter() - start) * 1000 / len(cases)
        results = score_cases(cases, retrieved, K)
        record = run_record(
            repo, corpus, repo / "llm.yaml", {"method": name, "k": K, "fetch": FETCH, **settings}
        )
        write_run(out_root / name, results, record, K)
        per_method[name] = {r["id"]: r["recall_at_k"] for r in results}
        recall = sum(per_method[name].values()) / len(results)
        pool_mean = sum(pool_recall) / len(pool_recall)
        rows.append(f"| {name} | {recall:.3f} | {pool_mean:.3f} | {ms:.0f} | {settings} |")

    table = "| Method | Recall@5 | Correct chunk in pool | ms per question | Settings |\n"
    table += "|---|---|---|---|---|\n"
    table += "\n".join(rows) + "\n"
    names = list(per_method)
    wins = "\n## Questions where the methods disagree\n\n"
    wins += "| Case | " + " | ".join(names) + " |\n|---|" + "---|" * len(names) + "\n"
    wins += "\n".join(
        f"| {cid} | " + " | ".join(f"{per_method[n][cid]:.2f}" for n in names) + " |"
        for cid in per_method[names[0]]
        if len({per_method[n][cid] for n in names}) > 1
    )
    (out_root / "comparison.md").write_text(table + wins + "\n")
    print(f"{table}\nwritten to {out_root}")


if __name__ == "__main__":
    main()
