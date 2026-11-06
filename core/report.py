import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from core.config import load_llm
from core.evalset import EvalCase
from core.recall import Retrieved, is_hit, recall_at_k


def score_cases(
    cases: list[EvalCase], retrieved: dict[str, list[Retrieved]], k: int = 5
) -> list[dict]:
    """Per-question results. Cases without gold get recall None; a case with no
    retrieval output scores as if nothing came back."""
    results = []
    for c in cases:
        chunks = retrieved.get(c.id, [])
        top = chunks[:k]
        results.append(
            {
                "id": c.id,
                "category": c.category,
                "question": c.question,
                "recall_at_k": recall_at_k(c.gold, chunks, k) if c.gold else None,
                "gold_found": [any(is_hit(g, ch) for ch in top) for g in c.gold],
                "gold_doc_ids": [g.doc_id for g in c.gold],
                "retrieved_doc_ids": [doc_id for doc_id, _ in top],
            }
        )
    return results


def summary_table(results: list[dict], k: int = 5) -> str:
    """Markdown table of mean Recall@k per category, plus overall."""
    rows: dict[str, list[float]] = {}
    for r in results:
        if r["recall_at_k"] is not None:
            rows.setdefault(r["category"], []).append(r["recall_at_k"])
    lines = [f"| Category | Cases scored | Recall@{k} |", "|---|---|---|"]
    for cat in sorted(rows):
        scores = rows[cat]
        lines.append(f"| {cat} | {len(scores)} | {sum(scores) / len(scores):.2f} |")
    every = [s for scores in rows.values() for s in scores]
    overall = f"{sum(every) / len(every):.2f}" if every else "n/a"
    lines.append(f"| **All** | {len(every)} | {overall} |")
    return "\n".join(lines) + "\n"


def corpus_hash(corpus_dir: Path) -> str:
    """One hash over every document's content hash, so any text change shows up."""
    entries = json.loads((corpus_dir / "registry.json").read_text())
    joined = "\n".join(
        f"{e['doc_id']}:{e['content_hash']}" for e in sorted(entries, key=lambda e: e["doc_id"])
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def run_record(repo: Path, corpus_dir: Path, llm_path: Path, retriever: dict) -> dict:
    """What produced a run: code version, corpus version, models and settings."""
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git(repo, "rev-parse", "HEAD"),
        # Uncommitted changes mean git_sha alone cannot reproduce the run.
        "git_dirty": bool(_git(repo, "status", "--porcelain")),
        "corpus_hash": corpus_hash(corpus_dir),
        "retriever": retriever,
        "llm": load_llm(llm_path).model_dump(),
    }


def write_run(out_dir: Path, results: list[dict], record: dict, k: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"run": record, "results": results}, indent=2, ensure_ascii=False) + "\n"
    )
    (out_dir / "summary.md").write_text(summary_table(results, k))
