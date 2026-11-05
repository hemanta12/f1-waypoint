import json
import sys
from pathlib import Path

from pydantic import ValidationError

from core.registry import Document, Source, hash_text, load_sources
from core.snapshot import find_raw


def validate_corpus(sources: list[Source], corpus_dir: Path) -> list[str]:
    """Return every problem found; an empty list means the corpus is complete."""
    registry_path = corpus_dir / "registry.json"
    if not registry_path.exists():
        return ["registry.json is missing"]
    problems: list[str] = []
    entries: dict[str, Document] = {}
    for raw_entry in json.loads(registry_path.read_text()):
        try:
            doc = Document.model_validate(raw_entry)
        except ValidationError as e:
            problems.append(
                f"{raw_entry.get('doc_id', '?')}: invalid entry ({e.error_count()} errors)"
            )
            continue
        entries[doc.doc_id] = doc

    listed = {s.doc_id for s in sources}
    for doc_id in sorted(entries.keys() - listed):
        problems.append(f"{doc_id}: in registry but not in sources.yaml")

    for s in sources:
        doc = entries.get(s.doc_id)
        if doc is None:
            problems.append(f"{s.doc_id}: not in registry")
            continue
        if find_raw(corpus_dir / "raw", s.doc_id) is None:
            problems.append(f"{s.doc_id}: raw file missing")
        text_path = corpus_dir / "text" / f"{s.doc_id}.txt"
        text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        if not text.strip():
            problems.append(f"{s.doc_id}: extracted text missing or empty")
        elif hash_text(text) != doc.content_hash:
            problems.append(f"{s.doc_id}: content_hash does not match extracted text")
    return problems


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "modules" / "opt_stem"
    found = validate_corpus(load_sources(root / "sources.yaml"), root / "corpus")
    for p in found:
        print(p)
    print("corpus OK" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
