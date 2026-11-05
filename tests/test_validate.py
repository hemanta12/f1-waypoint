import json
from datetime import date
from pathlib import Path

from core.registry import Source, load_sources
from core.snapshot import build_registry
from core.validate import validate_corpus

ROOT = Path(__file__).resolve().parent.parent


def _source(doc_id: str) -> Source:
    return Source(
        doc_id=doc_id,
        title="T",
        url="https://example.gov/x",
        tier="official",
        authoritative_for="x",
    )


def _built(tmp_path, *doc_ids):
    (tmp_path / "raw").mkdir()
    for d in doc_ids:
        (tmp_path / "raw" / f"{d}.html").write_text(f"<p>{d} text</p>")
    build_registry([_source(d) for d in doc_ids], "m", tmp_path, date(2026, 11, 5))
    return tmp_path


def test_complete_corpus_has_no_problems(tmp_path):
    corpus = _built(tmp_path, "a", "b")
    assert validate_corpus([_source("a"), _source("b")], corpus) == []


def test_reports_missing_registry():
    assert validate_corpus([], Path("/nonexistent")) == ["registry.json is missing"]


def test_reports_unregistered_source_and_missing_raw(tmp_path):
    corpus = _built(tmp_path, "a")
    (corpus / "raw" / "a.html").unlink()
    problems = validate_corpus([_source("a"), _source("b")], corpus)
    assert "a: raw file missing" in problems
    assert "b: not in registry" in problems


def test_reports_hand_edited_text(tmp_path):
    corpus = _built(tmp_path, "a")
    (corpus / "text" / "a.txt").write_text("edited by hand")
    assert "a: content_hash does not match extracted text" in validate_corpus(
        [_source("a")], corpus
    )


def test_reports_entry_without_hash(tmp_path):
    corpus = _built(tmp_path, "a")
    entries = json.loads((corpus / "registry.json").read_text())
    entries[0]["content_hash"] = ""
    (corpus / "registry.json").write_text(json.dumps(entries))
    problems = validate_corpus([_source("a")], corpus)
    assert problems[0].startswith("a: invalid entry")


def test_real_corpus_is_complete():
    module = ROOT / "modules" / "opt_stem"
    assert validate_corpus(load_sources(module / "sources.yaml"), module / "corpus") == []
