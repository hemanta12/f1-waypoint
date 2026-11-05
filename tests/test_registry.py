from datetime import date

import pytest
from pydantic import ValidationError

from core.registry import Document, apply_refetch, hash_text

FIRST = date(2026, 11, 5)
LATER = date(2026, 11, 12)


def _doc(text: str = "Report within 10 days.") -> Document:
    return Document(
        doc_id="uscis-stem-opt",
        module_id="opt_stem",
        title="STEM OPT",
        source_url="https://www.uscis.gov/example",
        source_tier="official",
        retrieved_at=FIRST,
        verified_on=FIRST,
        content_hash=hash_text(text),
    )


def test_hash_ignores_whitespace_but_not_words():
    assert hash_text("a  b\n c") == hash_text("a b c")
    assert hash_text("10 days") != hash_text("11 days")


def test_refetch_unchanged_moves_only_verified_on():
    doc = _doc()
    new = apply_refetch(doc, "Report  within 10 days.\n", LATER)
    assert new.verified_on == LATER
    assert new.retrieved_at == FIRST
    assert new.content_hash == doc.content_hash


def test_refetch_changed_moves_hash_and_retrieved_at():
    doc = _doc()
    new = apply_refetch(doc, "Report within 15 days.", LATER)
    assert new.content_hash == hash_text("Report within 15 days.")
    assert new.retrieved_at == LATER
    assert new.verified_on == LATER


def test_refetch_never_changes_identity():
    doc = _doc()
    new = apply_refetch(doc, "totally different", LATER)
    assert (new.doc_id, new.source_url, new.source_tier) == (
        doc.doc_id,
        doc.source_url,
        doc.source_tier,
    )


def test_refetch_revives_removed_doc():
    doc = _doc().model_copy(update={"status": "removed"})
    assert apply_refetch(doc, "Report within 10 days.", LATER).status == "active"


@pytest.mark.parametrize("bad", ["Has Caps", "under_score", "", "-lead"])
def test_doc_id_must_be_a_slug(bad):
    with pytest.raises(ValidationError):
        Document.model_validate({**_doc().model_dump(), "doc_id": bad})


def test_unknown_tier_rejected():
    with pytest.raises(ValidationError):
        Document.model_validate({**_doc().model_dump(), "source_tier": "blog"})


def test_real_source_list_loads_with_unique_ids():
    from pathlib import Path

    from core.registry import load_sources

    root = Path(__file__).resolve().parent.parent
    sources = load_sources(root / "modules" / "opt_stem" / "sources.yaml")
    assert len(sources) == len({s.doc_id for s in sources}) > 0
