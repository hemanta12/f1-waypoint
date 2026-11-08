import json

import numpy as np
import pytest

from core.index import Index, build_index, clean_results


def _corpus(tmp_path, docs):
    (tmp_path / "raw").mkdir()
    entries = []
    for doc_id, html, status in docs:
        (tmp_path / "raw" / f"{doc_id}.html").write_text(html)
        entries.append(
            {
                "doc_id": doc_id,
                "source_url": f"https://example.gov/{doc_id}",
                "content_hash": "0" * 64,
                "status": status,
            }
        )
    (tmp_path / "registry.json").write_text(json.dumps(entries))
    return tmp_path


def _fake_embed(texts):
    # One axis per keyword, so the test controls which chunk is nearest.
    keys = ["unemployment", "report", "fee"]
    v = np.array([[float(k in t.lower()) + 0.01 for k in keys] for t in texts])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_build_and_search_returns_metadata(tmp_path):
    words = " ".join(["filler"] * 40)
    corpus = _corpus(
        tmp_path,
        [
            (
                "a",
                f"<main><h2>Unemployment</h2><p>Unemployment limit. {words}</p></main>",
                "active",
            ),
            ("b", f"<main><h2>Reporting</h2><p>Report in 10 days. {words}</p></main>", "active"),
            ("gone", f"<main><h2>Fee</h2><p>Fee text. {words}</p></main>", "removed"),
        ],
    )
    db = tmp_path / "index.sqlite"
    assert build_index(corpus, db, _fake_embed, "fake") == 2

    index = Index(db)
    assert index.meta["embedding_model"] == "fake"
    (best, score), _ = index.search(_fake_embed(["report"])[0], k=2)
    assert best["chunk_id"] == "b#reporting"
    assert best["section"] == "Reporting"
    assert best["source_url"] == "https://example.gov/b"
    assert len(best["content_hash"]) == 64
    assert score > 0.9
    assert all(c["doc_id"] != "gone" for c in index.chunks)


def test_failed_build_keeps_the_old_index(tmp_path):
    corpus = _corpus(tmp_path, [("a", "<main><p>" + "word " * 40 + "</p></main>", "active")])
    db = tmp_path / "index.sqlite"
    build_index(corpus, db, _fake_embed, "first")

    def broken(texts):
        raise RuntimeError("model download failed")

    with pytest.raises(RuntimeError):
        build_index(corpus, db, broken, "second")
    assert Index(db).meta["embedding_model"] == "first"


def test_active_document_without_raw_file_fails_loudly(tmp_path):
    corpus = _corpus(tmp_path, [("a", "<p>x</p>", "active")])
    (corpus / "raw" / "a.html").unlink()
    with pytest.raises(FileNotFoundError):
        build_index(corpus, tmp_path / "index.sqlite", _fake_embed, "fake")


def _chunk(cid, doc, section, text, pos):
    return {
        "chunk_id": cid,
        "doc_id": doc,
        "section": section,
        "source_url": "u",
        "text": text,
        "pos": pos,
    }


RULE = "If an employer terminates a student, you are required to notify the DSO within five days."


def test_near_copy_from_another_page_is_dropped_and_the_gap_refilled():

    results = [
        (_chunk("a#x", "a", "X", RULE, 0), 0.9),
        (_chunk("b#x", "b", "X", RULE + " Extra words.", 5), 0.8),
        (_chunk("c#y", "c", "Y", "Report changes within 10 days of the change.", 9), 0.7),
    ]
    passages = clean_results(results, k=2)
    assert [p["chunk_ids"] for p in passages] == [["a#x"], ["c#y"]]


def test_same_text_in_the_same_document_is_not_treated_as_a_copy():

    results = [(_chunk("a#x", "a", "X", RULE, 0), 0.9), (_chunk("a#y", "a", "Y", RULE, 3), 0.8)]
    assert len(clean_results(results)) == 2


def test_pieces_of_one_section_join_in_page_order_at_the_best_rank():

    results = [
        (_chunk("a#x/2", "a", "X", "second part", 2), 0.9),
        (_chunk("c#y", "c", "Y", "other rule", 7), 0.8),
        (_chunk("a#x/1", "a", "X", "first part", 1), 0.7),
    ]
    first, second = clean_results(results)
    assert first["chunk_ids"] == ["a#x/1", "a#x/2"]
    assert first["text"] == "first part\nsecond part"
    assert first["score"] == 0.9
    assert second["chunk_ids"] == ["c#y"]


def test_no_results_gives_no_passages():

    assert clean_results([]) == []
