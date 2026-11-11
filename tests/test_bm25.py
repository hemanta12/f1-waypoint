import pytest

from core.bm25 import BM25Index, tokenize


@pytest.mark.parametrize("text", ["Form I-983", "form i-983", "the i983", "I983", "I 983"])
def test_form_number_spellings_share_one_token(text):
    assert "i983" in tokenize(text)


@pytest.mark.parametrize("dash", ["\u2010", "\u2011", "\u2013", "\u2014"])
def test_form_number_with_unicode_dash(dash):
    assert tokenize(f"Form I{dash}983") == ["form", "i983"]


def test_form_number_suffix_letter_kept():
    assert "i129s" in tokenize("File Form I-129S")


def test_form_number_does_not_leave_generic_number():
    tokens = tokenize("Your I-20 must be endorsed")
    assert "i20" in tokens
    assert "20" not in tokens
    assert "i" not in tokens


def test_cfr_citation_emits_itself_and_parents():
    tokens = tokenize("See 8 CFR 214.2(f)(10)(ii) for details")
    for t in ("214.2", "214.2(f)", "214.2(f)(10)", "214.2(f)(10)(ii)"):
        assert t in tokens, t


def test_broad_citation_query_matches_specific_chunk():
    chunk = set(tokenize("See 8 CFR 214.2(f)(10)(ii) for details"))
    query = set(tokenize("what does 214.2(f)(10) say"))
    assert "214.2(f)(10)" in query & chunk


def test_citation_with_letter_in_part_number():
    tokens = tokenize("EAD category 8 CFR 274a.12(c)(3)(i)")
    for t in ("274a.12", "274a.12(c)", "274a.12(c)(3)", "274a.12(c)(3)(i)"):
        assert t in tokens, t


def test_ead_category_code_is_one_token():
    assert tokenize("Form I-765 (c)(3)(C)") == ["form", "i765", "c3c"]
    assert tokenize("I-765(c)(3)(B)") == ["i765", "c3b"]


def test_ead_code_inside_citation_stays_citation():
    assert "c3i" not in tokenize("8 CFR 274a.12(c)(3)(i)")


def test_section_sign_citation():
    assert "214.2(f)" in tokenize("under § 214.2(f)(5)")


def test_plain_text_lowercased_and_common_words_dropped():
    assert tokenize("How do I get the STEM OPT extension") == ["get", "stem", "opt", "extension"]


def test_empty_text():
    assert tokenize("") == []


def _chunks():
    rows = [
        ("a", "Training plan", "Submit Form I-983 to your DSO."),
        ("b", "Unemployment", "You may not exceed 90 days of unemployment."),
        ("c", "Fees", "The filing fee for Form I-765 is listed below."),
    ]
    return [
        {
            "chunk_id": f"{d}-0",
            "doc_id": d,
            "section": s,
            "source_url": f"https://x.gov/{d}",
            "text": t,
            "pos": i,
        }
        for i, (d, s, t) in enumerate(rows)
    ]


def test_bm25_finds_form_number_written_differently():
    hits = BM25Index(_chunks()).search("who signs the i983?", k=3)
    assert hits[0][0]["doc_id"] == "a"


def test_bm25_uses_section_heading():
    hits = BM25Index(_chunks()).search("training plan", k=3)
    assert [h[0]["doc_id"] for h in hits] == ["a"]


def test_bm25_drops_chunks_with_no_shared_word():
    hits = BM25Index(_chunks()).search("unemployment", k=3)
    assert [h[0]["doc_id"] for h in hits] == ["b"]


@pytest.mark.parametrize("query", ["", "zzzz qqqq"])
def test_bm25_query_with_no_known_word_returns_nothing(query):
    assert BM25Index(_chunks()).search(query) == []


def test_bm25_retrieve_returns_passages():
    passages = BM25Index(_chunks()).retrieve("I-765 fee", k=1)
    assert passages[0]["doc_id"] == "c" and passages[0]["chunk_ids"] == ["c-0"]
