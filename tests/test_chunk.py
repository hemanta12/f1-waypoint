from pathlib import Path

import pytest

from core.chunk import fixed_size_chunks, merge_small_sections, split_section, structure_chunks
from core.snapshot import extract_sections

TEXT = " ".join(f"w{i}" for i in range(10))


def test_empty_text_gives_no_chunks():
    assert fixed_size_chunks("d", "   \n ") == []


def test_short_text_is_one_chunk():
    chunks = fixed_size_chunks("d", "a  b\nc", size=5, overlap=1)
    assert [c.text for c in chunks] == ["a b c"]
    assert chunks[0].chunk_id == "d#w0"


def test_windows_overlap_and_cover_every_word():
    chunks = fixed_size_chunks("d", TEXT, size=4, overlap=1)
    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert [c.chunk_id for c in chunks] == ["d#w0", "d#w3", "d#w6"]


def test_no_tiny_trailing_chunk_that_repeats_the_last_one():
    chunks = fixed_size_chunks("d", TEXT, size=5, overlap=0)
    assert [c.text for c in chunks] == ["w0 w1 w2 w3 w4", "w5 w6 w7 w8 w9"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (4, 4), (4, -1)])
def test_bad_sizes_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        fixed_size_chunks("d", TEXT, size=size, overlap=overlap)


def _words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def test_small_sections_fold_into_the_next_one():

    sections = [("Breadcrumb", "Home Students"), ("General", "x"), ("Rule", _words(30, "r"))]
    assert merge_small_sections(sections) == [("Rule", f"Home Students\n\nx\n\n{_words(30, 'r')}")]


def test_small_last_section_folds_into_the_one_before():

    sections = [("Rule", _words(30, "r")), ("Contact", "call us")]
    assert merge_small_sections(sections) == [("Rule", f"{_words(30, 'r')}\n\ncall us")]


def test_all_small_sections_still_keep_their_text():

    assert merge_small_sections([("A", "a"), ("B", "b")]) == [("B", "a\n\nb")]
    assert merge_small_sections([]) == []


def test_merging_the_real_corpus_drops_no_text():

    for raw in Path("modules/opt_stem/corpus/raw").glob("*.html"):
        sections = extract_sections(raw)
        before = " ".join(t for _, t in sections).split()
        after = " ".join(t for _, t in merge_small_sections(sections)).split()
        assert after == before, raw.name


def test_split_keeps_lines_whole_and_respects_max():

    text = "\n".join([_words(4, "a"), _words(4, "b"), _words(4, "c")])
    assert split_section(text, 8) == [f"{_words(4, 'a')}\n{_words(4, 'b')}", _words(4, "c")]


def test_long_line_splits_at_sentence_ends_then_words():

    assert split_section("One two three. Four five six.", 4) == ["One two three.", "Four five six."]
    assert split_section(_words(10, "w"), 4) == [_words(4, "w"), "w4 w5 w6 w7", "w8 w9"]


def test_structure_chunk_ids_come_from_sections_and_stay_unique():

    sections = [("Practical Training", _words(40, "a")), ("Practical Training", _words(40, "b"))]
    chunks = structure_chunks("ice", sections, max_words=30)
    assert [c.chunk_id for c in chunks] == [
        "ice#practical-training/1",
        "ice#practical-training/2",
        "ice#practical-training-2/1",
        "ice#practical-training-2/2",
    ]
    assert all(c.section == "Practical Training" for c in chunks)
