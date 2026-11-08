import pytest

from core.evalset import GoldPassage
from core.recall import is_hit, recall_at_k

QUOTE = "Report changes to your DSO within 10 days of the change"
GOLD = GoldPassage(doc_id="uscis-stem-opt", section="S", quote=QUOTE)
OTHER = GoldPassage(
    doc_id="dhs-unemployment-counter", section="S", quote="Post-completion OPT: 90 days."
)

HIT = ("uscis-stem-opt", f"Some text. {QUOTE}. More text.")
MISS = ("uscis-stem-opt", "Unrelated text about E-Verify.")


def test_chunk_with_quote_is_a_hit():
    assert is_hit(GOLD, HIT)


def test_whitespace_differences_still_hit():
    assert is_hit(GOLD, ("uscis-stem-opt", QUOTE.replace(" ", "\n   ")))


def test_same_text_in_another_document_is_not_a_hit():
    assert not is_hit(GOLD, ("dhs-stem-opt-faq", QUOTE))


def test_gold_in_top_five_scores_one():
    assert recall_at_k([GOLD], [MISS, MISS, HIT]) == 1.0


def test_gold_only_at_rank_six_scores_zero():
    assert recall_at_k([GOLD], [MISS] * 5 + [HIT]) == 0.0


def test_half_of_two_gold_passages_scores_half():
    assert recall_at_k([GOLD, OTHER], [HIT]) == 0.5


def test_no_retrieved_chunks_scores_zero():
    assert recall_at_k([GOLD], []) == 0.0


def test_gold_found_twice_counts_once():
    assert recall_at_k([GOLD], [HIT, HIT]) == 1.0


def test_no_gold_raises():
    with pytest.raises(ValueError):
        recall_at_k([], [HIT])


def test_alternate_passage_counts_as_a_hit():
    from core.evalset import Passage

    alt = Passage(doc_id="dhs-stem-opt-faq", section="S", quote="within five business days")
    gold = GOLD.model_copy(update={"alternates": [alt]})
    assert is_hit(gold, ("dhs-stem-opt-faq", "notify the DSO within five business days."))
    assert not is_hit(gold, ("uscis-opt", "notify the DSO within five business days."))
