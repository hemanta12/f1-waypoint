from core.answer import Response
from core.answer_score import facts_present, score_answers
from core.citations import Citation, CitedClaim
from core.evalset import EvalCase, GoldPassage

CHUNKS = {
    "c": {"doc_id": "doc", "text": "You get 90 days of unemployment."},
    "x": {"doc_id": "doc", "text": "Unrelated text."},
}


def case(id, category="deadline", facts=("90 days",), section="General Information"):
    return EvalCase(
        id=id,
        category=category,
        question="q",
        should_refuse=False,
        expected_facts=list(facts),
        gold=[GoldPassage(doc_id="doc", section=section, quote="90 days")],
    )


def answered(text="You get 90 days.", chunk="c"):
    return Response(
        status="answered",
        answer=text,
        claims=[CitedClaim(text=text, chunk_ids=[chunk], markers=[1])],
        citations=[Citation(n=1, chunk_id=chunk, doc_id="doc", section=None, source_url="u")],
    )


def test_facts_match_on_numbers_not_word_order():
    r = answered("An additional 60 days are allowed.")
    assert facts_present(["60 additional days"], r)
    assert not facts_present(["90 days"], r)
    assert facts_present(["additional 60 days"], r)


def test_fact_without_numbers_must_appear_as_text():
    r = answered("Count from the Actual OPT Start Date.")
    assert facts_present(["actual opt start date"], r)
    assert not facts_present(["EAD start date"], r)


def test_scores_each_outcome_separately():
    cases = [case("a"), case("b"), case("c"), case("d"), case("e")]
    responses = {
        "a": answered(),
        "b": answered("You get 30 days."),
        "c": answered(chunk="x"),
        "d": Response(status="passages_only", problems=["30 not in cited chunks"]),
        "e": Response(status="refused", reason="weak_retrieval"),
    }
    s = score_answers(cases, responses, CHUNKS)
    assert (s["answered"], s["fallback"], s["refused"]) == (3, 1, 1)
    assert s["fallback_rate"] == 1 / 4
    assert s["citation_correctness"] == 2 / 3
    assert s["deadline_accuracy"] == 2 / 3
    outcomes = {m["id"]: m["outcome"] for m in s["misses"]}
    assert outcomes == {
        "b": "deadline_fact_missing",
        "c": "cites_wrong_passage",
        "d": "fallback",
        "e": "refused:weak_retrieval",
    }


def test_missing_response_is_an_error_and_empty_denominators_are_none():
    s = score_answers([case("a")], {}, CHUNKS)
    assert s["errors"] == 1
    assert s["fallback_rate"] is None and s["deadline_accuracy"] is None


def test_long_answers_are_counted():
    s = score_answers([case("a")], {"a": answered("word " * 130 + "90 days")}, CHUNKS)
    assert s["over_120_words"] == 1


def test_forbidden_fact_is_reported_even_when_the_expected_fact_is_there():
    c = case("a").model_copy(update={"forbidden_facts": ["30-day"]})
    s = score_answers(
        [c], {"a": answered("You get 90 days, or an additional 30-day period.")}, CHUNKS
    )
    assert s["forbidden_stated"] == 1
    assert {"id": "a", "outcome": "forbidden_fact_stated"} in s["misses"]
    assert score_answers([c], {"a": answered()}, CHUNKS)["forbidden_stated"] == 0


def test_month_in_a_fact_must_match_too():
    assert not facts_present(["April 1"], answered("It lasts until October 1."))
    assert facts_present(["April 1"], answered("It lasts until April 1 of the next year."))


def test_quotes_only_counts_as_a_cited_answer_not_a_fallback():
    r = answered()
    r.status, r.answer = "quotes_only", None
    s = score_answers([case("a")], {"a": r}, CHUNKS)
    assert (s["answered"], s["quotes_only"], s["fallback"]) == (0, 1, 0)
    assert s["citation_correctness"] == 1.0 and s["fallback_rate"] == 0.0


def test_acronym_matches_its_spelled_out_form_in_a_fact():
    r = answered("Wait until you receive your EAD card.")
    fact = ["receive your Employment Authorization Document"]
    aliases = {"EAD": "Employment Authorization Document"}
    assert facts_present(fact, r, aliases)
    assert not facts_present(fact, r)
    assert not facts_present(["receive your Social Security card"], r, aliases)
