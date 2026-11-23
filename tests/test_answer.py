import json
from pathlib import Path

import httpx
import pytest

from core.answer import FALLBACK_MESSAGE, QUOTES_MESSAGE, UNAVAILABLE_MESSAGE, Answerer
from core.citations import check_generation, cite, key_terms
from core.config import LlmConfig, RoleConfig, load_module
from core.gate import ClassifierFailed, GateResult, Passage, ReferralOut
from core.generate import Claim, Generation, generator_input, system_prompt
from core.llm import LlmError, Usage

MODULE = load_module(
    Path(__file__).resolve().parent.parent / "modules" / "opt_stem" / "module.yaml"
)
ROLE = RoleConfig(model="vendor/model", provider="acme", max_output_tokens=600, temperature=0)
LLM = LlmConfig(classifier=ROLE, generator=ROLE)

TEXT = (
    "Post-completion OPT: students may accrue 90 days of unemployment during the "
    "initial period.[60] Report a change of employer to your DSO within 10 days."
)
QUOTE = "students may accrue 90 days of unemployment during the initial period"
CHUNKS = {
    "doc-1": {
        "chunk_id": "doc-1",
        "doc_id": "doc",
        "section": "General Information",
        "source_url": "https://example.gov/doc",
        "text": TEXT,
    },
    "other-1": {
        "chunk_id": "other-1",
        "doc_id": "other",
        "section": None,
        "source_url": "https://example.gov/other",
        "text": "Students must report a change of address to the DSO within 10 days.",
    },
}
PASSAGE = Passage(
    doc_id="doc",
    section="General Information",
    source_url="https://example.gov/doc",
    chunk_ids=["doc-1"],
    text=TEXT,
)


def number_check(statement, texts):
    """Stand-in for the hand-written check: every number must appear in a cited chunk."""
    import re

    joined = " ".join(texts)
    return [f"{n} not in cited chunks" for n in re.findall(r"\d+", statement) if n not in joined]


def gen(**over):
    base = dict(
        status="answered",
        direct_answer="You can be unemployed for 90 days.",
        rule="90 days",
        claims=[Claim(text=QUOTE, chunk_ids=["doc-1"])],
        not_covered=None,
    )
    return Generation(**{**base, **over})


class FakeGate:
    def __init__(self, result, generation=None, fail=False):
        self.result, self.generation, self.fail = result, generation, fail
        self.module, self.llm = MODULE, LLM
        self.bodies = []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handler))

    def run(self, question):
        return self.result

    def _handler(self, request):
        self.bodies.append(json.loads(request.content))
        if self.fail:
            return httpx.Response(500, json={"error": "down"})
        content = self.generation.model_dump_json()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 7}},
        )


def answerable():
    return GateResult(status="answerable", passages=[PASSAGE], classifier_usage=Usage())


def answerable_with_both():
    other = PASSAGE.model_copy(update={"doc_id": "other", "chunk_ids": ["other-1"]})
    return GateResult(status="answerable", passages=[PASSAGE, other], classifier_usage=Usage())


def case_specific():
    return GateResult(
        status="refused",
        reason="case_specific",
        message=MODULE.refusals["case_specific"],
        referral=ReferralOut(key="dso", label="Your school's DSO", url=None),
        passages=[PASSAGE],
    )


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def ask(gate):
    return Answerer(gate, CHUNKS, number_check).ask("how long can i be unemployed")


def test_good_answer_is_numbered_and_carries_the_disclaimer():
    r = ask(FakeGate(answerable(), gen()))
    assert r.status == "answered"
    assert r.disclaimer == MODULE.disclaimer
    assert r.claims[0].markers == [1]
    assert r.citations[0].chunk_id == "doc-1" and r.citations[0].section == "General Information"


def test_paraphrased_quote_falls_back_to_passages_only():
    bad = gen(
        claims=[
            Claim(
                text="students may accrue 30 days of unemployment during the initial period",
                chunk_ids=["doc-1"],
            )
        ]
    )
    r = ask(FakeGate(answerable(), bad))
    assert r.status == "passages_only"
    assert r.message == FALLBACK_MESSAGE
    assert r.passages == [PASSAGE]
    assert r.answer is None and r.claims == []
    assert any("not verbatim" in p for p in r.problems)
    assert "problems" not in r.model_dump()


def test_invented_fact_without_a_number_is_caught():
    made_up = "students must also notify their employer about every day of unemployment"
    check = check_generation(
        gen(claims=[Claim(text=made_up, chunk_ids=["doc-1"])]), {"doc-1"}, CHUNKS, number_check
    )
    assert check.quotes


def test_quote_matching_ignores_case_marks_footnotes_and_end_punctuation():
    quote = (
        "Post\u2010completion OPT: STUDENTS may accrue 90 days of unemployment "
        "during the initial period."
    )
    check = check_generation(
        gen(claims=[Claim(text=quote, chunk_ids=["doc-1"])]), {"doc-1"}, CHUNKS, number_check
    )
    assert check.problems == []
    spans_footnote = "during the initial period. Report a change of employer to your DSO"
    check = check_generation(
        gen(
            claims=[Claim(text=spans_footnote, chunk_ids=["doc-1"])],
            direct_answer="Tell your DSO.",
            rule=None,
        ),
        {"doc-1"},
        CHUNKS,
        number_check,
    )
    assert check.problems == []


def test_short_quote_fails():
    check = check_generation(
        gen(claims=[Claim(text="90 days of unemployment", chunk_ids=["doc-1"])]),
        {"doc-1"},
        CHUNKS,
        number_check,
    )
    assert any("under 8 words" in p for p in check.quotes)


def test_only_chunks_holding_the_quote_are_cited():
    claim = Claim(text=QUOTE, chunk_ids=["other-1", "doc-1"])
    r = ask(FakeGate(answerable_with_both(), gen(claims=[claim])))
    assert r.status == "answered"
    assert [c.chunk_id for c in r.citations] == ["doc-1"]


def test_chunk_not_shown_or_invented_or_missing_falls_back():
    for ids in (["other-1"], ["nope-9"], []):
        bad = gen(claims=[Claim(text=QUOTE, chunk_ids=ids)])
        assert check_generation(bad, {"doc-1"}, CHUNKS, number_check).quotes
    other = gen(claims=[Claim(text=QUOTE, chunk_ids=["other-1"])])
    assert ask(FakeGate(answerable(), other)).status == "passages_only"


def test_unsupported_direct_answer_shows_the_verified_quotes_alone():
    r = ask(FakeGate(answerable(), gen(direct_answer="You can be unemployed for 120 days.")))
    assert r.status == "quotes_only"
    assert r.message == QUOTES_MESSAGE
    assert r.answer is None and r.rule is None and r.not_covered is None
    assert r.claims[0].text == QUOTE and r.citations[0].chunk_id == "doc-1"
    assert r.disclaimer == MODULE.disclaimer
    assert any("direct_answer" in p for p in r.problems)


def test_echoed_question_number_in_direct_answer_shows_quotes_alone():
    r = ask(FakeGate(answerable(), gen(direct_answer="At 85 days you are within the 90 days.")))
    assert r.status == "quotes_only"


def test_answered_without_claims_fails():
    assert check_generation(gen(claims=[]), {"doc-1"}, CHUNKS, number_check).quotes


def test_no_support_becomes_a_weak_retrieval_refusal():
    empty = Generation(
        status="no_support", direct_answer=None, rule=None, claims=[], not_covered=None
    )
    r = ask(FakeGate(answerable(), empty))
    assert (r.status, r.reason) == ("refused", "weak_retrieval")
    assert r.referral.key == MODULE.default_referrals["weak_retrieval"]
    assert r.claims == []


def test_case_specific_refusal_carries_cited_claims_and_uses_the_extra_rule():
    fake = FakeGate(
        case_specific(), gen(direct_answer="The rule is 90 days.", not_covered="Your 85 days?")
    )
    r = ask(fake)
    assert (r.status, r.reason) == ("refused", "case_specific")
    assert r.message == MODULE.refusals["case_specific"]
    assert r.claims and r.citations
    assert r.disclaimer is None
    assert r.not_covered is None
    assert "Do not give a verdict" in fake.bodies[0]["messages"][0]["content"]


def test_case_specific_with_a_failed_quote_is_still_a_plain_refusal():
    bad = gen(
        claims=[
            Claim(
                text="It is 30 days of unemployment for every single student.", chunk_ids=["doc-1"]
            )
        ]
    )
    r = ask(FakeGate(case_specific(), bad))
    assert (r.status, r.reason, r.claims) == ("refused", "case_specific", [])
    assert r.problems


def test_case_specific_with_a_failed_summary_keeps_the_quoted_rule():
    r = ask(FakeGate(case_specific(), gen(direct_answer="The rule is 85 days.")))
    assert (r.status, r.reason) == ("refused", "case_specific")
    assert r.answer is None and r.claims[0].text == QUOTE


def test_other_refusals_and_rejections_skip_the_generator():
    for result in (
        GateResult(status="refused", reason="prediction", message="no", passages=[PASSAGE]),
        GateResult(status="rejected", problem="too_long", message="too long"),
    ):
        fake = FakeGate(result)
        r = ask(fake)
        assert r.status == result.status
        assert fake.bodies == []


def test_generator_failure_shows_passages_with_a_notice_without_retry():
    fake = FakeGate(answerable(), fail=True)
    r = ask(fake)
    assert len(fake.bodies) == 1
    assert (r.status, r.message, r.passages) == ("passages_only", UNAVAILABLE_MESSAGE, [PASSAGE])
    assert r.error == "unavailable"
    assert "error" not in r.model_dump() and r.answer is None


def test_generator_failure_on_case_specific_is_a_plain_refusal():
    r = ask(FakeGate(case_specific(), fail=True))
    assert (r.status, r.reason, r.claims) == ("refused", "case_specific", [])
    assert r.error == "unavailable"


def test_classifier_failure_shows_its_passages_or_raises_when_there_are_none():
    def gate_failing_with(passages):
        fake = FakeGate(answerable())

        def run(question):
            raise ClassifierFailed(LlmError("rate_limited", "HTTP 429"), passages)

        fake.run = run
        return fake

    r = ask(gate_failing_with([PASSAGE]))
    assert (r.status, r.message, r.error) == ("passages_only", UNAVAILABLE_MESSAGE, "rate_limited")
    with pytest.raises(LlmError):
        ask(gate_failing_with([]))


def test_markers_follow_first_use_and_share_chunks():
    claims = [
        Claim(text="a", chunk_ids=["other-1", "doc-1"]),
        Claim(text="b", chunk_ids=["doc-1", "doc-1"]),
    ]
    cited, citations = cite(claims, CHUNKS)
    assert [c.markers for c in cited] == [[1, 2], [2]]
    assert [c.chunk_id for c in citations] == ["other-1", "doc-1"]


def test_prompt_carries_passages_with_chunk_ids_and_not_the_case_rule_by_default():
    assert '[chunk_ids: ["doc-1"]] doc | General Information' in generator_input("q?", [PASSAGE])
    assert "Do not give a verdict" not in system_prompt()
    assert "Do not give a verdict" in system_prompt(case_specific=True)


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("The clock starts on the EAD start date, not graduation day.", ["EAD"]),
        ("No, the SEVP Portal is only for post-completion OPT.", ["SEVP", "SEVP Portal", "OPT"]),
        ("First tell your DSO, then submit a new Form I-983.", ["DSO", "Form I-983"]),
        ("Count from the Actual OPT Start Date.", ["OPT", "Actual OPT Start Date"]),
        ("Students file EADs early.", ["EAD"]),
        ("You have 60 days after your OPT EAD expires.", ["OPT", "EAD"]),
        ("It covers STEM OPT students.", ["STEM", "OPT"]),
    ],
)
def test_key_terms(text, terms):
    assert key_terms(text) == terms


def test_swapped_term_in_direct_answer_shows_quotes_alone():
    quote = "SEVIS counts days by looking back to the Actual OPT Start Date for that period"
    chunks = {**CHUNKS, "doc-1": {**CHUNKS["doc-1"], "text": quote + "."}}
    swapped = gen(
        claims=[Claim(text=quote, chunk_ids=["doc-1"])],
        direct_answer="The clock starts on the EAD start date.",
        rule="Actual OPT Start Date",
    )
    check = check_generation(swapped, {"doc-1"}, chunks, number_check, MODULE.term_aliases)
    assert check.summary == ["direct_answer: term 'EAD' not in quotes"]
    same = swapped.model_copy(update={"direct_answer": "It starts on the Actual OPT Start Date."})
    assert (
        check_generation(same, {"doc-1"}, chunks, number_check, MODULE.term_aliases).problems == []
    )


def test_acronym_matches_its_spelled_out_form_in_a_quote():
    quote = "You may not begin work before you receive your Employment Authorization Document"
    chunks = {**CHUNKS, "doc-1": {**CHUNKS["doc-1"], "text": quote}}
    g = gen(
        claims=[Claim(text=quote, chunk_ids=["doc-1"])],
        direct_answer="Wait for your EAD.",
        rule=None,
    )
    assert check_generation(g, {"doc-1"}, chunks, number_check, MODULE.term_aliases).problems == []


def test_rule_must_be_copied_from_a_quote():
    check = check_generation(gen(rule="ninety days total"), {"doc-1"}, CHUNKS, number_check)
    assert "rule is not copied from a quote" in check.summary


def test_generic_subject_terms_need_no_quote_but_other_acronyms_still_do():
    quote = "Students must report the termination of their training within 10 days of the event"
    chunks = {**CHUNKS, "doc-1": {**CHUNKS["doc-1"], "text": quote + "."}}
    claims = [Claim(text=quote, chunk_ids=["doc-1"])]
    topic = gen(claims=claims, direct_answer="OPT students must report within 10 days.", rule=None)
    args = ({"doc-1"}, chunks, number_check, MODULE.term_aliases, MODULE.generic_terms)
    assert check_generation(topic, *args).problems == []
    recipient = gen(claims=claims, direct_answer="Report to your DSO within 10 days.", rule=None)
    assert check_generation(recipient, *args).summary == ["direct_answer: term 'DSO' not in quotes"]


def test_citation_carries_its_documents_verified_on_date():
    from datetime import date

    chunks = {k: {**v, "verified_on": date(2026, 11, 5)} for k, v in CHUNKS.items()}
    r = Answerer(FakeGate(answerable(), gen()), chunks, number_check).ask("q")
    assert r.citations[0].verified_on == date(2026, 11, 5)
    assert r.model_dump(mode="json")["citations"][0]["verified_on"] == "2026-11-05"


def test_quote_with_a_slip_snaps_to_the_source_text_and_is_checked_against_it():
    slipped = "students can accrue 90 days of unemployment during the initial period"
    r = ask(FakeGate(answerable(), gen(claims=[Claim(text=slipped, chunk_ids=["doc-1"])])))
    assert r.status == "answered"
    source = "students may accrue 90 days of unemployment during the initial period.[60]"
    assert r.claims[0].text == source
    assert "claim 1 snapped to source text" in r.problems


@pytest.mark.parametrize(
    "quote",
    [
        "students may accrue 30 days of unemployment during the initial period",  # number
        "students must notify the employer about each day of unemployment in the period",
        "students may accrue unemployment",  # too short
    ],
)
def test_quotes_that_do_not_snap(quote):
    from core.citations import snap_quote

    assert snap_quote(quote, TEXT) is None


def test_elided_quote_shows_the_whole_source_span():
    from core.citations import snap_quote

    text = "Report these to your DSO within 10 days: your name; your address; your email address."
    quote = "Report these to your DSO within 10 days: ... your email address"
    assert snap_quote(quote, text) == text
    assert snap_quote("Report these to your DSO ... your phone number", text) is None
