import json
from pathlib import Path

import httpx
import pytest

from core.config import LlmConfig, RoleConfig, load_module
from core.gate import ClassifierFailed, Gate
from core.llm import LlmError

MODULE = load_module(
    Path(__file__).resolve().parent.parent / "modules" / "opt_stem" / "module.yaml"
)
ROLE = RoleConfig(model="vendor/model", provider="acme", max_output_tokens=100, temperature=0)
LLM = LlmConfig(classifier=ROLE, generator=ROLE)
STRONG, WEAK = 5.0, -5.0


def _chunk(text="You must report a change of employer within 10 days."):
    return {
        "chunk_id": "doc-1",
        "doc_id": "doc",
        "section": "Reporting",
        "source_url": "https://example.gov/doc",
        "text": text,
        "pos": 0,
    }


class Spy:
    """Fake search and fake model that record whether they were used."""

    def __init__(self, score, label, referral=None, fail=False):
        self.score, self.label, self.referral, self.fail = score, label, referral, fail
        self.searches, self.bodies = [], []

    def search(self, question):
        self.searches.append(question)
        return [(_chunk(), self.score)]

    def handler(self, request):
        self.bodies.append(json.loads(request.content))
        if self.fail:
            return httpx.Response(500, json={"error": "down"})
        content = json.dumps({"label": self.label, "referral": self.referral})
        reply = {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 9}}
        return httpx.Response(200, json=reply)

    def gate(self):
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        return Gate(MODULE, LLM, self.search, client)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_screen_rejection_skips_search_and_model():
    spy = Spy(STRONG, "answerable")
    result = spy.gate().run("my A-number is A123456789")
    assert (result.status, result.problem) == ("rejected", "identifier")
    assert "123456789" not in result.message
    assert spy.searches == [] and spy.bodies == []


def test_prefilter_refuses_without_search_or_model():
    spy = Spy(STRONG, "answerable")
    result = spy.gate().run("can i apply for a green card during OPT")
    assert (result.status, result.reason, result.referral.key) == (
        "refused",
        "out_of_scope",
        "attorney",
    )
    assert result.message == MODULE.refusals["out_of_scope"]
    assert spy.searches == [] and spy.bodies == []


def test_strong_retrieval_and_answerable_goes_on_with_passages():
    spy = Spy(STRONG, "answerable")
    result = spy.gate().run("how long do i have to report a new employer")
    assert result.status == "answerable"
    assert result.reason is None and result.referral is None
    assert result.passages[0].doc_id == "doc"
    assert result.top_score == STRONG
    assert result.classifier_usage.prompt_tokens == 9


def test_classifier_sees_the_question_and_the_best_passage():
    spy = Spy(STRONG, "answerable")
    spy.gate().run("how long do i have to report")
    user_message = spy.bodies[0]["messages"][1]["content"]
    assert "how long do i have to report" in user_message
    assert "within 10 days" in user_message


def test_weak_retrieval_beats_an_answerable_label():
    result = Spy(WEAK, "answerable").gate().run("can i drive for uber on OPT")
    assert (result.status, result.reason, result.referral.key) == (
        "refused",
        "weak_retrieval",
        "dso",
    )


def test_weak_retrieval_beats_case_specific():
    result = Spy(WEAK, "case_specific", "dso").gate().run("im at 88 days, am i ok")
    assert result.reason == "weak_retrieval"


def test_case_specific_with_strong_retrieval_keeps_passages_for_the_cited_rule():
    result = Spy(STRONG, "case_specific", "dso").gate().run("im at 88 days, am i ok")
    assert (result.reason, result.referral.key) == ("case_specific", "dso")
    assert result.passages and result.message == MODULE.refusals["case_specific"]


def test_legal_strategy_beats_weak_retrieval():
    result = Spy(WEAK, "legal_strategy", "dso").gate().run("whats the best trick to reset my days")
    assert (result.reason, result.referral.key) == ("legal_strategy", "attorney")


def test_out_of_scope_uses_the_classifiers_referral():
    result = Spy(WEAK, "out_of_scope", "j1_responsible_officer").gate().run("im on a j-1")
    assert result.referral.key == "j1_responsible_officer"
    assert result.referral.url is None


def test_classifier_failure_is_raised_with_passages_only_when_retrieval_was_strong():
    with pytest.raises(ClassifierFailed) as strong:
        Spy(STRONG, "answerable", fail=True).gate().run("how long does OPT last")
    assert [p.chunk_ids for p in strong.value.passages] == [["doc-1"]]
    assert isinstance(strong.value, LlmError)
    with pytest.raises(ClassifierFailed) as weak:
        Spy(WEAK, "answerable", fail=True).gate().run("how long does OPT last")
    assert weak.value.passages == []


def test_no_search_results_is_weak_retrieval():
    gate = Spy(STRONG, "answerable").gate()
    gate.search = lambda q: []
    assert gate.run("anything about OPT").reason == "weak_retrieval"


def test_not_in_force_quotes_mark_their_chunks_and_a_stale_quote_fails():
    from core.config import NotInForce
    from core.gate import not_in_force_ids

    chunks = [
        _chunk("You have an additional  30-day period to depart.") | {"chunk_id": "doc-1"},
        _chunk("You must report within 10 days.") | {"chunk_id": "doc-2"},
    ]
    entry = NotInForce(doc_id="doc", reason="postponed", quotes=["additional 30-day period"])
    assert not_in_force_ids(chunks, [entry]) == {"doc-1"}
    stale = entry.model_copy(update={"quotes": ["a sentence that is gone"]})
    with pytest.raises(ValueError, match="matches no doc chunk"):
        not_in_force_ids(chunks, [stale])


def test_real_not_in_force_quotes_are_in_the_corpus_text():
    from core.config import load_not_in_force
    from core.registry import normalize_text

    module_dir = Path(__file__).resolve().parent.parent / "modules" / "opt_stem"
    for entry in load_not_in_force(module_dir / "not_in_force.yaml"):
        text = normalize_text((module_dir / "corpus" / "text" / f"{entry.doc_id}.txt").read_text())
        missing = [q[:60] for q in entry.quotes if normalize_text(q) not in text]
        assert missing == [], f"{entry.doc_id}: quotes not in the corpus text: {missing}"


def test_a_not_in_force_passage_is_dropped_and_leaves_its_notice():
    spy = Spy(STRONG, "answerable")
    stale = {**_chunk("The grace period is 30 days."), "chunk_id": "old-1", "not_in_force": True}
    stale["notice"] = "A postponed rule would change this."
    spy.search = lambda q: [(stale, 6.0), (_chunk(), STRONG)]
    result = spy.gate().run("how long is the grace period after OPT")
    assert result.notices == ["A postponed rule would change this."]
    assert [p.chunk_ids for p in result.passages] == [["doc-1"]]
    assert "30 days" not in spy.bodies[0]["messages"][-1]["content"]


def test_a_weakly_matching_not_in_force_passage_leaves_no_notice():
    spy = Spy(STRONG, "answerable")
    stale = {**_chunk("Border commuter students."), "chunk_id": "old-1", "not_in_force": True}
    stale["notice"] = "A postponed rule would change this."
    spy.search = lambda q: [(_chunk(), STRONG), (stale, 1.0)]
    assert spy.gate().run("how do i report a new employer").notices == []
