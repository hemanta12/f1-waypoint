from pathlib import Path

import pytest

from core.classifier_prompt import Classification
from core.config import load_module
from core.evalset import EvalCase
from core.refusal import Decision, refusal_questions, resolve, score_refusals


def _answerable(id_, question):
    return EvalCase(
        id=id_,
        category="reporting",
        question=question,
        should_refuse=False,
        expected_facts=["10 days"],
        gold=[{"doc_id": "d", "section": "s", "quote": "q"}],
    )


def _refusal(id_, question, reason, referral, paraphrases=()):
    return EvalCase(
        id=id_,
        category="refusal",
        question=question,
        should_refuse=True,
        expected_reason=reason,
        expected_referral=referral,
        paraphrases=list(paraphrases),
        gold=[{"doc_id": "d", "section": "s", "quote": "q"}] if reason == "case_specific" else [],
    )


CASES = [
    _answerable("a1", "how long to report"),
    _refusal("r1", "will i be approved", "prediction", "uscis", ["odds of approval"]),
    _refusal("r2", "am i out of status", "case_specific", "dso"),
]


def test_questions_include_paraphrases():
    assert [q for _, q in refusal_questions(CASES)] == [
        "how long to report",
        "will i be approved",
        "odds of approval",
        "am i out of status",
    ]


def test_perfect_gate_scores_one():
    decisions = {
        "how long to report": Decision(None),
        "will i be approved": Decision("prediction", "uscis"),
        "odds of approval": Decision("prediction", "uscis"),
        "am i out of status": Decision("case_specific", "dso"),
    }
    score = score_refusals(CASES, decisions)
    assert score["refusal_accuracy"] == 1.0
    assert score["reason_accuracy"] == 1.0
    assert score["referral_accuracy"] == 1.0
    assert score["misses"] == []


def test_right_refusal_wrong_reason_and_referral_is_a_miss():
    decisions = {
        "how long to report": Decision("legal_strategy", "attorney"),
        "will i be approved": Decision("weak_retrieval", "dso"),
        "odds of approval": Decision("prediction", "uscis"),
        "am i out of status": Decision("case_specific", "attorney"),
    }
    score = score_refusals(CASES, decisions)
    assert score["refusal_accuracy"] == 0.75
    assert score["reason_accuracy"] == 2 / 3
    assert score["referral_accuracy"] == 1 / 3
    assert {m["question"] for m in score["misses"]} == {
        "how long to report",
        "will i be approved",
        "am i out of status",
    }


def test_missing_decision_counts_as_error_and_miss():
    score = score_refusals(CASES, {"how long to report": Decision(None)})
    assert score["errors"] == 3
    assert score["refusal_accuracy"] == 0.25
    assert score["reason_accuracy"] == 0.0
    assert all(m["error"] for m in score["misses"] if m["question"] != "how long to report")


def test_empty_case_list_has_no_accuracy():
    assert score_refusals([], {})["refusal_accuracy"] is None


MODULE = load_module(
    Path(__file__).resolve().parent.parent / "modules" / "opt_stem" / "module.yaml"
)
STRONG, WEAK = 5.0, -5.0


@pytest.mark.parametrize(
    "score,label,referral,expected",
    [
        (STRONG, "answerable", None, Decision(None)),
        (WEAK, "answerable", None, Decision("weak_retrieval", "dso")),
        (None, "answerable", None, Decision("weak_retrieval", "dso")),
        (STRONG, "case_specific", None, Decision("case_specific", "dso")),
        (WEAK, "case_specific", "dso", Decision("weak_retrieval", "dso")),
        (WEAK, "legal_strategy", None, Decision("legal_strategy", "attorney")),
        (WEAK, "prediction", "dso", Decision("prediction", "uscis")),
        (None, "out_of_scope", "irs", Decision("out_of_scope", "irs")),
    ],
)
def test_resolve_combines_retrieval_score_and_label(score, label, referral, expected):
    assert resolve(MODULE, score, Classification(label=label, referral=referral)) == expected
