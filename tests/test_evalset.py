from pathlib import Path

import pytest
from pydantic import ValidationError

from core.evalset import EvalCase, load_eval_set, validate_eval_set

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "modules" / "opt_stem"
CORPUS = MODULE / "corpus"

GOLD = {
    "doc_id": "uscis-stem-opt",
    "section": "Reporting requirements",
    "quote": "Report changes to the following information to your DSO within 10 days of the change",
}


def _answerable(**over):
    return {
        "id": "report-address-change",
        "category": "reporting",
        "question": "I moved. How long do I have to tell anyone?",
        "should_refuse": False,
        "expected_facts": ["10 days", "DSO"],
        "gold": [GOLD],
    } | over


def _refusal(**over):
    return {
        "id": "predict-approval",
        "category": "refusal",
        "question": "Will my STEM OPT get approved?",
        "should_refuse": True,
        "expected_reason": "prediction",
        "expected_referral": "prediction",
        "paraphrases": ["What are my odds of approval?"],
    } | over


def test_valid_cases_load():
    EvalCase.model_validate(_answerable())
    EvalCase.model_validate(_refusal())


@pytest.mark.parametrize(
    "case",
    [
        _answerable(gold=[]),
        _answerable(expected_facts=[]),
        _answerable(expected_reason="prediction"),
        _answerable(paraphrases=["x"]),
        _answerable(should_refuse=True),
        _refusal(expected_referral=None),
        _refusal(category="deadline"),
        _refusal(expected_reason="case_specific"),
        _answerable(unknown_field=1),
        _refusal(expected_reason="made_up"),
    ],
)
def test_bad_combinations_rejected(case):
    with pytest.raises(ValidationError):
        EvalCase.model_validate(case)


def test_case_specific_with_gold_is_valid():
    EvalCase.model_validate(_refusal(expected_reason="case_specific", gold=[GOLD]))


def test_real_quote_passes_and_reflowed_quote_still_matches():
    reflowed = GOLD | {"quote": GOLD["quote"].replace(" ", "\n  ", 3)}
    cases = [
        EvalCase.model_validate(_answerable()),
        EvalCase.model_validate(_answerable(id="b", gold=[reflowed])),
    ]
    problems, _ = validate_eval_set(cases, CORPUS)
    assert problems == []


def test_reports_bad_gold_and_duplicate_ids():
    cases = [
        EvalCase.model_validate(_answerable(gold=[GOLD | {"quote": "within 11 days"}])),
        EvalCase.model_validate(_answerable(gold=[GOLD | {"doc_id": "no-such-doc"}])),
    ]
    problems, _ = validate_eval_set(cases, CORPUS)
    assert "report-address-change: duplicate id" in problems
    assert "report-address-change: quote not found in uscis-stem-opt" in problems
    assert "report-address-change: gold doc_id no-such-doc not in registry" in problems


def test_expected_fact_must_appear_in_a_gold_quote():
    cases = [
        EvalCase.model_validate(_answerable(expected_facts=["10 DAYS", "dso"])),
        EvalCase.model_validate(_answerable(id="b", expected_facts=["90 days", "ask DSO"])),
    ]
    problems, _ = validate_eval_set(cases, CORPUS)
    assert problems == [
        "b: expected fact '90 days' not in any gold quote",
        "b: expected fact 'ask DSO' not in any gold quote",
    ]


def test_count_drift_is_a_warning_not_a_problem():
    problems, warnings = validate_eval_set([EvalCase.model_validate(_answerable())], CORPUS)
    assert problems == []
    assert "reporting: 1 cases, target about 15" in warnings
    assert "prediction: 0 cases, target 3" in warnings


def test_loader_reads_every_file_and_rejects_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path)
    (tmp_path / "a.yaml").write_text(
        "- id: a\n  category: refusal\n  question: q\n  should_refuse: true\n"
        "  expected_reason: out_of_scope\n  expected_referral: out_of_scope\n"
    )
    (tmp_path / "b.yaml").write_text("")
    assert [c.id for c in load_eval_set(tmp_path)] == ["a"]


def test_real_eval_set_is_valid():
    if not list((MODULE / "eval").glob("*.yaml")):
        pytest.skip("no eval cases written yet")
    problems, _ = validate_eval_set(load_eval_set(MODULE / "eval"), CORPUS)
    assert problems == []
