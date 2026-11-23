import json

import pytest

from core.accepted import ACCEPTED, REPO, accept, check, differences
from core.config import LlmConfig, RoleConfig, load_llm

ROLE = RoleConfig(model="a/model", provider="acme", max_output_tokens=100, temperature=0)


def _accepted(llm=None):
    llm = llm or LlmConfig(classifier=ROLE, generator=ROLE)
    return {"llm": llm.model_dump()}


def test_matching_config_has_no_differences():
    assert differences(LlmConfig(classifier=ROLE, generator=ROLE), _accepted()) == []


def test_changed_model_and_token_cap_are_reported_by_role_and_field():
    changed = ROLE.model_copy(update={"model": "b/model", "max_output_tokens": 200})
    found = differences(LlmConfig(classifier=changed, generator=ROLE), _accepted())
    assert found == [
        "classifier.model: accepted 'a/model', llm.yaml has 'b/model'",
        "classifier.max_output_tokens: accepted 100, llm.yaml has 200",
    ]


def test_accept_records_the_run_and_refuses_a_run_with_errors(tmp_path):
    run = tmp_path / "refusal-1"
    run.mkdir()
    body = {
        "run": {
            "created_at": "t",
            "git_sha": "abc",
            "retriever": {"threshold": -2.0},
            "llm": _accepted()["llm"],
        },
        "questions": 97,
        "refusal_accuracy": 0.9,
        "reason_accuracy": 0.8,
        "referral_accuracy": 0.8,
        "errors": 0,
    }
    (run / "results.json").write_text(json.dumps(body))
    record = accept(run, tmp_path / "accepted.json")
    assert record["accepted_from"] == "refusal-1" and record["threshold"] == -2.0
    assert json.loads((tmp_path / "accepted.json").read_text()) == record

    (run / "results.json").write_text(json.dumps({**body, "errors": 2}))
    with pytest.raises(SystemExit, match="2 errors"):
        accept(run, tmp_path / "other.json")


def test_missing_accepted_run_is_a_failure(tmp_path):
    assert "no accepted run" in check(REPO / "llm.yaml", tmp_path / "missing.json")[0]


def test_llm_yaml_matches_the_last_accepted_run():
    problems = check()
    assert problems == [], (
        "llm.yaml changed since the last accepted eval run:\n  "
        + "\n  ".join(problems)
        + "\nRun the eval, then: uv run python -m core.accepted accept runs/<run>"
    )
    assert load_llm(REPO / "llm.yaml") and ACCEPTED.exists()


def test_accept_takes_an_answers_run(tmp_path):
    run = tmp_path / "answers-1"
    run.mkdir()
    body = {
        "run": {
            "created_at": "t",
            "git_sha": "abc",
            "retriever": {"threshold": -2.0},
            "llm": _accepted()["llm"],
        },
        "answerable_cases": 52,
        "fallback_rate": 0.02,
        "citation_correctness": 0.9,
        "deadline_accuracy": 0.95,
        "errors": 0,
    }
    (run / "results.json").write_text(json.dumps(body))
    record = accept(run, tmp_path / "accepted.json")
    assert record["metrics"]["fallback_rate"] == 0.02 and "questions" not in record["metrics"]
