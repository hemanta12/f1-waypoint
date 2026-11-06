import json
from pathlib import Path

from core.evalset import EvalCase, load_eval_set
from core.report import corpus_hash, run_record, score_cases, summary_table, write_run

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "modules" / "opt_stem"
CORPUS = MODULE / "corpus"

QUOTE = "Report changes to your DSO within 10 days of the change"


def _case(id_, category="reporting", gold=True):
    if not gold:
        return EvalCase(
            id=id_,
            category="refusal",
            question="q",
            should_refuse=True,
            expected_reason="prediction",
            expected_referral="uscis",
        )
    return EvalCase(
        id=id_,
        category=category,
        question="q",
        should_refuse=False,
        expected_facts=["10 days"],
        gold=[{"doc_id": "d", "section": "s", "quote": QUOTE}],
    )


FAKE = {
    "hit": [("d", "noise"), ("d", QUOTE)],
    "miss": [("d", "noise")] * 5 + [("d", QUOTE)],
}


def test_scores_each_case_and_skips_cases_without_gold():
    cases = [_case("hit"), _case("miss"), _case("no-output"), _case("refuse", gold=False)]
    by_id = {r["id"]: r for r in score_cases(cases, FAKE)}
    assert by_id["hit"]["recall_at_k"] == 1.0
    assert by_id["hit"]["gold_found"] == [True]
    assert by_id["miss"]["recall_at_k"] == 0.0
    assert by_id["miss"]["retrieved_doc_ids"] == ["d"] * 5
    assert by_id["no-output"]["recall_at_k"] == 0.0
    assert by_id["refuse"]["recall_at_k"] is None


def test_summary_table_averages_per_category_and_overall():
    cases = [_case("hit"), _case("miss"), _case("hit2", category="deadline")]
    fake = FAKE | {"hit2": FAKE["hit"]}
    table = summary_table(score_cases(cases, fake))
    assert "| deadline | 1 | 1.00 |" in table
    assert "| reporting | 2 | 0.50 |" in table
    assert "| **All** | 3 | 0.67 |" in table


def test_summary_table_with_nothing_scored():
    assert "| **All** | 0 | n/a |" in summary_table([])


def test_corpus_hash_changes_when_a_document_changes(tmp_path):
    entries = [{"doc_id": "b", "content_hash": "1"}, {"doc_id": "a", "content_hash": "2"}]
    (tmp_path / "registry.json").write_text(json.dumps(entries))
    before = corpus_hash(tmp_path)
    (tmp_path / "registry.json").write_text(json.dumps(list(reversed(entries))))
    assert corpus_hash(tmp_path) == before
    entries[0]["content_hash"] = "3"
    (tmp_path / "registry.json").write_text(json.dumps(entries))
    assert corpus_hash(tmp_path) != before


def test_full_run_on_real_eval_set_with_fake_retrieval(tmp_path):
    cases = load_eval_set(MODULE / "eval")
    fake = {c.id: [(g.doc_id, g.quote) for g in c.gold] for c in cases}
    record = run_record(ROOT, CORPUS, ROOT / "llm.yaml", {"name": "fake"})
    write_run(tmp_path, score_cases(cases, fake), record)

    saved = json.loads((tmp_path / "results.json").read_text())
    assert len(saved["results"]) == len(cases)
    assert len(saved["run"]["git_sha"]) == 40
    assert saved["run"]["corpus_hash"] == corpus_hash(CORPUS)
    assert saved["run"]["llm"]["generator"]["max_output_tokens"] == 600
    assert "| **All** |" in (tmp_path / "summary.md").read_text()
    assert "1.00" in (tmp_path / "summary.md").read_text().splitlines()[-1]
