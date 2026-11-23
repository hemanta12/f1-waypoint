"""Run the full answer path (gate, generator, citation check) over the answerable eval cases.

uv run python -m core.answer_eval   # needs a current index and OPENROUTER_API_KEY
"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from core.answer import Response, load_answerer
from core.answer_score import score_answers
from core.citation_core import check_statement
from core.evalset import load_eval_set
from core.gate import GateResult
from core.llm import LlmError, Usage
from core.report import run_record

# Each question makes two calls (classifier, then generator), so the pause also follows the first.
# 3 s still gave upstream 429s on the provider (5 and 10 of 52 cases), so this is longer.
PAUSE_SECONDS = 6


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    module_dir = repo / "modules" / "opt_stem"
    answerer = load_answerer(repo, module_dir, check_statement)
    cases = [c for c in load_eval_set(module_dir / "eval") if not c.should_refuse]

    run_gate = answerer.gate.run

    paused = 0.0

    def paced_gate(question: str) -> GateResult:
        nonlocal paused
        result = run_gate(question)
        time.sleep(PAUSE_SECONDS)
        paused += PAUSE_SECONDS
        return result

    answerer.gate.run = paced_gate

    usage = Usage()
    responses: dict[str, Response | None] = {}
    seconds: dict[str, float] = {}
    for case in cases:
        time.sleep(PAUSE_SECONDS)
        start, paused_before = time.perf_counter(), paused
        try:
            r = answerer.ask(case.question)
        except LlmError as e:
            print(f"error: {case.id}: {e}")
            responses[case.id] = None
            continue
        if r.error:
            print(f"error: {case.id}: {r.error}")
            responses[case.id] = None
            continue
        seconds[case.id] = time.perf_counter() - start - (paused - paused_before)
        usage.prompt_tokens += r.usage.prompt_tokens
        usage.completion_tokens += r.usage.completion_tokens
        responses[case.id] = r

    score = score_answers(cases, responses, answerer.chunks, answerer.gate.module.term_aliases)
    record = run_record(
        repo,
        module_dir / "corpus",
        repo / "llm.yaml",
        {
            "gate": "screen, keyword prefilter, retrieval, passage-aware classifier, resolve",
            "threshold": answerer.gate.module.retrieval_threshold,
            "citation_check": "core.citation_core.check_statement",
        },
    )
    out_dir = repo / "runs" / f"answers-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True)
    graded = []
    for c in cases:
        r = responses[c.id]
        shown = {**r.model_dump(mode="json"), "problems": r.problems} if r else {"status": "error"}
        graded.append(
            {
                "id": c.id,
                "question": c.question,
                "expected_facts": c.expected_facts,
                "seconds": seconds.get(c.id),
                **shown,
            }
        )
    timing = {
        "median_seconds": statistics.median(seconds.values()) if seconds else None,
        "max_seconds": max(seconds.values()) if seconds else None,
    }
    body = {"run": record, "usage": usage.model_dump(), **timing, **score, "responses": graded}
    (out_dir / "results.json").write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    print(
        f"answerable cases {score['answerable_cases']}  errors {score['errors']}\n"
        f"answered {score['answered']}  quotes only {score['quotes_only']}  "
        f"fallback {score['fallback']}  refused {score['refused']}\n"
        f"fallback rate        {pct(score['fallback_rate'])}\n"
        f"citation correctness {pct(score['citation_correctness'])}\n"
        f"deadline accuracy    {pct(score['deadline_accuracy'])}"
        f" ({score['deadline_answered']} answered)\n"
        f"over 120 words       {score['over_120_words']}\n"
        f"forbidden facts      {score['forbidden_stated']}\n"
        f"seconds per question, pauses excluded: median {timing['median_seconds'] or 0:.1f}, "
        f"max {timing['max_seconds'] or 0:.1f}\n"
        f"tokens {usage.prompt_tokens} in, {usage.completion_tokens} out\n"
        f"misses {len(score['misses'])}, written to {out_dir}"
    )


if __name__ == "__main__":
    main()
