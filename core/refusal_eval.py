"""Run the refusal gate over the eval set and score it.

uv run python -m core.refusal_eval   # needs a current index and OPENROUTER_API_KEY
"""

import json
import time
from datetime import datetime
from pathlib import Path

from core.evalset import load_eval_set
from core.gate import load_gate
from core.llm import LlmError, Usage
from core.refusal import Decision, refusal_questions, score_refusals
from core.report import run_record

# One request every three seconds keeps a provider's rate limit from turning into 429 errors.
PAUSE_SECONDS = 3


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    module_dir = repo / "modules" / "opt_stem"
    gate = load_gate(repo, module_dir)
    cases = load_eval_set(module_dir / "eval")
    usage = Usage()

    decisions: dict[str, Decision | None] = {}
    for _, question in refusal_questions(cases):
        time.sleep(PAUSE_SECONDS)
        try:
            result = gate.run(question)
        except LlmError as e:
            print(f"error: {question[:50]}: {e}")
            decisions[question] = None
            continue
        usage.prompt_tokens += result.classifier_usage.prompt_tokens
        usage.completion_tokens += result.classifier_usage.completion_tokens
        if result.status == "rejected":
            decisions[question] = None
        else:
            key = result.referral.key if result.referral else None
            decisions[question] = Decision(result.reason, key)

    score = score_refusals(cases, decisions)
    record = run_record(
        repo,
        module_dir / "corpus",
        repo / "llm.yaml",
        {
            "gate": "screen, keyword prefilter, retrieval, passage-aware classifier, resolve",
            "threshold": gate.module.retrieval_threshold,
        },
    )
    out_dir = repo / "runs" / f"refusal-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True)
    body = {"run": record, "classifier_usage": usage.model_dump(), **score}
    (out_dir / "results.json").write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")

    print(
        f"questions {score['questions']}  errors {score['errors']}\n"
        f"refusal accuracy  {score['refusal_accuracy']:.3f}\n"
        f"reason accuracy   {score['reason_accuracy']:.3f}\n"
        f"referral accuracy {score['referral_accuracy']:.3f}\n"
        f"classifier tokens {usage.prompt_tokens} in, {usage.completion_tokens} out\n"
        f"misses {len(score['misses'])}, written to {out_dir}"
    )


if __name__ == "__main__":
    main()
