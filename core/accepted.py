"""The last accepted eval run, and the check that llm.yaml still matches it.

uv run python -m core.accepted check
uv run python -m core.accepted accept runs/refusal-<timestamp>

A change to a model, provider or token cap changes what the eval measured, so it needs a new
accepted run before it can pass CI.
"""

import json
import sys
from pathlib import Path

from core.config import LlmConfig, load_llm

REPO = Path(__file__).resolve().parent.parent
ACCEPTED = REPO / "modules" / "opt_stem" / "accepted_run.json"
METRICS = ("questions", "refusal_accuracy", "reason_accuracy", "referral_accuracy", "errors")


def differences(llm: LlmConfig, accepted: dict) -> list[str]:
    """One line per llm.yaml field that no longer matches the accepted run."""
    found = []
    now = llm.model_dump()
    for role, fields in accepted["llm"].items():
        for name, value in fields.items():
            if now[role][name] != value:
                found.append(f"{role}.{name}: accepted {value!r}, llm.yaml has {now[role][name]!r}")
    return found


def accept(run_dir: Path, out: Path = ACCEPTED) -> dict:
    """Record a finished eval run as the one llm.yaml is checked against."""
    body = json.loads((run_dir / "results.json").read_text())
    if body["errors"]:
        raise SystemExit(f"{run_dir.name} had {body['errors']} errors; accept a clean run.")
    record = {
        "accepted_from": run_dir.name,
        "created_at": body["run"]["created_at"],
        "git_sha": body["run"]["git_sha"],
        "threshold": body["run"]["retriever"]["threshold"],
        "llm": body["run"]["llm"],
        "metrics": {k: body[k] for k in METRICS},
    }
    out.write_text(json.dumps(record, indent=2) + "\n")
    return record


def check(llm_path: Path = REPO / "llm.yaml", accepted_path: Path = ACCEPTED) -> list[str]:
    if not accepted_path.exists():
        return [f"no accepted run at {accepted_path}; run the eval, then accept it"]
    return differences(load_llm(llm_path), json.loads(accepted_path.read_text()))


def main() -> None:
    command, *rest = sys.argv[1:] or ["check"]
    if command == "accept" and len(rest) == 1:
        record = accept(Path(rest[0]))
        print(f"accepted {record['accepted_from']}: {record['metrics']}")
    elif command == "check":
        problems = check()
        if problems:
            print("llm.yaml no longer matches the last accepted eval run:")
            print("\n".join(f"  {p}" for p in problems))
            print("Run `uv run python -m core.refusal_eval`, then `core.accepted accept <run>`.")
            raise SystemExit(1)
        print("llm.yaml matches the last accepted eval run.")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
