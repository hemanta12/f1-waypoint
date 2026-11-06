import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.registry import Document, normalize_text

Category = Literal["deadline", "reporting", "i983", "employer", "near_miss", "refusal"]
Reason = Literal["weak_retrieval", "prediction", "case_specific", "legal_strategy", "out_of_scope"]

TARGET_COUNTS = {
    "deadline": 20,
    "reporting": 15,
    "i983": 5,
    "employer": 5,
    "near_miss": 5,
    "refusal": 15,
}
TARGET_PER_REASON = 3


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldPassage(_Strict):
    doc_id: str
    section: str = Field(min_length=1)
    # Verbatim from the corpus text; whitespace differences are ignored.
    quote: str = Field(min_length=1)


class EvalCase(_Strict):
    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    category: Category
    question: str = Field(min_length=1)
    should_refuse: bool
    expected_facts: list[str] = []
    gold: list[GoldPassage] = []
    expected_reason: Reason | None = None
    expected_referral: str | None = None
    paraphrases: list[str] = []

    @model_validator(mode="after")
    def _check_combination(self):
        if self.should_refuse != (self.category == "refusal"):
            raise ValueError("should_refuse must be true exactly when category is refusal")
        if self.should_refuse:
            if self.expected_reason is None or not self.expected_referral:
                raise ValueError("a refusal needs expected_reason and expected_referral")
            if self.expected_reason == "case_specific" and not self.gold:
                raise ValueError("a case_specific refusal needs gold for the general rule")
        else:
            if self.expected_reason is not None or self.expected_referral is not None:
                raise ValueError("an answerable case has no expected_reason or expected_referral")
            if not self.gold or not self.expected_facts:
                raise ValueError("an answerable case needs gold and expected_facts")
            if self.paraphrases:
                raise ValueError("paraphrases are for refusal cases only")
        return self


def load_eval_set(eval_dir: Path) -> list[EvalCase]:
    files = sorted(Path(eval_dir).glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no eval files in {eval_dir}")
    cases: list[EvalCase] = []
    for f in files:
        for raw in yaml.safe_load(f.read_text()) or []:
            cases.append(EvalCase.model_validate(raw))
    return cases


def validate_eval_set(cases: list[EvalCase], corpus_dir: Path) -> tuple[list[str], list[str]]:
    """Return (problems, warnings). Problems fail the check; warnings are count drift."""
    problems: list[str] = []
    ids = Counter(c.id for c in cases)
    problems += [f"{i}: duplicate id" for i, n in sorted(ids.items()) if n > 1]

    registry = {
        d["doc_id"]: Document.model_validate(d)
        for d in json.loads((corpus_dir / "registry.json").read_text())
    }
    texts: dict[str, str] = {}
    for c in cases:
        for g in c.gold:
            doc = registry.get(g.doc_id)
            if doc is None:
                problems.append(f"{c.id}: gold doc_id {g.doc_id} not in registry")
                continue
            if doc.status == "removed":
                problems.append(f"{c.id}: gold doc_id {g.doc_id} is removed")
            if g.doc_id not in texts:
                path = corpus_dir / "text" / f"{g.doc_id}.txt"
                texts[g.doc_id] = normalize_text(path.read_text(encoding="utf-8"))
            if normalize_text(g.quote) not in texts[g.doc_id]:
                problems.append(f"{c.id}: quote not found in {g.doc_id}")
        quotes = [normalize_text(g.quote).lower() for g in c.gold]
        for fact in c.expected_facts:
            if not any(normalize_text(fact).lower() in q for q in quotes):
                problems.append(f"{c.id}: expected fact {fact!r} not in any gold quote")

    warnings: list[str] = []
    by_category = Counter(c.category for c in cases)
    for cat, target in TARGET_COUNTS.items():
        if by_category[cat] != target:
            warnings.append(f"{cat}: {by_category[cat]} cases, target about {target}")
    by_reason = Counter(c.expected_reason for c in cases if c.should_refuse)
    for reason in Reason.__args__:
        if by_reason[reason] != TARGET_PER_REASON:
            warnings.append(f"{reason}: {by_reason[reason]} cases, target {TARGET_PER_REASON}")
    return problems, warnings


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "modules" / "opt_stem"
    loaded = load_eval_set(root / "eval")
    found, drift = validate_eval_set(loaded, root / "corpus")
    for p in found:
        print(p)
    for w in drift:
        print(f"warning: {w}")
    print(f"{len(loaded)} cases OK" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
