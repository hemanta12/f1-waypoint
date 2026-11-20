"""Scorers for the refusal gate: did it refuse the right questions, for the right reason."""

from dataclasses import dataclass

from core.classifier_prompt import Classification
from core.config import ModuleConfig, Reason
from core.evalset import EvalCase


@dataclass(frozen=True)
class Decision:
    reason: Reason | None  # None means the gate let the question through to be answered
    referral: str | None = None


def resolve(module: ModuleConfig, top_score: float | None, label: Classification) -> Decision:
    """Combine the retrieval score and the classifier label into one decision.

    The classifier's prediction, legal_strategy and out_of_scope labels win over weak
    retrieval: those refusals need no retrieved text. case_specific does not, because
    its answer must cite the general rule, so weak retrieval beats it."""
    reason = label.label
    if reason in ("prediction", "legal_strategy", "out_of_scope"):
        return Decision(reason, module.referral_for(reason, label.referral))
    if top_score is None or top_score < module.retrieval_threshold:
        return Decision("weak_retrieval", module.referral_for("weak_retrieval"))
    if reason == "answerable":
        return Decision(None)
    return Decision(reason, module.referral_for(reason, label.referral))


def refusal_questions(cases: list[EvalCase]) -> list[tuple[EvalCase, str]]:
    """Every question the gate is scored on: each case plus its paraphrases."""
    return [(c, q) for c in cases for q in [c.question, *c.paraphrases]]


def score_refusals(cases: list[EvalCase], decisions: dict[str, Decision | None]) -> dict:
    """`decisions` maps question text to the gate's decision. A missing or None decision
    (the gate errored) counts as wrong everywhere and is reported in `errors`."""
    pairs = refusal_questions(cases)
    refused_right = reason_right = referral_right = errors = 0
    should_refuse = 0
    misses = []
    for case, question in pairs:
        got = decisions.get(question)
        if got is None:
            errors += 1
        if case.should_refuse:
            should_refuse += 1
        if got is not None and (got.reason is not None) == case.should_refuse:
            refused_right += 1
        if case.should_refuse and got is not None:
            reason_right += got.reason == case.expected_reason
            referral_right += got.reason is not None and got.referral == case.expected_referral
        wrong = got is None or (
            got.reason != (case.expected_reason if case.should_refuse else None)
            or (case.should_refuse and got.referral != case.expected_referral)
        )
        if wrong:
            misses.append(
                {
                    "id": case.id,
                    "question": question,
                    "expected_reason": case.expected_reason,
                    "expected_referral": case.expected_referral,
                    "got_reason": got.reason if got else None,
                    "got_referral": got.referral if got else None,
                    "error": got is None,
                }
            )
    return {
        "questions": len(pairs),
        "refusal_accuracy": refused_right / len(pairs) if pairs else None,
        "reason_accuracy": reason_right / should_refuse if should_refuse else None,
        "referral_accuracy": referral_right / should_refuse if should_refuse else None,
        "errors": errors,
        "misses": misses,
    }
