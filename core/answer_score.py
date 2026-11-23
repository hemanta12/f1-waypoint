"""Scorers for generated answers: deadline accuracy, citation correctness, fallback rate."""

import re

from core.answer import Response
from core.evalset import EvalCase
from core.recall import is_hit

WORD_LIMIT = 120
_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
)


def _answer_text(r: Response) -> str:
    return " ".join([r.answer or "", r.rule or "", *(c.text for c in r.claims)])


def _spell_out(text: str, aliases: dict[str, str]) -> str:
    """Lowercase, with each acronym replaced by its spelled-out form, so "EAD" matches
    "Employment Authorization Document" on either side."""
    for acronym, spelled in aliases.items():
        text = re.sub(rf"\b{re.escape(acronym)}s?\b", spelled, text)
    return text.lower()


def facts_present(facts: list[str], r: Response, aliases: dict[str, str] | None = None) -> bool:
    """A fact with numbers is present when every one of its numbers is in the answer ("60
    additional days" matches "an additional 60 days"); a fact without numbers must appear as
    text, with acronyms spelled out on both sides. Hand-grading is the check on this leniency."""
    aliases = aliases or {}
    text = _answer_text(r)
    spelled = _spell_out(text, aliases)
    numbers = set(re.findall(r"\d+", text))
    months = set(_MONTH.findall(text.lower()))
    for fact in facts:
        wanted = re.findall(r"\d+", fact)
        # "April 1" must not pass on an answer that says "October 1".
        ok = (
            set(wanted) <= numbers and set(_MONTH.findall(fact.lower())) <= months
            if wanted
            else _spell_out(fact, aliases) in spelled
        )
        if not ok:
            return False
    return True


def cites_gold(case: EvalCase, r: Response, chunks: dict[str, dict]) -> bool:
    """A cited chunk holds a gold quote, the same test Recall@5 uses. Section names are not
    compared: the index splits sections finer than the gold section."""
    return any(
        is_hit(g, (chunks[c.chunk_id]["doc_id"], chunks[c.chunk_id]["text"]))
        for g in case.gold
        for c in r.citations
    )


def score_answers(
    cases: list[EvalCase],
    responses: dict[str, Response | None],
    chunks: dict[str, dict],
    aliases: dict[str, str] | None = None,
) -> dict:
    """`responses` maps an answerable case id to its response. None means the call errored.
    Refusal cases are scored by the gate scorers, not here."""
    answerable = [c for c in cases if not c.should_refuse]
    answered = quotes_only = fallback = refused = errors = cited = 0
    deadline_answered = deadline_right = over_words = forbidden = 0
    misses = []
    for case in answerable:
        r = responses.get(case.id)
        if r is None or r.status == "rejected":
            errors += 1
            misses.append({"id": case.id, "outcome": "error"})
        elif r.status == "passages_only":
            fallback += 1
            misses.append({"id": case.id, "outcome": "fallback", "problems": r.problems})
        elif r.status == "refused":
            refused += 1
            misses.append({"id": case.id, "outcome": f"refused:{r.reason}"})
        else:
            # quotes_only is a cited answer without the summary sentence; it is scored the same.
            answered += r.status == "answered"
            quotes_only += r.status == "quotes_only"
            over_words += len(_answer_text(r).split()) > WORD_LIMIT
            if case.forbidden_facts and facts_present(case.forbidden_facts, r, aliases):
                forbidden += 1
                misses.append({"id": case.id, "outcome": "forbidden_fact_stated"})
            right = cites_gold(case, r, chunks)
            cited += right
            if not right:
                misses.append({"id": case.id, "outcome": "cites_wrong_passage"})
            if case.category == "deadline":
                deadline_answered += 1
                ok = facts_present(case.expected_facts, r, aliases)
                deadline_right += ok
                if not ok:
                    misses.append({"id": case.id, "outcome": "deadline_fact_missing"})
    shown = answered + quotes_only
    checked = shown + fallback
    return {
        "answerable_cases": len(answerable),
        "answered": answered,
        "quotes_only": quotes_only,
        "fallback": fallback,
        "refused": refused,
        "errors": errors,
        "fallback_rate": fallback / checked if checked else None,
        "citation_correctness": cited / shown if shown else None,
        "over_120_words": over_words,
        "forbidden_stated": forbidden,
        "deadline_answered": deadline_answered,
        "deadline_accuracy": deadline_right / deadline_answered if deadline_answered else None,
        "misses": misses,
    }
