"""The core citation check: every number, duration and form ID in a statement must appear
verbatim in a cited chunk.

It does not judge meaning. A statement can pass with the right numbers and still say
the wrong thing; hand-grading covers that.
"""

import re

_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-"})
_FORM = re.compile(r"\b[A-Z]{1,2}-\d{1,4}[A-Z]?\b")
_NUMBER = re.compile(r"(?<![\d.])\d+(?:,\d{3})*(?:\.\d+)?")
_DURATION = re.compile(
    r"(\d+(?:,\d{3})*(?:\.\d+)?)(?:[\s-]+[a-z]+){0,2}?[\s-]+(day|week|month|year|hour)s?\b"
)
# "one" is left out: it is mostly a pronoun ("one of the following").
_WORDS = {
    w: str(n)
    for n, w in enumerate(
        "two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen twenty".split(),
        start=2,
    )
} | {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
     "eighty": "80", "ninety": "90"}  # fmt: skip
_WORD = re.compile(r"\b(" + "|".join(_WORDS) + r")\b")


def _normalize(text: str) -> str:
    """Unicode dashes to "-", spelled-out numbers to digits ("ninety days" is "90 days")."""
    return _WORD.sub(lambda m: _WORDS[m.group(1)], text.translate(_DASHES).lower())


def _number(raw: str) -> str:
    return raw.replace(",", "")


def check_statement(statement: str, cited_texts: list[str]) -> list[str]:
    """Problems found; empty means every number, duration and form ID is in a cited chunk.

    A number must be a whole number in a chunk ("10" does not match "100"). A duration must
    also match its unit: "90 days" fails against a chunk that has "60 days" and "90 hours".
    A cited section number such as "214.2" is checked like any other number, so a claim
    that cites a CFR section fails unless the chunk text carries it."""
    chunk = _normalize(" ".join(cited_texts))
    forms = {f.lower() for f in _FORM.findall(" ".join(cited_texts).translate(_DASHES))}
    numbers = {_number(n) for n in _NUMBER.findall(chunk)}
    durations = {(_number(n), u) for n, u in _DURATION.findall(chunk)}

    stated = statement.translate(_DASHES)
    problems = [
        f"form {f} not in cited chunks"
        for f in dict.fromkeys(_FORM.findall(stated))
        if f.lower() not in forms
    ]
    rest = _normalize(_FORM.sub(" ", stated))
    for raw in dict.fromkeys(_NUMBER.findall(rest)):
        if _number(raw) not in numbers:
            problems.append(f"number {raw} not in cited chunks")
    for raw, unit in dict.fromkeys(_DURATION.findall(rest)):
        n = _number(raw)
        if n in numbers and (n, unit) not in durations:
            problems.append(f"duration {raw} {unit}(s) not in cited chunks")
    return problems
