import re

# Subject names only. Wording that merely sounds like a forecast or a case is the
# classifier's job, so no verb or pronoun patterns belong here.
_RULES = [
    ("out_of_scope", "irs", [r"\b1040\b", r"\btax(es)? (return|form)s?\b", r"\bturbotax\b"]),
    ("out_of_scope", "attorney", [r"\bgreen card\b", r"\blabor certification\b"]),
]

_COMPILED = [(r, ref, [re.compile(p, re.I) for p in ps]) for r, ref, ps in _RULES]


def prefilter(question: str) -> tuple[str, str] | None:
    for reason, referral, patterns in _COMPILED:
        if any(p.search(question) for p in patterns):
            return reason, referral
    return None
