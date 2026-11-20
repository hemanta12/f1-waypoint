"""Input screen: runs before anything else, rejects what the tool must never process."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

MAX_CHARS = 500

# A-number: "A" then 7 to 9 digits, with optional dashes, spaces or "#" between them.
_A_NUMBER = re.compile(r"\bA[-\s#]?\d(?:[-\s]?\d){6,8}\b", re.I)
# Receipt number: a 3-letter service center code then 10 digits (EAC2412345678,
# WAC-24-123-45678).
_RECEIPT = re.compile(r"\b[A-Z]{3}[-\s]?\d(?:[-\s]?\d){9}\b", re.I)

# SEVIS ID: "N" then 10 digits.
_SEVIS_ID = re.compile(r"\bN(?:[-\s]?\d){10}\b", re.I)

Problem = Literal["empty", "too_long", "identifier"]

# None of these messages repeat the user's text, so a rejected identifier is never echoed.
MESSAGES: dict[Problem, str] = {
    "empty": "Type a question about OPT or STEM OPT.",
    "too_long": f"Please keep your question under {MAX_CHARS} characters.",
    "identifier": (
        "Please remove any A-number, receipt number or SEVIS ID. This tool doesn't need "
        "them and doesn't store them."
    ),
}


@dataclass(frozen=True)
class ScreenResult:
    problem: Problem | None

    @property
    def ok(self) -> bool:
        return self.problem is None

    @property
    def message(self) -> str | None:
        return MESSAGES[self.problem] if self.problem else None


def screen(question: str) -> ScreenResult:
    text = question.strip()
    if not text:
        return ScreenResult("empty")
    if len(text) > MAX_CHARS:
        return ScreenResult("too_long")
    # NFKC turns full-width digits and letters into plain ones before matching.
    folded = unicodedata.normalize("NFKC", text)
    if any(p.search(folded) for p in (_A_NUMBER, _RECEIPT, _SEVIS_ID)):
        return ScreenResult("identifier")
    return ScreenResult(None)
