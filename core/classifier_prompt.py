from typing import Literal

from pydantic import BaseModel, ConfigDict

from core.config import ClassifierExample, Label, Reason

Referral = Literal["dso", "uscis", "attorney", "irs", "j1_responsible_officer"]


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Label
    referral: Referral | None


# `weak_retrieval` is decided by the score threshold, never by the classifier.
assert set(Label.__args__) - {"answerable"} <= set(Reason.__args__)

SYSTEM_PROMPT = """\
You sort questions for a tool that answers F-1 OPT and STEM OPT questions from official \
sources. You do not answer the question. You return one label.

Labels:
- answerable: asks what a rule, form, deadline or process is, or how something works. This \
is the default. Mentioning the user's own job, status or situation as context does not change \
it ("I'm on STEM OPT and moved, how long do I have to report?" is answerable).
- prediction: asks for a guess about a future agency decision, timing or odds \
("will I be approved", "when will my card arrive"). Asking how long a rule's deadline is \
is not a prediction.
- case_specific: gives the user's own facts, numbers or dates and asks for a verdict on THEM: \
whether they are in status, late, eligible or in trouble ("I'm at 88 days, am I ok"). \
Without the user's own facts it is not case_specific.
- legal_strategy: asks how to get around a rule, how to word an explanation to an agency, or \
which legal path is best ("how do I avoid the 90 day limit"). Asking what a rule requires \
is not strategy.
- out_of_scope: about something other than F-1 OPT or STEM OPT (J-1, taxes, green cards, \
H-1B beyond OPT).

Questions like "can I", "does X count", "is it allowed" or "do I have to" ask what a rule \
says, so they are answerable unless they also give the user's own facts and ask for a verdict.

Use answerable unless the question clearly fits another label. If it fits several, use the \
first in this order: legal_strategy, prediction, case_specific, out_of_scope.

Referral (the one place to send the user):
- dso: school or status questions
- uscis: decisions or timing
- attorney: legal strategy, green card
- irs: tax
- j1_responsible_officer: J-1
Use null when the label is answerable.

Return only JSON: {"label": "...", "referral": "..." or null}"""

PASSAGE_RULE = (
    "\n\nYou may also be shown the best-matching passage from the official sources. "
    "If that passage directly states the rule the question asks about, the question is "
    "answerable."
)

MAX_PASSAGE_CHARS = 1200


def system_prompt(examples: list[ClassifierExample]) -> str:
    """The shared definitions, then this module's own labeled examples."""
    lines = "\n".join(f"- {e.question} => {e.label}" for e in examples)
    return f"{SYSTEM_PROMPT}{PASSAGE_RULE}\n\nExamples (question => label):\n{lines}"


def classifier_input(question: str, passage: str | None) -> str:
    """The user message: the question and the best-matching passage, cut to a fixed length."""
    shown = (passage or "")[:MAX_PASSAGE_CHARS] or "(none found)"
    return f"Question: {question}\n\nBest-matching passage:\n{shown}"
