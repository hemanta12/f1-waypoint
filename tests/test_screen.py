from pathlib import Path

import pytest

from core.evalset import load_eval_set
from core.refusal import refusal_questions
from core.screen import MAX_CHARS, MESSAGES, screen

EVAL = Path(__file__).resolve().parent.parent / "modules" / "opt_stem" / "eval"


@pytest.mark.parametrize(
    "question",
    [
        "my A-number is A123456789, can i work",
        "A 123 456 789 is mine",
        "a-123-456-789",
        "A#12345678",
        "A1234567",
        "my receipt EAC2412345678 is pending",
        "WAC-24-123-45678 status?",
        "lin 2412345678",
        "my SEVIS ID is N0012345678",
        "n-001-234-5678 is my id",
    ],
)
def test_identifiers_are_rejected(question):
    assert screen(question).problem == "identifier"


def test_full_width_a_number_is_rejected():
    assert screen("Ａ１２３４５６７８９ can i work").problem == "identifier"


@pytest.mark.parametrize(
    "question",
    [
        "can i start a 90 day unemployment period on a form I-765",
        "how long is the 24 month STEM extension",
        "my I-20 is dated 2025-05-15, can i apply",
        "what is the 60 day grace period",
        "case number 1234567890 was rejected",
        "i need N95 masks at my workplace for 1234567",
    ],
)
def test_ordinary_numbers_pass(question):
    assert screen(question).ok


def test_length_cap_is_500_after_trimming():
    assert screen("a" * MAX_CHARS).ok
    assert screen("a" * (MAX_CHARS + 1)).problem == "too_long"
    assert screen("  " + "a" * MAX_CHARS + "  ").ok


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_empty_input_is_rejected(question):
    assert screen(question).problem == "empty"


def test_message_never_repeats_the_input():
    result = screen("my A-number is A123456789")
    assert "123456789" not in result.message
    assert all("A1" not in m for m in MESSAGES.values())


def test_no_eval_question_is_falsely_rejected():
    rejected = [q for _, q in refusal_questions(load_eval_set(EVAL)) if not screen(q).ok]
    assert rejected == []
