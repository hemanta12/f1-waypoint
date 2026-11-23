import pytest

from core.citation_core import check_statement

CHUNK = (
    "Post-completion OPT: 90 days. You must report a change within 10 days. "
    "Apply on Form I‐765 up to 90 days before the program end date. "
    "STEM OPT adds a 24-month extension and 60 days of unemployment."
)


@pytest.mark.parametrize(
    "statement",
    [
        "You are allowed 90 days of unemployment.",
        "Report within 10 days.",
        "File Form I-765 (the Unicode hyphen in the source still matches).",
        "The extension is 24 months.",
        "You get ninety days.",
        "No numbers at all in this sentence.",
        "The extension adds 60 days.",
    ],
)
def test_supported_statements_pass(statement):
    assert check_statement(statement, [CHUNK]) == []


@pytest.mark.parametrize(
    ("statement", "expect"),
    [
        ("You are allowed 120 days.", "number 120"),
        ("Report within 100 days.", "number 100"),
        ("File Form I-983 first.", "form I-983"),
        ("Apply on I-20 instead.", "form I-20"),
        ("You get thirty days.", "number 30"),
    ],
)
def test_unsupported_statements_fail(statement, expect):
    problems = check_statement(statement, [CHUNK])
    assert any(expect in p for p in problems)


def test_duration_unit_must_match():
    text = "You have 60 days to leave and 90 hours of training."
    problems = check_statement("You have 90 days.", [text])
    assert problems == ["duration 90 day(s) not in cited chunks"]


def test_number_in_other_chunk_does_not_count():
    assert check_statement("Report within 10 days.", ["Nothing here."]) != []


def test_thousands_separator_and_empty_cited_list():
    assert check_statement("The fee is 1,000 dollars.", ["Fee: $1000."]) == []
    assert check_statement("Report within 10 days.", []) != []
    assert check_statement("No digits here.", []) == []
