import pytest

from core.prefilter import prefilter


@pytest.mark.parametrize(
    "question,expected",
    [
        ("how do i file my 1040-NR tax return as an F-1 student", ("out_of_scope", "irs")),
        ("which tax forms do i file as a nonresident on OPT", ("out_of_scope", "irs")),
        ("can i use turbotax on F-1", ("out_of_scope", "irs")),
        ("can i apply for a green card during OPT", ("out_of_scope", "attorney")),
        ("how does PERM labor certification work for F-1 students", ("out_of_scope", "attorney")),
    ],
)
def test_subject_keywords_are_refused(question, expected):
    assert prefilter(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "how long does OPT last",
        "will my OPT application get approved",
        "what are my chances USCIS approves my STEM OPT",
        "is my OPT going to come before my start date",
        "what taxes are withheld from my OPT pay",
        "does my employer need to file my I-983 on time",
        "im on a J-1, how does academic training work",
        "",
    ],
)
def test_wording_alone_never_triggers_a_refusal(question):
    assert prefilter(question) is None
