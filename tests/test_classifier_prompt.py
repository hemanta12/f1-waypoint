from typing import get_args

import pytest
from pydantic import ValidationError

from core.classifier_prompt import (
    MAX_PASSAGE_CHARS,
    SYSTEM_PROMPT,
    Classification,
    Label,
    Referral,
    classifier_input,
    system_prompt,
)
from core.config import ClassifierExample


def test_prompt_names_every_label_and_referral():
    for name in (*get_args(Label), *get_args(Referral)):
        assert f"- {name}:" in SYSTEM_PROMPT


def test_answerable_may_have_no_referral():
    assert Classification(label="answerable", referral=None).referral is None


def test_unknown_label_is_rejected():
    with pytest.raises(ValidationError):
        Classification(label="weak_retrieval", referral=None)


def test_system_prompt_lists_the_modules_examples():
    examples = [ClassifierExample(question="is it allowed?", label="answerable")]
    prompt = system_prompt(examples)
    assert prompt.startswith(SYSTEM_PROMPT)
    assert "- is it allowed? => answerable" in prompt
    assert "best-matching passage" in prompt


def test_classifier_input_has_question_and_cut_passage():
    message = classifier_input("how long?", "x" * (MAX_PASSAGE_CHARS + 500))
    assert message.startswith("Question: how long?")
    assert message.count("x") == MAX_PASSAGE_CHARS


@pytest.mark.parametrize("passage", [None, ""])
def test_classifier_input_without_a_passage_says_so(passage):
    assert "(none found)" in classifier_input("how long?", passage)
