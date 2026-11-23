from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Referral(_Strict):
    label: str
    url: HttpUrl | None = None


Reason = Literal["weak_retrieval", "prediction", "case_specific", "legal_strategy", "out_of_scope"]


Label = Literal["answerable", "prediction", "case_specific", "legal_strategy", "out_of_scope"]


class ClassifierExample(_Strict):
    question: str = Field(min_length=1)
    label: Label


class ModuleConfig(_Strict):
    id: str
    covered_topics: list[str] = Field(min_length=1)
    retrieval_threshold: float
    disclaimer: str = Field(min_length=1)
    # Acronym -> spelled-out form, so a direct answer that says "EAD" matches a quote that
    # says "Employment Authorization Document".
    term_aliases: dict[str, str] = {}
    # Terms that name the module's subject (OPT, STEM). A summary may say them without a quote
    # repeating them; every other acronym and name must appear in a quote.
    generic_terms: list[str] = []
    refusals: dict[Reason, str]
    referrals: dict[str, Referral]
    default_referrals: dict[Reason, str]
    classifier_examples: list[ClassifierExample]

    @model_validator(mode="after")
    def _examples_cover_every_label(self):
        missing = set(get_args(Label)) - {e.label for e in self.classifier_examples}
        if missing:
            raise ValueError(f"classifier_examples missing labels: {sorted(missing)}")
        return self

    @model_validator(mode="after")
    def _defaults_cover_every_reason(self):
        missing = set(get_args(Reason)) - set(self.default_referrals)
        if missing:
            raise ValueError(f"default_referrals missing reasons: {sorted(missing)}")
        unknown = set(self.default_referrals.values()) - set(self.referrals)
        if unknown:
            raise ValueError(f"default_referrals name unknown referrals: {sorted(unknown)}")
        return self

    def referral_for(self, reason: Reason, suggested: str | None = None) -> str:
        """Only out_of_scope trusts the classifier's pick (J-1, tax and green card differ);
        every other reason has one fixed destination."""
        if reason == "out_of_scope" and suggested in self.referrals:
            return suggested
        return self.default_referrals[reason]


class NotInForce(_Strict):
    doc_id: str
    reason: str = Field(min_length=1)
    quotes: list[str] = Field(min_length=1)
    # Shown to the student when a question would have retrieved one of these passages.
    notice: str | None = None


class RoleConfig(_Strict):
    model: str
    provider: str
    max_output_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)


class LlmConfig(_Strict):
    classifier: RoleConfig
    generator: RoleConfig


def _read(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_module(path: Path) -> ModuleConfig:
    return ModuleConfig.model_validate(_read(path))


def load_llm(path: Path) -> LlmConfig:
    return LlmConfig.model_validate(_read(path))


def load_not_in_force(path: Path) -> list[NotInForce]:
    """An absent file means every passage is in force."""
    if not Path(path).exists():
        return []
    return [NotInForce.model_validate(e) for e in _read(path) or []]
