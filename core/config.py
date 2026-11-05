from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Referral(_Strict):
    label: str
    url: HttpUrl


class ModuleConfig(_Strict):
    id: str
    covered_topics: list[str] = Field(min_length=1)
    retrieval_threshold: float
    referrals: dict[str, Referral]


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
