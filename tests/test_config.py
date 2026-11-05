from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config import LlmConfig, load_llm, load_module

ROOT = Path(__file__).resolve().parent.parent


def test_opt_stem_module_loads():
    module = load_module(ROOT / "modules" / "opt_stem" / "module.yaml")
    assert module.id == "opt_stem"
    assert module.covered_topics


def test_llm_config_loads_both_roles():
    llm = load_llm(ROOT / "llm.yaml")
    assert llm.classifier.max_output_tokens > 0
    assert llm.generator.max_output_tokens > 0


def test_routing_flags_cannot_be_set_in_config():
    role = {"model": "m", "provider": "p", "max_output_tokens": 1, "temperature": 0}
    with pytest.raises(ValidationError):
        LlmConfig.model_validate(
            {
                "classifier": {**role, "allow_fallbacks": True},
                "generator": role,
            }
        )


def test_missing_role_is_rejected():
    with pytest.raises(ValidationError):
        LlmConfig.model_validate({"classifier": {}})
