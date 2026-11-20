from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config import LlmConfig, ModuleConfig, load_llm, load_module

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


def test_referral_comes_from_the_reason_not_the_model():
    module = load_module(ROOT / "modules" / "opt_stem" / "module.yaml")
    assert module.referral_for("case_specific", None) == "dso"
    assert module.referral_for("case_specific", "attorney") == "dso"
    assert module.referral_for("legal_strategy", "dso") == "attorney"
    assert module.referral_for("prediction", None) == "uscis"


def test_out_of_scope_uses_the_classifier_pick_when_valid():
    module = load_module(ROOT / "modules" / "opt_stem" / "module.yaml")
    assert module.referral_for("out_of_scope", "irs") == "irs"
    assert module.referral_for("out_of_scope", None) == "dso"
    assert module.referral_for("out_of_scope", "made-up") == "dso"


def test_default_referral_must_name_a_real_referral():
    module = load_module(ROOT / "modules" / "opt_stem" / "module.yaml").model_dump(mode="json")
    module["default_referrals"]["prediction"] = "nobody"
    with pytest.raises(ValidationError):
        ModuleConfig.model_validate(module)


def test_every_label_needs_a_classifier_example():
    module = load_module(ROOT / "modules" / "opt_stem" / "module.yaml").model_dump(mode="json")
    module["classifier_examples"] = [
        e for e in module["classifier_examples"] if e["label"] != "case_specific"
    ]
    with pytest.raises(ValidationError, match="case_specific"):
        ModuleConfig.model_validate(module)
