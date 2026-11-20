import json

import httpx
import pytest

from core.classifier_prompt import Classification
from core.config import RoleConfig
from core.llm import USER_MESSAGES, LlmError, Usage, complete_json

ROLE = RoleConfig(model="vendor/model", provider="acme", max_output_tokens=100, temperature=0)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _reply(content, usage=None):
    body = {"choices": [{"message": {"content": content}}]}
    if usage:
        body["usage"] = usage
    return httpx.Response(200, json=body)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_request_is_pinned_and_strict():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return _reply(
            '{"label": "answerable", "referral": null}',
            {"prompt_tokens": 40, "completion_tokens": 7, "cost": 0.1},
        )

    out, usage = complete_json(ROLE, "sys", "q", Classification, _client(handler))
    assert out == Classification(label="answerable", referral=None)
    assert usage == Usage(prompt_tokens=40, completion_tokens=7)
    body = seen["body"]
    assert body["provider"] == {
        "order": ["acme"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert body["model"] == "vendor/model"
    assert body["max_tokens"] == 100
    assert body["reasoning"] == {"effort": "none"}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert seen["auth"] == "Bearer test-key"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(429, json={"error": "rate limited"}),
        httpx.Response(200, json={"error": "no provider"}),
        httpx.Response(200, json={"choices": []}),
        _reply("not json"),
        _reply('{"label": "weak_retrieval", "referral": null}'),
        _reply('{"label": "answerable", "referral": null, "extra": 1}'),
    ],
)
def test_bad_responses_raise_llm_error(response):
    with pytest.raises(LlmError):
        complete_json(ROLE, "s", "q", Classification, _client(lambda r: response))


@pytest.mark.parametrize(
    "response,kind",
    [
        (httpx.Response(429, json={"error": "slow down"}), "rate_limited"),
        (httpx.Response(408, json={}), "timeout"),
        (httpx.Response(401, json={"error": "bad key"}), "setup"),
        (httpx.Response(402, json={"error": "no credits"}), "setup"),
        (httpx.Response(400, json={"error": "bad request"}), "setup"),
        (httpx.Response(404, json={"error": "no endpoints"}), "unavailable"),
        (httpx.Response(503, json={"error": "down"}), "unavailable"),
        (_reply("not json"), "bad_response"),
        (httpx.Response(200, json={"choices": []}), "bad_response"),
    ],
)
def test_each_failure_has_a_kind_and_a_plain_user_message(response, kind):
    with pytest.raises(LlmError) as caught:
        complete_json(ROLE, "s", "q", Classification, _client(lambda r: response))
    assert caught.value.kind == kind
    assert caught.value.user_message == USER_MESSAGES[kind]


def test_timeout_and_network_failure_are_told_apart():
    def slow(request):
        raise httpx.ReadTimeout("too slow")

    def down(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(LlmError) as timeout:
        complete_json(ROLE, "s", "q", Classification, _client(slow))
    with pytest.raises(LlmError) as unreachable:
        complete_json(ROLE, "s", "q", Classification, _client(down))
    assert (timeout.value.kind, unreachable.value.kind) == ("timeout", "unavailable")


def test_developer_detail_names_the_model_and_provider_but_user_message_does_not():
    response = httpx.Response(429, json={"error": "slow down"})
    with pytest.raises(LlmError) as caught:
        complete_json(ROLE, "s", "q", Classification, _client(lambda r: response))
    assert "vendor/model via acme" in str(caught.value)
    assert "acme" not in caught.value.user_message


def test_unset_role_is_refused_without_a_request():
    unset = ROLE.model_copy(update={"model": "unset"})
    with pytest.raises(LlmError, match="unset"):
        complete_json(unset, "s", "q", Classification, _client(lambda r: pytest.fail("sent")))


def test_missing_key_is_refused(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(LlmError, match="OPENROUTER_API_KEY"):
        complete_json(ROLE, "s", "q", Classification, _client(lambda r: pytest.fail("sent")))


def test_missing_usage_counts_as_zero():
    handler = lambda r: _reply('{"label": "answerable", "referral": null}')  # noqa: E731
    _, usage = complete_json(ROLE, "s", "q", Classification, _client(handler))
    assert usage == Usage()


def test_http_error_includes_response_body():
    response = httpx.Response(404, json={"error": {"message": "no endpoints found"}})
    with pytest.raises(LlmError, match="404.*no endpoints found"):
        complete_json(ROLE, "s", "q", Classification, _client(lambda r: response))
