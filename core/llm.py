"""OpenRouter chat completions, pinned to one provider with no fallback routing."""

import json
import os
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

from core.config import RoleConfig

URL = "https://openrouter.ai/api/v1/chat/completions"
UNSET = "unset"


Kind = Literal["rate_limited", "timeout", "unavailable", "bad_response", "setup"]

# What the user sees. It names the cause in plain words and never repeats their question or
# any provider text.
USER_MESSAGES: dict[Kind, str] = {
    "rate_limited": "The AI service is busy right now. Please wait a minute and try again.",
    "timeout": "The AI service took too long to answer. Please try again.",
    "unavailable": (
        "The AI service is not responding right now. Please try again in a few minutes."
    ),
    "bad_response": "The AI service sent back an answer this tool couldn't read. Please try again.",
    "setup": (
        "This tool can't reach its AI service because of a setup problem on our side. "
        "Trying again won't help."
    ),
}

# Statuses that mean our request or credentials are wrong, not that the service is busy.
_SETUP_STATUSES = {400, 401, 402, 403}


class LlmError(Exception):
    """`str(error)` is the developer detail for logs; `user_message` is safe to show."""

    def __init__(self, kind: Kind, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind

    @property
    def user_message(self) -> str:
        return USER_MESSAGES[self.kind]


def _status_error(resp: httpx.Response, role: RoleConfig) -> LlmError:
    where = f"{role.model} via {role.provider}"
    detail = f"HTTP {resp.status_code} from {where}: {resp.text[:300]}"
    if resp.status_code == 429:
        return LlmError("rate_limited", detail)
    if resp.status_code == 408:
        return LlmError("timeout", detail)
    if resp.status_code in _SETUP_STATUSES:
        return LlmError("setup", detail)
    return LlmError("unavailable", detail)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


def complete_json[T: BaseModel](
    role: RoleConfig,
    system: str,
    user: str,
    schema: type[T],
    client: httpx.Client | None = None,
) -> tuple[T, Usage]:
    if UNSET in (role.model, role.provider):
        raise LlmError("setup", "llm.yaml model and provider are unset for this role")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LlmError("setup", "OPENROUTER_API_KEY is not set")

    body = {
        "model": role.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": role.max_output_tokens,
        "reasoning": {"effort": "none"},
        "temperature": role.temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        },
        # require_parameters: structured output support is per provider endpoint.
        "provider": {
            "order": [role.provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    try:
        resp = (client or httpx).post(
            URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=30
        )
        if resp.is_error:
            raise _status_error(resp, role)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = Usage.model_validate(data.get("usage") or {})
        return schema.model_validate(json.loads(content)), usage
    except httpx.TimeoutException as e:
        raise LlmError("timeout", f"no answer in 30 s from {role.provider}: {e}") from e
    except httpx.HTTPError as e:
        raise LlmError("unavailable", f"{type(e).__name__} reaching {role.provider}: {e}") from e
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as e:
        raise LlmError("bad_response", f"{type(e).__name__}: {e}") from e
