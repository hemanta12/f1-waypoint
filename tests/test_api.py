import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from core.answer import Response
from core.citations import Citation, CitedClaim
from core.llm import USER_MESSAGES, LlmError


class FakeAnswerer:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.questions = response, error, []

    def ask(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.response


def client(answerer):
    return TestClient(create_app(lambda: answerer))


ANSWERED = Response(
    status="answered",
    answer="You get 90 days.",
    rule="90 days",
    claims=[CitedClaim(text="90 days.", chunk_ids=["c"], markers=[1])],
    citations=[Citation(n=1, chunk_id="c", doc_id="d", section=None, source_url="https://x.gov")],
    disclaimer="Not legal advice.",
    problems=["internal detail"],
)


def test_ask_returns_the_response_without_developer_fields():
    fake = FakeAnswerer(ANSWERED)
    with client(fake) as c:
        r = c.post("/ask", json={"question": "how many unemployment days?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["citations"][0]["n"] == 1
    assert "problems" not in body and "usage" not in body
    assert fake.questions == ["how many unemployment days?"]


@pytest.mark.parametrize("kind", list(USER_MESSAGES))
def test_provider_failure_returns_plain_words_and_no_detail(kind):
    error = LlmError(kind, "HTTP 500 from vendor/model: secret provider text")
    with client(FakeAnswerer(error=error)) as c:
        r = c.post("/ask", json={"question": "q"})
    assert r.status_code in (429, 500, 502, 503, 504)
    assert r.json() == {"error": kind, "message": USER_MESSAGES[kind]}
    assert "secret" not in r.text


def test_rate_limit_is_429():
    with client(FakeAnswerer(error=LlmError("rate_limited", "x"))) as c:
        assert c.post("/ask", json={"question": "q"}).status_code == 429


def test_bad_request_never_echoes_the_input():
    with client(FakeAnswerer(ANSWERED)) as c:
        r = c.post("/ask", json={"question": 12345, "module": "opt_stem"})
        missing = c.post("/ask", json={})
    assert r.status_code == 422 and "12345" not in r.text
    assert missing.status_code == 422 and missing.json()["error"] == "bad_request"


def test_unknown_module_is_404_and_skips_the_answerer():
    fake = FakeAnswerer(ANSWERED)
    with client(fake) as c:
        r = c.post("/ask", json={"question": "q", "module": "j1"})
    assert r.status_code == 404 and fake.questions == []


def test_health_reports_module_and_corpus_hash():
    with client(FakeAnswerer(ANSWERED)) as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["module"] == "opt_stem"
    assert len(body["corpus_hash"]) == 64
