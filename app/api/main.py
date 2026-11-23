"""The HTTP API.

uv run uvicorn app.api.main:app

A question is never echoed back: validation errors carry no input, and a provider failure
returns only the plain-words `user_message`.
"""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.answer import Answerer, Response, load_answerer
from core.citation_core import check_statement
from core.llm import Kind, LlmError
from core.report import corpus_hash

REPO = Path(__file__).resolve().parents[2]
MODULE_ID = "opt_stem"

STATUS: dict[Kind, int] = {
    "rate_limited": 429,
    "timeout": 504,
    "unavailable": 503,
    "bad_response": 502,
    "setup": 500,
}

log = logging.getLogger("uvicorn.error")


class AskRequest(BaseModel):
    question: str
    module: str = MODULE_ID


def _load() -> Answerer:
    return load_answerer(REPO, REPO / "modules" / MODULE_ID, check_statement)


def create_app(load: Callable[[], Answerer] = _load) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.answerer = load()
        app.state.corpus_hash = corpus_hash(REPO / "modules" / MODULE_ID / "corpus")
        yield

    app = FastAPI(title="F-1 Waypoint", lifespan=lifespan)

    @app.exception_handler(LlmError)
    def llm_error(_: Request, e: LlmError) -> JSONResponse:
        log.error("llm %s: %s", e.kind, e)
        return JSONResponse(
            status_code=STATUS[e.kind], content={"error": e.kind, "message": e.user_message}
        )

    @app.exception_handler(RequestValidationError)
    def bad_request(_: Request, __: RequestValidationError) -> JSONResponse:
        # The default 422 body repeats the submitted input.
        return JSONResponse(
            status_code=422,
            content={"error": "bad_request", "message": "Send a question as text."},
        )

    @app.post("/ask", response_model=Response)
    def ask(body: AskRequest, request: Request) -> Response:
        if body.module != MODULE_ID:
            raise HTTPException(404, f"Only the {MODULE_ID} module is available.")
        return request.app.state.answerer.ask(body.question)

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", "module": MODULE_ID, "corpus_hash": request.app.state.corpus_hash}

    return app


app = create_app()
