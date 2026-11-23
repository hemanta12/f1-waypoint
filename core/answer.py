"""One question in, one response out: the gate, then the generator, then the citation check.

Claims are verbatim quotes. A quote that is not in its chunk shows the passages only; a direct
answer that fails the check shows the verified quotes without it. A failed provider call
shows the retrieved passages with a notice; it raises LlmError only when there are no useful
passages. Nothing is retried.
"""

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from core.citations import Citation, CitedClaim, CoreCheck, check_generation, cite, load_chunks
from core.config import Reason
from core.gate import ClassifierFailed, Gate, Passage, ReferralOut, load_gate
from core.generate import Claim, generate
from core.llm import Kind, LlmError, Usage
from core.registry import load_verified_on

log = logging.getLogger(__name__)

UNAVAILABLE_MESSAGE = (
    "The AI service isn't answering right now, so here are the official passages on this. "
    "Read them below, or try again in a minute."
)
FALLBACK_MESSAGE = (
    "I found official passages on this, but I couldn't confirm an answer from them. "
    "Read the passages below."
)
QUOTES_MESSAGE = (
    "I couldn't summarize this safely, so here is what the official sources say, word for word."
)


class Response(BaseModel):
    status: Literal["answered", "quotes_only", "refused", "rejected", "passages_only"]
    message: str | None = None  # every status but answered
    reason: Reason | None = None  # refused
    referral: ReferralOut | None = None  # refused
    answer: str | None = None
    rule: str | None = None
    claims: list[CitedClaim] = []  # answered, quotes_only, or refused for case_specific
    citations: list[Citation] = []
    not_covered: str | None = None
    disclaimer: str | None = None  # answered, quotes_only
    passages: list[Passage] = []  # passages_only
    notices: list[str] = []  # answered, quotes_only, passages_only: a rule here is in flux
    # Developer detail and cost; not part of what a user sees.
    problems: list[str] = Field(default=[], exclude=True)
    usage: Usage = Field(default=Usage(), exclude=True)
    # Set when a provider call failed; the eval counts these as errors, not fallbacks.
    error: Kind | None = Field(default=None, exclude=True)


class Answerer:
    def __init__(self, gate: Gate, chunks: dict[str, dict], core_check: CoreCheck) -> None:
        self.gate = gate
        self.chunks = chunks
        self.core_check = core_check

    def ask(self, question: str) -> Response:
        try:
            gated = self.gate.run(question)
        except ClassifierFailed as e:
            if not e.passages:
                raise
            return _unavailable(e, e.passages, Usage())
        usage = gated.classifier_usage
        if gated.status == "rejected":
            return Response(status="rejected", message=gated.message, usage=usage)

        refused = gated.status == "refused"
        if refused and (gated.reason != "case_specific" or not gated.passages):
            return _refusal(gated, usage)

        try:
            gen, gen_usage = generate(
                self.gate.llm.generator,
                question,
                gated.passages,
                case_specific=refused,
                client=self.gate.client,
            )
        except LlmError as e:
            if refused:
                log.error("llm %s: %s", e.kind, e)
                plain = _refusal(gated, usage, [f"llm {e.kind}: {e}"])
                return plain.model_copy(update={"error": e.kind})
            return _unavailable(e, gated.passages, usage, gated.notices)
        usage = Usage(
            prompt_tokens=usage.prompt_tokens + gen_usage.prompt_tokens,
            completion_tokens=usage.completion_tokens + gen_usage.completion_tokens,
        )

        if gen.status == "no_support":
            if refused:
                return _refusal(gated, usage)
            return self._weak_retrieval(usage)

        shown = {c for p in gated.passages for c in p.chunk_ids}
        check = check_generation(
            gen,
            shown,
            self.chunks,
            self.core_check,
            self.gate.module.term_aliases,
            self.gate.module.generic_terms,
        )
        if check.quotes:
            if refused:
                return _refusal(gated, usage, check.problems)
            return Response(
                status="passages_only",
                message=FALLBACK_MESSAGE,
                notices=gated.notices,
                passages=gated.passages,
                problems=check.problems,
                usage=usage,
            )

        verified = [
            Claim(text=text, chunk_ids=ids)
            for text, ids in zip(check.texts, check.sources, strict=True)
        ]
        claims, citations = cite(verified, self.chunks)
        summary_ok = not check.summary
        body = dict(
            answer=gen.direct_answer if summary_ok else None,
            rule=gen.rule if summary_ok else None,
            claims=claims,
            citations=citations,
            problems=check.problems + [f"claim {i} snapped to source text" for i in check.snapped],
            usage=usage,
            notices=gated.notices,
        )
        if refused:
            # not_covered is unchecked and, here, tends to echo the user's own numbers.
            return Response(
                status="refused",
                message=gated.message,
                reason=gated.reason,
                referral=gated.referral,
                **body,
            )
        return Response(
            status="answered" if summary_ok else "quotes_only",
            message=None if summary_ok else QUOTES_MESSAGE,
            disclaimer=self.gate.module.disclaimer,
            not_covered=gen.not_covered if summary_ok else None,
            **body,
        )

    def _weak_retrieval(self, usage: Usage) -> Response:
        module = self.gate.module
        key = module.referral_for("weak_retrieval")
        referral = module.referrals[key]
        return Response(
            status="refused",
            message=module.refusals["weak_retrieval"],
            reason="weak_retrieval",
            referral=ReferralOut(
                key=key, label=referral.label, url=str(referral.url) if referral.url else None
            ),
            usage=usage,
        )


def _unavailable(
    e: LlmError, passages: list[Passage], usage: Usage, notices: list[str] | None = None
) -> Response:
    log.error("llm %s: %s", e.kind, e)
    return Response(
        status="passages_only",
        message=UNAVAILABLE_MESSAGE,
        passages=passages,
        notices=notices or [],
        problems=[f"llm {e.kind}: {e}"],
        usage=usage,
        error=e.kind,
    )


def _refusal(gated, usage: Usage, problems: list[str] | None = None) -> Response:
    return Response(
        status="refused",
        message=gated.message,
        reason=gated.reason,
        referral=gated.referral,
        problems=problems or [],
        usage=usage,
    )


def load_answerer(repo: Path, module_dir: Path, core_check: CoreCheck) -> Answerer:
    chunks = load_chunks(module_dir / "index.sqlite")
    dates = load_verified_on(module_dir / "corpus")
    for chunk in chunks.values():
        chunk["verified_on"] = dates.get(chunk["doc_id"])
    return Answerer(load_gate(repo, module_dir), chunks, core_check)
