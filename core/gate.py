"""The refusal gate: one question in, one decision out.

Order: screen, subject-keyword prefilter, retrieval, then the passage-aware classifier. The
score threshold and the classifier's label are combined by `resolve`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from core.bm25 import BM25Index
from core.classifier_prompt import Classification, classifier_input, system_prompt
from core.config import LlmConfig, ModuleConfig, Reason, load_llm, load_module
from core.index import Index, clean_results, rrf
from core.llm import Usage, complete_json
from core.prefilter import prefilter
from core.refusal import resolve
from core.rerank import Reranker
from core.screen import Problem, screen

FETCH = 10
KEEP = 5
RERANKER = "Xenova/ms-marco-MiniLM-L-12-v2"

# Ranked (chunk, reranker score) pairs, best first.
Search = Callable[[str], list[tuple[dict, float]]]


class Passage(BaseModel):
    doc_id: str
    section: str | None
    source_url: str
    chunk_ids: list[str]
    text: str


class ReferralOut(BaseModel):
    key: str
    label: str
    url: str | None


class GateResult(BaseModel):
    status: Literal["rejected", "refused", "answerable"]
    problem: Problem | None = None  # rejected
    reason: Reason | None = None  # refused
    message: str | None = None  # text for the user when rejected or refused
    referral: ReferralOut | None = None  # refused
    passages: list[Passage] = []  # empty when rejected or caught by the prefilter
    top_score: float | None = None
    classifier_usage: Usage = Usage()


class Gate:
    def __init__(
        self,
        module: ModuleConfig,
        llm: LlmConfig,
        search: Search,
        client: httpx.Client | None = None,
    ) -> None:
        self.module = module
        self.llm = llm
        self.search = search
        self.client = client
        self.prompt = system_prompt(module.classifier_examples)

    def run(self, question: str) -> GateResult:
        """Raises LlmError if the classifier call fails. No retry: show `error.user_message`."""
        checked = screen(question)
        if not checked.ok:
            return GateResult(status="rejected", problem=checked.problem, message=checked.message)
        if hit := prefilter(question):
            return self._refused(hit[0], hit[1], [], None, Usage())

        ranked = self.search(question)
        passages = clean_results(ranked, KEEP)
        top_score = ranked[0][1] if ranked else None
        message = classifier_input(question, passages[0]["text"] if passages else None)
        label, usage = complete_json(
            self.llm.classifier, self.prompt, message, Classification, self.client
        )
        decision = resolve(self.module, top_score, label)
        if decision.reason is None:
            return GateResult(
                status="answerable",
                passages=_passages(passages),
                top_score=top_score,
                classifier_usage=usage,
            )
        return self._refused(decision.reason, decision.referral, passages, top_score, usage)

    def _refused(
        self,
        reason: Reason,
        suggested: str | None,
        passages: list[dict],
        top_score: float | None,
        usage: Usage,
    ) -> GateResult:
        key = self.module.referral_for(reason, suggested)
        referral = self.module.referrals[key]
        return GateResult(
            status="refused",
            reason=reason,
            message=self.module.refusals[reason],
            referral=ReferralOut(
                key=key, label=referral.label, url=str(referral.url) if referral.url else None
            ),
            passages=_passages(passages),
            top_score=top_score,
            classifier_usage=usage,
        )


def _passages(raw: list[dict]) -> list[Passage]:
    return [Passage.model_validate({k: p[k] for k in Passage.model_fields}) for p in raw]


def load_gate(repo: Path, module_dir: Path) -> Gate:
    from fastembed import TextEmbedding

    module = load_module(module_dir / "module.yaml")
    index = Index(module_dir / "index.sqlite")
    embedder = TextEmbedding(index.meta["embedding_model"])
    bm25 = BM25Index(index.chunks)
    reranker = Reranker(RERANKER)

    def search(question: str) -> list[tuple[dict, float]]:
        vector = next(iter(embedder.query_embed([question])))
        pool = rrf(index.search(vector, FETCH), bm25.search(question, FETCH))
        return reranker.rerank(question, pool)

    return Gate(module, load_llm(repo / "llm.yaml"), search)
