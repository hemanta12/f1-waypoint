"""The refusal gate: one question in, one decision out.

Order: screen, subject-keyword prefilter, retrieval, then the passage-aware classifier. The
score threshold and the classifier's label are combined by `resolve`.
"""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from core.bm25 import BM25Index
from core.classifier_prompt import Classification, classifier_input, system_prompt
from core.config import (
    LlmConfig,
    ModuleConfig,
    NotInForce,
    Reason,
    load_llm,
    load_module,
    load_not_in_force,
)
from core.index import Index, clean_results, rrf
from core.llm import LlmError, Usage, complete_json
from core.prefilter import prefilter
from core.refusal import resolve
from core.registry import load_verified_on, normalize_text
from core.rerank import Reranker
from core.screen import Problem, screen

FETCH = 10
KEEP = 5
RERANKER = "Xenova/ms-marco-MiniLM-L-12-v2"
# A not-in-force passage scoring this high means the question is about its rule. Tuned on the
# eval: 9 of 97 questions, every answerable one on topic; rank-based (top 5) fired on 18 and
# missed grace-period wording.
NOTICE_MIN_SCORE = 3.0

# Ranked (chunk, reranker score) pairs, best first.
Search = Callable[[str], list[tuple[dict, float]]]


class Passage(BaseModel):
    doc_id: str
    section: str | None
    source_url: str
    chunk_ids: list[str]
    text: str
    verified_on: date | None = None


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
    notices: list[str] = []  # a not-in-force passage scored NOTICE_MIN_SCORE or more


class ClassifierFailed(LlmError):
    """The classifier call failed after retrieval. `passages` is empty when retrieval was
    weak, since those passages do not answer the question."""

    def __init__(self, error: LlmError, passages: list[Passage]) -> None:
        super().__init__(error.kind, str(error))
        self.passages = passages


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
        """Raises ClassifierFailed if the classifier call fails. No retry."""
        checked = screen(question)
        if not checked.ok:
            return GateResult(status="rejected", problem=checked.problem, message=checked.message)
        if hit := prefilter(question):
            return self._refused(hit[0], hit[1], [], None, Usage())

        ranked = self.search(question)
        notices = list(
            dict.fromkeys(
                c["notice"] for c, s in ranked if c.get("notice") and s >= NOTICE_MIN_SCORE
            )
        )
        ranked = [(c, s) for c, s in ranked if not c.get("not_in_force")]
        passages = clean_results(ranked, KEEP)
        top_score = ranked[0][1] if ranked else None
        message = classifier_input(question, passages[0]["text"] if passages else None)
        try:
            label, usage = complete_json(
                self.llm.classifier, self.prompt, message, Classification, self.client
            )
        except LlmError as e:
            strong = top_score is not None and top_score >= self.module.retrieval_threshold
            raise ClassifierFailed(e, _passages(passages) if strong else []) from e
        decision = resolve(self.module, top_score, label)
        if decision.reason is None:
            return GateResult(
                status="answerable",
                passages=_passages(passages),
                top_score=top_score,
                classifier_usage=usage,
                notices=notices,
            )
        return self._refused(
            decision.reason, decision.referral, passages, top_score, usage, notices
        )

    def _refused(
        self,
        reason: Reason,
        suggested: str | None,
        passages: list[dict],
        top_score: float | None,
        usage: Usage,
        notices: list[str] | None = None,
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
            notices=notices or [],
        )


def _passages(raw: list[dict]) -> list[Passage]:
    return [Passage.model_validate({k: p[k] for k in Passage.model_fields if k in p}) for p in raw]


def not_in_force_ids(chunks: list[dict], entries: list[NotInForce]) -> set[str]:
    """Chunk IDs holding any not-in-force quote. Raises if a quote matches no chunk, which
    means the corpus or chunking changed and the list must be checked again."""
    found: set[str] = set()
    for entry in entries:
        texts = {
            c["chunk_id"]: normalize_text(c["text"]) for c in chunks if c["doc_id"] == entry.doc_id
        }
        for quote in entry.quotes:
            hits = {cid for cid, text in texts.items() if normalize_text(quote) in text}
            if not hits:
                raise ValueError(
                    f"not-in-force quote matches no {entry.doc_id} chunk: {quote[:60]}"
                )
            found |= hits
    return found


def load_gate(repo: Path, module_dir: Path) -> Gate:
    from fastembed import TextEmbedding

    module = load_module(module_dir / "module.yaml")
    index = Index(module_dir / "index.sqlite")
    dates = load_verified_on(module_dir / "corpus")
    for chunk in index.chunks:
        chunk["verified_on"] = dates.get(chunk["doc_id"])
    embedder = TextEmbedding(index.meta["embedding_model"])
    bm25 = BM25Index(index.chunks)
    reranker = Reranker(RERANKER)
    by_id = {c["chunk_id"]: c for c in index.chunks}
    for entry in load_not_in_force(module_dir / "not_in_force.yaml"):
        for chunk_id in not_in_force_ids(index.chunks, [entry]):
            by_id[chunk_id]["not_in_force"] = True
            if entry.notice:
                by_id[chunk_id]["notice"] = entry.notice

    # Not-in-force chunks are ranked too, so Gate.run can tell when one would have been shown,
    # then drops them before the classifier or generator sees anything.
    def search(question: str) -> list[tuple[dict, float]]:
        vector = next(iter(embedder.query_embed([question])))
        pool = rrf(index.search(vector, FETCH), bm25.search(question, FETCH))
        return reranker.rerank(question, pool)

    return Gate(module, load_llm(repo / "llm.yaml"), search)
