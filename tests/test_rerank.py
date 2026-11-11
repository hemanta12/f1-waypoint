from core.rerank import Reranker


class _FakeModel:
    def rerank(self, query, documents):
        # Scores each document by how many query words it contains.
        return [sum(w in d.lower() for w in query.lower().split()) for d in documents]


def _reranker():
    r = Reranker.__new__(Reranker)
    r.model_name, r.model = "fake", _FakeModel()
    return r


def _chunk(cid, text):
    return {"chunk_id": cid, "doc_id": cid, "section": None, "text": text}


def test_rerank_reorders_and_replaces_scores():
    results = [(_chunk("a", "fee schedule"), 0.9), (_chunk("b", "unemployment days limit"), 0.1)]
    out = _reranker().rerank("unemployment days", results)
    assert [(c["chunk_id"], s) for c, s in out] == [("b", 2.0), ("a", 0.0)]


def test_rerank_empty_pool():
    assert _reranker().rerank("anything", []) == []
