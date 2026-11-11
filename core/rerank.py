"""Cross-encoder reranking: reads the question and each candidate chunk together."""

from core.index import chunk_text


class Reranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self.model = TextCrossEncoder(model_name)

    def rerank(self, query: str, results: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
        """Same chunks, reordered by the reranker; each score is replaced by the
        reranker's raw score (not 0 to 1)."""
        if not results:
            return []
        scores = list(self.model.rerank(query, [chunk_text(c) for c, _ in results]))
        ranked = sorted(zip(results, scores, strict=True), key=lambda x: -x[1])
        return [(chunk, float(score)) for (chunk, _), score in ranked]
