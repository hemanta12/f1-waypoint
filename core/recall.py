from core.evalset import GoldPassage
from core.registry import normalize_text

# A retrieved chunk or passage as (doc_id, text).
Retrieved = tuple[str, str]


def is_hit(gold: GoldPassage, chunk: Retrieved) -> bool:
    doc_id, text = chunk
    return any(
        doc_id == p.doc_id and normalize_text(p.quote) in normalize_text(text)
        for p in (gold, *gold.alternates)
    )


def recall_at_k(gold: list[GoldPassage], retrieved: list[Retrieved], k: int = 5) -> float:
    """Fraction of gold passages hit by at least one of the first k retrieved chunks."""
    if not gold:
        raise ValueError("a case without gold passages cannot be scored")
    top = retrieved[:k]
    found = sum(1 for g in gold if any(is_hit(g, chunk) for chunk in top))
    return found / len(gold)
