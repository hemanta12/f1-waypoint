"""Keyword search with BM25 over the same chunks as the vector index."""

import re

import bm25s
import numpy as np

from core.index import chunk_text, clean_results

# Tokenizer rules (decided 2026-11-09):
# 1. Form numbers: I-983, i-983, I983 and I 983 all become one token, "i983".
#    A plain split gives "i" + "983", which never matches a query written "i983",
#    and turns I-20 into "20", which matches "20 days".
# 2. CFR citations: 214.2(f)(10)(ii) is kept whole and also emits its parents
#    214.2, 214.2(f), 214.2(f)(10), so a broad query matches a chunk citing a sub-part.
# 3. Common English words are dropped (NLTK's list). Students ask casually ("my",
#    "do", "how") but the corpus is formal, so BM25 rated those words rare and
#    important and returned first-person FAQ chunks. Measured: Recall@5 0.509 -> 0.618.
# 4. STEM OPT EAD category codes: (c)(3)(C) becomes one token, "c3c".
# Everything else: lowercase, split on \w+.

# One pass, left to right, so tokens keep text order. At each position the citation
# pattern is tried first, then the form pattern, then a plain word; reordering the
# alternatives lets \w+ split citations and forms back into generic pieces.
# A one-digit form (I-9) needs the hyphen, so "can i 2 ..." is not read as a form.
# A trailing letter is kept (I-129S is its own form), so the plural "I-94s" becomes
# "i94s" and misses "i94"; it appears once in the corpus.
# Some official pages write form numbers with a Unicode hyphen (U+2010) instead of "-".
_DASH = "-\u2010\u2011\u2013\u2014"
_TOKEN = re.compile(
    rf"(?P<cite>\d+[a-z]?\.\d+(?:\([a-z0-9]+\))*)"
    rf"|(?P<form>\bi(?:[{_DASH}]\d{{1,3}}|[{_DASH} ]?\d{{2,3}})[a-z]?\b)"
    r"|(?P<ead>\(c\)\(3\)\([abc]\))"
    r"|(?P<word>\w+)"
)
_PART = re.compile(r"\([a-z0-9]+\)")
_STOPWORDS = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves he him his
    himself she her hers herself it its itself they them their theirs themselves what
    which who whom this that these those am is are was were be been being have has had
    having do does did doing a an the and but if or because as until while of at by for
    with about against between into through during before after above below to from up
    down in out on off over under again further then once here there when where why how
    all any both each few more most other some such no nor not only own same so than too
    very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn
    hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn
    """.split()
)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _TOKEN.finditer(text.lower()):
        if m.lastgroup == "cite":
            cite = m.group()
            parts = _PART.findall(cite)
            base = cite[: len(cite) - sum(map(len, parts))]
            tokens += [base + "".join(parts[:i]) for i in range(len(parts) + 1)]
        elif m.lastgroup == "form":
            tokens.append(re.sub(rf"[{_DASH} ]", "", m.group()))
        elif m.lastgroup == "ead":
            tokens.append(re.sub(r"[()]", "", m.group()))
        elif m.group() not in _STOPWORDS:
            tokens.append(m.group())
    return tokens


class BM25Index:
    """Built in memory from the vector index's chunks on every load, so the two
    searches always cover the same chunks and there is no second file to sync."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.bm25 = bm25s.BM25()
        self.bm25.index([tokenize(chunk_text(c)) for c in chunks], show_progress=False)

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        # bm25s fails on a query with no known words, and zero-score chunks share
        # no word with the query, so neither may be returned as a result.
        ids = self.bm25.get_tokens_ids(tokenize(query))
        if not ids:
            return []
        scores = self.bm25.get_scores_from_ids(ids)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    def retrieve(self, query: str, k: int = 5, fetch: int = 10) -> list[dict]:
        return clean_results(self.search(query, fetch), k)
