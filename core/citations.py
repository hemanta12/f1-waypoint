"""Check that a generation is backed by the chunks it cites, and number its citations."""

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel

from core.generate import Claim, Generation

# The hand-written check: (statement, supporting texts) -> problems. Empty means every
# number, duration and form ID in the statement appears in a supporting text.
CoreCheck = Callable[[str, list[str]], list[str]]


class CitedClaim(BaseModel):
    text: str
    chunk_ids: list[str]
    markers: list[int]


class Citation(BaseModel):
    n: int
    chunk_id: str
    doc_id: str
    section: str | None
    source_url: str
    verified_on: date | None = None


def load_chunks(db_path: Path) -> dict[str, dict]:
    """chunk_id -> its row, without the vectors."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            "SELECT chunk_id, doc_id, section, source_url, text FROM chunks"
        ).fetchall()
    finally:
        db.close()
    return {r["chunk_id"]: dict(r) for r in rows}


MIN_QUOTE_WORDS = 8
_FOOTNOTE = re.compile(r"\[\d+\]")
_MARKS = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2010": "-", "\u2011": "-",
     "\u2013": "-", "\u2014": "-"}
)  # fmt: skip


def _quote_form(text: str) -> str:
    """What must match between a quote and its chunk: words and order. Ignored: whitespace,
    case, curly versus straight quotes and dashes, footnote markers, and end punctuation."""
    text = _FOOTNOTE.sub("", text).translate(_MARKS)
    return re.sub(r"\s+", " ", text).strip(" .,;:").casefold()


def snap_quote(quote: str, text: str) -> str | None:
    """The chunk's own words for a quote that differs from them in a word or two, or None.

    The returned span is cut from `text`, so it is always verbatim source text. The quote is
    lined up on the longest run it shares with the text; at most one word in ten (at least
    one) may differ, and numbers may not."""
    raw = text.split()
    have = [_quote_form(t) for t in raw]
    if _ELLIPSIS.search(quote):
        return _fill_elision(quote, raw, have)
    want = [w for w in (_quote_form(t) for t in quote.split()) if w]
    if len(want) < MIN_QUOTE_WORDS:
        return None
    longest = SequenceMatcher(None, have, want, autojunk=False).find_longest_match(
        0, len(have), 0, len(want)
    )
    if longest.size < 2:
        return None
    start = max(0, longest.a - longest.b)
    end = min(len(raw), start + len(want))
    same = sum(b.size for b in SequenceMatcher(None, have[start:end], want).get_matching_blocks())
    if max(len(want), end - start) - same > max(1, len(want) // 10):
        return None
    span = " ".join(raw[start:end])
    # A changed number is a misread, not a copying slip.
    if re.findall(r"\d+", _FOOTNOTE.sub("", quote)) != re.findall(r"\d+", _FOOTNOTE.sub("", span)):
        return None
    return span


_ELLIPSIS = re.compile(r"…|\.\.\.")
MAX_ELIDED_WORDS = 30


def _fill_elision(quote: str, raw: list[str], have: list[str]) -> str | None:
    """A quote with "..." gets the whole source span from its first piece to its last, so a
    skipped "not" can never change what the student reads."""
    pieces = [[w for w in (_quote_form(t) for t in p.split()) if w] for p in _ELLIPSIS.split(quote)]
    pieces = [p for p in pieces if p]
    if sum(len(p) for p in pieces) < MIN_QUOTE_WORDS:
        return None
    start = pos = None
    for piece in pieces:
        at = next(
            (
                i
                for i in range(pos or 0, len(have) - len(piece) + 1)
                if have[i : i + len(piece)] == piece
            ),
            None,
        )
        if at is None:
            return None
        start = at if start is None else start
        pos = at + len(piece)
    skipped = pos - start - sum(len(p) for p in pieces)
    return " ".join(raw[start:pos]) if skipped <= MAX_ELIDED_WORDS else None


_SENTENCE = re.compile(r"(?<=[.!?:])\s+")
_ACRONYM = re.compile(r"^[A-Z]{2,}s?$")


def key_terms(text: str) -> list[str]:
    """Acronyms (EAD, DSO) and capitalized names of two or more words with at least one
    non-acronym (Actual OPT Start Date, Form I-983). A sentence's first word does not start a
    name, so "The SEVP Portal" gives "SEVP Portal"; punctuation after a word ends the name."""
    terms: list[str] = []
    run: list[str] = []

    def flush() -> None:
        # A run of acronyms only ("OPT EAD") is not a name; each acronym is checked alone.
        if len(run) >= 2 and not all(_ACRONYM.match(w) for w in run):
            terms.append(" ".join(run))
        run.clear()

    for sentence in _SENTENCE.split(text):
        for i, raw in enumerate(sentence.split()):
            word = raw.strip("()\"'.,;:!?")
            if _ACRONYM.match(word):
                terms.append(word.removesuffix("s"))
            if i > 0 and word[:1].isupper():
                run.append(word)
            else:
                flush()
            if raw[-1:] in ",;:.!?)":
                flush()
        flush()
    return list(dict.fromkeys(terms))


def missing_terms(
    statement: str, quotes: list[str], aliases: dict[str, str], generic: list[str]
) -> list[str]:
    """Key terms of the statement that no quote contains, by name or by the acronym's
    spelled-out form. Generic terms (the module's subject) are not required."""
    joined = _quote_form(" ".join(quotes))
    words = set(re.findall(r"[a-z0-9-]+", joined))
    missing = []
    for term in key_terms(statement):
        if term in generic:
            continue
        if " " in term:
            ok = _quote_form(term) in joined
        else:
            spelled = aliases.get(term)
            ok = term.casefold() in words or bool(spelled and _quote_form(spelled) in joined)
        if not ok:
            missing.append(term)
    return missing


@dataclass
class CheckResult:
    """`quotes` problems mean a claim is not verbatim source text: show passages only.
    `summary` problems mean only the direct answer or rule failed: show the quotes alone."""

    quotes: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    sources: list[list[str]] = field(default_factory=list)  # per claim, chunks holding it
    texts: list[str] = field(default_factory=list)  # per claim, the verified source text
    snapped: list[int] = field(default_factory=list)  # claims replaced by their source text

    @property
    def problems(self) -> list[str]:
        return self.quotes + self.summary


def check_generation(
    gen: Generation,
    shown_ids: set[str],
    chunks: dict[str, dict],
    core_check: CoreCheck,
    aliases: dict[str, str] | None = None,
    generic: list[str] | None = None,
) -> CheckResult:
    """Developer-facing problems; never shown to users."""
    result = CheckResult()
    if not gen.claims:
        result.quotes.append("answered without claims")
    for i, claim in enumerate(gen.claims, 1):
        unknown = [c for c in claim.chunk_ids if c not in shown_ids or c not in chunks]
        if not claim.chunk_ids or unknown:
            result.quotes.append(f"claim {i} cites no shown chunk: {unknown or 'none'}")
            result.sources.append([])
            result.texts.append(claim.text)
            continue
        text = claim.text
        quote = _quote_form(text)
        if len(quote.split()) < MIN_QUOTE_WORDS:
            result.quotes.append(f"claim {i} is under {MIN_QUOTE_WORDS} words")
        holders = [c for c in claim.chunk_ids if quote in _quote_form(chunks[c]["text"])]
        if not holders:
            snaps = [s for c in claim.chunk_ids if (s := snap_quote(text, chunks[c]["text"]))]
            if snaps:
                text = snaps[0]
                quote = _quote_form(text)
                holders = [c for c in claim.chunk_ids if quote in _quote_form(chunks[c]["text"])]
                result.snapped.append(i)
        if not holders:
            result.quotes.append(f"claim {i} is not verbatim text of the chunks it cites")
        result.sources.append(holders)
        result.texts.append(text)
    if result.quotes:
        return result
    quotes = result.texts
    if not gen.direct_answer:
        result.summary.append("answered without a direct_answer")
    if gen.direct_answer:
        result.summary += [f"direct_answer: {p}" for p in core_check(gen.direct_answer, quotes)]
        result.summary += [
            f"direct_answer: term {t!r} not in quotes"
            for t in missing_terms(gen.direct_answer, quotes, aliases or {}, generic or [])
        ]
    if gen.rule and _quote_form(gen.rule) not in _quote_form(" ".join(quotes)):
        result.summary.append("rule is not copied from a quote")
    return result


def cite(claims: list[Claim], chunks: dict[str, dict]) -> tuple[list[CitedClaim], list[Citation]]:
    """Number chunks [1], [2], ... in order of first use. Pass claims whose chunk_ids are the
    chunks the check found holding each quote."""
    numbers: dict[str, int] = {}
    cited = []
    for claim in claims:
        ids = list(dict.fromkeys(claim.chunk_ids))
        markers = [numbers.setdefault(c, len(numbers) + 1) for c in ids]
        cited.append(CitedClaim(text=claim.text, chunk_ids=ids, markers=markers))
    citations = [
        Citation(
            n=n,
            chunk_id=cid,
            doc_id=chunks[cid]["doc_id"],
            section=chunks[cid]["section"],
            source_url=chunks[cid]["source_url"],
            verified_on=chunks[cid].get("verified_on"),
        )
        for cid, n in numbers.items()
    ]
    return cited, citations
