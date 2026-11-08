import re

from pydantic import BaseModel, ConfigDict, Field

from core.snapshot import Section


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    doc_id: str
    text: str = Field(min_length=1)
    section: str | None = None


def embed_input(chunk: Chunk) -> str:
    # Later pieces of a split section lose their heading; give it back for embedding.
    if chunk.section and not chunk.text.startswith(chunk.section):
        return f"{chunk.section}\n{chunk.text}"
    return chunk.text


def merge_small_sections(sections: list[Section], min_words: int = 30) -> list[Section]:
    """Fold each section under `min_words` into the next one (the last into the one
    before), keeping the bigger section's heading. Nothing is dropped: the joined
    text before and after is the same."""
    merged: list[Section] = []
    pending = ""
    for heading, text in sections:
        text = f"{pending}\n\n{text}" if pending else text
        if len(text.split()) < min_words:
            pending = text
            continue
        merged.append((heading, text))
        pending = ""
    if pending:
        if merged:
            heading, text = merged.pop()
            merged.append((heading, f"{text}\n\n{pending}"))
        else:
            merged.append((sections[-1][0], pending))
    return merged


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _units(text: str, max_words: int) -> list[str]:
    """Lines (block boundaries), then sentences for a too-long line, then word
    windows only for a single sentence longer than max_words."""
    out = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if len(line.split()) <= max_words:
            out.append(line)
            continue
        for sentence in _SENTENCE_END.split(line):
            words = sentence.split()
            out += [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
    return out


def split_section(text: str, max_words: int) -> list[str]:
    pieces: list[list[str]] = []
    count = 0
    for unit in _units(text, max_words):
        n = len(unit.split())
        if not pieces or count + n > max_words:
            pieces.append([])
            count = 0
        pieces[-1].append(unit)
        count += n
    return ["\n".join(p) for p in pieces]


def _slug(heading: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (heading or "").lower()).strip("-")[:60] or "intro"


def structure_chunks(doc_id: str, sections: list[Section], max_words: int = 150) -> list[Chunk]:
    """One section per chunk, split at block and sentence ends when it is too long.
    chunk_id is doc_id plus section slug, so it survives edits to other sections."""
    chunks = []
    seen: dict[str, int] = {}
    for heading, text in merge_small_sections(sections):
        slug = _slug(heading)
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"
        pieces = split_section(text, max_words)
        for i, piece in enumerate(pieces, 1):
            suffix = f"/{i}" if len(pieces) > 1 else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{slug}{suffix}", doc_id=doc_id, text=piece, section=heading
                )
            )
    return chunks


def fixed_size_chunks(doc_id: str, text: str, size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Baseline: windows of `size` words, each starting `size - overlap` words after
    the last. Ignores headings and sentences on purpose; it is the thing to beat."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + size]
        # chunk_id is positional, so any edit above a chunk renumbers it.
        chunks.append(Chunk(chunk_id=f"{doc_id}#w{start}", doc_id=doc_id, text=" ".join(window)))
        if start + size >= len(words):
            break
    return chunks
