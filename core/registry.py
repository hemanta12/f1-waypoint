import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    module_id: str
    title: str = Field(min_length=1)
    source_url: HttpUrl
    source_tier: Literal["official"]
    retrieved_at: date
    verified_on: date
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["active", "removed"] = "active"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    url: HttpUrl
    tier: Literal["official"]
    authoritative_for: str = Field(min_length=1)


def load_sources(path: Path) -> list[Source]:
    sources = [Source.model_validate(s) for s in yaml.safe_load(Path(path).read_text())]
    ids = [s.doc_id for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate doc_id in sources")
    return sources


def hash_text(text: str) -> str:
    # Collapse whitespace so reflowed markup with identical words is not a "change".
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def apply_refetch(doc: Document, new_text: str, today: date) -> Document:
    """Update a registry entry after a successful re-fetch.

    doc_id, source_url, source_tier, title are never touched. verified_on always
    moves; retrieved_at and content_hash move only when the extracted text changed.
    """
    new_hash = hash_text(new_text)
    update = {"verified_on": today, "status": "active"}
    if new_hash != doc.content_hash:
        update |= {"content_hash": new_hash, "retrieved_at": today}
    return doc.model_copy(update=update)
