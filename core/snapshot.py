import json
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path

import pymupdf

from core.registry import Document, Source, apply_refetch, hash_text

_SKIP_TAGS = {"script", "style", "noscript", "template"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "table"}


class _TextExtractor(HTMLParser):
    """Page text, with each table row written as one self-contained line.

    A row becomes "Header A: cell A | Header B: cell B", so a chunk that holds
    only that row still says what each number means.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0
        self._table_depth = 0
        self._headers: list[str] = []
        self._row: list[tuple[str, int]] = []
        self._row_is_header = True
        self._cell: list[str] | None = None
        self._cell_span = 1
        self._carry: dict[int, list] = {}

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._headers, self._carry = [], {}
                self.parts.append("\n")
        elif self._table_depth == 1 and tag == "tr":
            self._row, self._row_is_header = [], True
        elif self._table_depth == 1 and tag in {"td", "th"}:
            self._cell = []
            span = dict(attrs).get("rowspan") or "1"
            self._cell_span = int(span) if span.isdigit() else 1
            if tag == "td":
                self._row_is_header = False
        elif tag in _BLOCK_TAGS:
            if self._cell is not None:
                self._cell.append(" ")
            else:
                self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1
            if not self._table_depth:
                self.parts.append("\n")
        elif self._table_depth == 1 and tag in {"td", "th"} and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append((text, self._cell_span))
            self._cell = None
        elif self._table_depth == 1 and tag == "tr":
            self._end_row()

    def _take_carry(self, col: int, out: list[str]) -> bool:
        if col not in self._carry:
            return False
        text, remaining = self._carry[col]
        out.append(text)
        if remaining <= 1:
            del self._carry[col]
        else:
            self._carry[col][1] = remaining - 1
        return True

    def _end_row(self):
        # A cell with rowspan=N also belongs to the next N-1 rows; repeat it there
        # so those rows keep their first column instead of shifting left.
        row: list[str] = []
        col = 0
        for text, span in self._row:
            while self._take_carry(col, row):
                col += 1
            row.append(text)
            if span > 1:
                self._carry[col] = [text, span - 1]
            col += 1
        while self._take_carry(col, row):
            col += 1
        if not any(row):
            return
        if self._row_is_header and not self._headers:
            self._headers = [h.rstrip(":").strip() for h in row]
            self.parts.append("\n" + " | ".join(self._headers) + "\n")
            return
        if len(self._headers) == len(row):
            cells = [f"{h}: {c}" for h, c in zip(self._headers, row, strict=True)]
        else:
            cells = row
        self.parts.append(" | ".join(cells) + "\n")

    def handle_data(self, data):
        if self._skip:
            return
        (self._cell if self._cell is not None else self.parts).append(data)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            return "\n\n".join(page.get_text() for page in pdf)
    if suffix == ".xml":
        return " ".join(ET.fromstring(path.read_bytes()).itertext())
    parser = _TextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()


def find_raw(raw_dir: Path, doc_id: str) -> Path | None:
    matches = [
        p for p in raw_dir.glob(f"{doc_id}.*") if p.suffix.lower() in {".html", ".pdf", ".xml"}
    ]
    return matches[0] if matches else None


_CANONICAL = re.compile(
    r"<link[^>]+rel=[\"']canonical[\"'][^>]+href=[\"']([^\"']+)|"
    r"<meta[^>]+property=[\"']og:url[\"'][^>]+content=[\"']([^\"']+)",
    re.IGNORECASE,
)


def _norm(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url.strip()).rstrip("/").lower()


def sort_inbox(sources: list[Source], raw_dir: Path) -> list[str]:
    """Move browser-saved files from raw/inbox to raw/<doc_id>.<ext>.

    HTML is matched by the page's canonical URL, PDFs by the URL's file name.
    Returns the names of files that matched nothing.
    """
    inbox = raw_dir / "inbox"
    if not inbox.is_dir():
        return []
    by_url = {_norm(str(s.url)): s.doc_id for s in sources}
    by_pdf_name = {
        str(s.url).rsplit("/", 1)[-1].lower(): s.doc_id
        for s in sources
        if str(s.url).lower().endswith(".pdf")
    }
    unmatched = []
    for f in sorted(inbox.iterdir()):
        suffix = f.suffix.lower()
        doc_id = None
        if suffix == ".pdf":
            doc_id = by_pdf_name.get(f.name.lower())
        elif suffix in {".html", ".htm"}:
            m = _CANONICAL.search(f.read_text(encoding="utf-8", errors="replace"))
            if m:
                doc_id = by_url.get(_norm(m.group(1) or m.group(2)))
        if doc_id:
            f.rename(raw_dir / f"{doc_id}{'.pdf' if suffix == '.pdf' else '.html'}")
        elif f.is_file() and not f.name.startswith("."):
            unmatched.append(f.name)
    return unmatched


def build_registry(
    sources: list[Source],
    module_id: str,
    corpus_dir: Path,
    today: date,
) -> tuple[list[Document], list[str]]:
    """Return (registry entries, doc_ids with no usable raw file).

    A new document is dated by the raw file's modified time, i.e. when it was
    saved. An existing one goes through apply_refetch so unchanged text keeps
    its retrieved_at.
    """
    raw_dir, text_dir = corpus_dir / "raw", corpus_dir / "text"
    registry_path = corpus_dir / "registry.json"
    text_dir.mkdir(parents=True, exist_ok=True)
    known = {}
    if registry_path.exists():
        known = {
            d["doc_id"]: Document.model_validate(d) for d in json.loads(registry_path.read_text())
        }

    docs: list[Document] = []
    missing: list[str] = []
    for s in sources:
        raw = find_raw(raw_dir, s.doc_id)
        text = extract_text(raw).strip() if raw else ""
        if not text:
            missing.append(s.doc_id)
            continue
        (text_dir / f"{s.doc_id}.txt").write_text(text, encoding="utf-8")
        if s.doc_id in known:
            docs.append(apply_refetch(known[s.doc_id], text, today))
        else:
            saved = datetime.fromtimestamp(raw.stat().st_mtime).date()
            docs.append(
                Document(
                    doc_id=s.doc_id,
                    module_id=module_id,
                    title=s.title,
                    source_url=s.url,
                    source_tier=s.tier,
                    retrieved_at=saved,
                    verified_on=saved,
                    content_hash=hash_text(text),
                )
            )
    registry_path.write_text(
        json.dumps([d.model_dump(mode="json") for d in docs], indent=2) + "\n", encoding="utf-8"
    )
    return docs, missing


if __name__ == "__main__":
    from core.config import load_module
    from core.registry import load_sources

    root = Path(__file__).resolve().parent.parent / "modules" / "opt_stem"
    module = load_module(root / "module.yaml")
    sources = load_sources(root / "sources.yaml")
    for name in sort_inbox(sources, root / "corpus" / "raw"):
        print(f"  unmatched in inbox (rename to <doc_id>.html or .pdf): {name}")
    docs, missing = build_registry(sources, module.id, root / "corpus", date.today())
    print(f"registered {len(docs)}; missing {len(missing)}")
    for s in sources:
        if s.doc_id in missing:
            print(f"  missing: {s.doc_id}  {s.url}")
