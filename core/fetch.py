import re
import sys
import time
from pathlib import Path

import httpx

from core.registry import Source, load_sources

USER_AGENT = "F1-Waypoint/0.1 (research project; builds a cited Q&A corpus from official sources)"
DELAY_SECONDS = 1.0

_ECFR_SECTION = re.compile(r"ecfr\.gov/current/title-(\d+)/.*/part-(\d+)/.*/section-(\d+\.\d+)$")


def request_url(url: str, client: httpx.Client) -> str:
    # eCFR's section pages are an app shell; the documented API returns the text.
    # The API rejects dates later than the title's up_to_date_as_of, so look it up.
    m = _ECFR_SECTION.search(url)
    if not m:
        return url
    title, part, section = m.groups()
    titles = client.get("https://www.ecfr.gov/api/versioner/v1/titles.json").raise_for_status()
    as_of = next(
        t["up_to_date_as_of"] for t in titles.json()["titles"] if t["number"] == int(title)
    )
    return (
        f"https://www.ecfr.gov/api/versioner/v1/full/{as_of}/title-{title}.xml"
        f"?part={part}&section={section}"
    )


def extension(content_type: str) -> str:
    if "pdf" in content_type:
        return ".pdf"
    if "xml" in content_type:
        return ".xml"
    return ".html"


def fetch_all(sources: list[Source], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf,application/xml"}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=30) as client:
        for i, s in enumerate(sources):
            if i:
                time.sleep(DELAY_SECONDS)
            try:
                r = client.get(request_url(str(s.url), client))
                r.raise_for_status()
                if not r.content:
                    raise ValueError("empty body")
            except (httpx.HTTPError, ValueError) as e:
                results[s.doc_id] = f"FAIL {str(e).splitlines()[0]}"
                continue
            ext = extension(r.headers.get("content-type", ""))
            (out_dir / f"{s.doc_id}{ext}").write_bytes(r.content)
            results[s.doc_id] = f"ok {len(r.content)} bytes"
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "modules" / "opt_stem"
    outcome = fetch_all(load_sources(root / "sources.yaml"), root / "corpus" / "raw")
    for doc_id, status in outcome.items():
        print(f"{doc_id}: {status}")
    sys.exit(any(s.startswith("FAIL") for s in outcome.values()))
