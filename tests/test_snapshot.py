from datetime import date

import pymupdf

from core.registry import Source
from core.snapshot import build_registry, extract_text

D1, D2 = date(2026, 11, 5), date(2026, 11, 12)


def _source(doc_id: str) -> Source:
    return Source(
        doc_id=doc_id,
        title="T",
        url="https://example.gov/x",
        tier="official",
        authoritative_for="x",
    )


def _corpus(tmp_path):
    (tmp_path / "raw").mkdir()
    return tmp_path


def test_html_extraction_drops_scripts_and_keeps_text(tmp_path):
    f = tmp_path / "a.html"
    f.write_text("<html><script>var x=1</script><p>Report in 10 days</p></html>")
    assert extract_text(f) == "Report in 10 days"


def test_pdf_and_xml_extraction(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Form I-983")
    pdf.save(tmp_path / "a.pdf")
    xml = tmp_path / "b.xml"
    xml.write_text("<DIV><P>8 CFR</P><P>214.2</P></DIV>")
    assert "Form I-983" in extract_text(tmp_path / "a.pdf")
    assert extract_text(xml) == "8 CFR 214.2"


def test_missing_and_empty_files_are_reported_not_registered(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "raw" / "empty.html").write_text("<html><script>x</script></html>")
    docs, missing = build_registry([_source("gone"), _source("empty")], "m", corpus, D1)
    assert docs == []
    assert missing == ["gone", "empty"]


def test_rebuild_keeps_retrieved_at_when_text_is_unchanged(tmp_path):
    corpus = _corpus(tmp_path)
    raw = corpus / "raw" / "doc.html"
    raw.write_text("<p>Report in 10 days</p>")
    first, _ = build_registry([_source("doc")], "m", corpus, D1)
    second, _ = build_registry([_source("doc")], "m", corpus, D2)
    assert second[0].content_hash == first[0].content_hash
    assert second[0].verified_on == D2
    assert second[0].retrieved_at == first[0].retrieved_at

    raw.write_text("<p>Report in 15 days</p>")
    third, _ = build_registry([_source("doc")], "m", corpus, D2)
    assert third[0].content_hash != first[0].content_hash
    assert third[0].retrieved_at == D2


def test_sort_inbox_matches_canonical_html_and_pdf_name(tmp_path):
    from core.snapshot import sort_inbox

    raw = tmp_path / "raw"
    (raw / "inbox").mkdir(parents=True)
    page = _source("page").model_copy(update={"url": "https://www.uscis.gov/stem-opt"})
    form = _source("form").model_copy(update={"url": "https://www.ice.gov/doclib/i983.pdf"})
    (raw / "inbox" / "STEM OPT | USCIS.html").write_text(
        '<link rel="canonical" href="https://www.uscis.gov/stem-opt/">'
    )
    (raw / "inbox" / "i983.pdf").write_bytes(b"%PDF")
    (raw / "inbox" / "mystery.html").write_text("<p>no canonical</p>")
    assert sort_inbox([page, form], raw) == ["mystery.html"]
    assert (raw / "page.html").exists()
    assert (raw / "form.pdf").exists()


def test_table_rows_keep_their_headers(tmp_path):
    f = tmp_path / "t.html"
    f.write_text(
        "<p>Unemployment</p><table><thead><tr><th>If you received</th><th>Total</th></tr></thead>"
        "<tbody><tr><td>Initial OPT</td><td>90 days</td></tr>"
        "<tr><td>24-month extension</td><td>150 days</td></tr></tbody></table><p>After</p>"
    )
    lines = extract_text(f).splitlines()
    assert "If you received: Initial OPT | Total: 90 days" in lines
    assert "If you received: 24-month extension | Total: 150 days" in lines
    assert lines[-1] == "After"


def test_table_without_header_row_still_separates_cells(tmp_path):
    f = tmp_path / "t.html"
    f.write_text("<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>")
    assert "c | d" in extract_text(f).splitlines()


def test_rowspan_cell_is_repeated_on_the_rows_it_covers(tmp_path):
    f = tmp_path / "t.html"
    f.write_text(
        "<table><tr><th>Degree</th><th>For</th></tr>"
        '<tr><td rowspan="2">STEM</td><td>Pre-completion</td></tr>'
        "<tr><td>Post-completion</td></tr>"
        "<tr><td>Non-STEM</td><td>Pre-completion</td></tr></table>"
    )
    lines = extract_text(f).splitlines()
    assert "Degree: STEM | For: Post-completion" in lines
    assert "Degree: Non-STEM | For: Pre-completion" in lines


def test_trailing_colon_in_header_is_not_doubled(tmp_path):
    f = tmp_path / "t.html"
    f.write_text("<table><tr><th>When:</th></tr><tr><td>6 months</td></tr></table>")
    assert "When: 6 months" in extract_text(f).splitlines()
