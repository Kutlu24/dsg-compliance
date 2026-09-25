"""Ingestion from fedlex.admin.ch (Swiss federal legislation).

Fedlex's public-facing site (www.fedlex.admin.ch) is a JS-rendered Angular
app - fetching it directly returns an empty shell, not the law text. There
is a real data/metadata layer at fedlex.data.admin.ch (RDF/jolux ontology,
SPARQL endpoint at https://fedlex.data.admin.ch/sparqlendpoint), but it does
not expose full article text through a simple, documented property either
(confirmed 2026-09-07: even the community fedlex-mcp project only surfaces
metadata and links back to the JS site, not full text).

What DOES work, verified 2026-09-07 against the real DSG text: Fedlex
publishes a PDF/A of every consolidated law version at a predictable
filestore URL, built from the law's (year, number) identifier, its
consolidation date (from jolux:dateEntryInForce via the RDF endpoint), and
language:

    https://www.fedlex.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/
        {year}/{num}/{date:%Y%m%d}/{lang}/pdf-a/
        fedlex-data-admin-ch-eli-cc-{year}-{num}-{date:%Y%m%d}-{lang}-pdf-a.pdf

This module downloads that PDF, extracts text with pypdf, and splits it into
per-article chunks (each starting at a line like "Art. 12 Titel").
"""
from __future__ import annotations

import re
from datetime import date
from io import BytesIO

import httpx
from pydantic import BaseModel

RDF_BASE = "https://fedlex.data.admin.ch/eli/cc"
FILESTORE_BASE = "https://www.fedlex.admin.ch/filestore/fedlex.data.admin.ch/eli/cc"

_DATE_ENTRY_IN_FORCE_RE = re.compile(
    r"jolux:dateEntryInForce[^>]*>([^<]+)<"
)
_PUBLICATION_DATE_RE = re.compile(
    r"jolux:publicationDate[^>]*>([^<]+)<"
)
_ARTICLE_HEADER_RE = re.compile(
    r"^Art\.\s+(\d+[a-z]?)\s+(.+)$", re.MULTILINE
)


class Article(BaseModel):
    law_id: str  # e.g. "dsg"
    law_title: str
    article_number: str  # e.g. "6", "6a"
    heading: str
    text: str
    source_url: str


def _get_consolidation_date(year: int, num: int, client: httpx.Client) -> date:
    """Looks up the current consolidation date (jolux:dateEntryInForce,
    falling back to jolux:publicationDate) for a law via the Fedlex RDF
    endpoint - this is the {date} path segment the filestore PDF URL needs."""
    resp = client.get(
        f"{RDF_BASE}/{year}/{num}",
        headers={"Accept": "application/rdf+xml"},
        follow_redirects=True,
        timeout=30.0,
    )
    resp.raise_for_status()
    m = _DATE_ENTRY_IN_FORCE_RE.search(resp.text) or _PUBLICATION_DATE_RE.search(resp.text)
    if not m:
        raise RuntimeError(
            f"Could not find dateEntryInForce/publicationDate for eli/cc/{year}/{num} "
            "- Fedlex's RDF response shape may have changed, check manually."
        )
    return date.fromisoformat(m.group(1))


def _download_pdf(year: int, num: int, consolidation_date: date, lang: str, client: httpx.Client) -> bytes:
    date_str = consolidation_date.strftime("%Y%m%d")
    url = (
        f"{FILESTORE_BASE}/{year}/{num}/{date_str}/{lang}/pdf-a/"
        f"fedlex-data-admin-ch-eli-cc-{year}-{num}-{date_str}-{lang}-pdf-a.pdf"
    )
    resp = client.get(url, follow_redirects=True, timeout=60.0, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    if resp.headers.get("content-type", "").split(";")[0] != "application/pdf":
        raise RuntimeError(f"Expected a PDF from {url}, got {resp.headers.get('content-type')}")
    return resp.content


def _extract_text(pdf_bytes: bytes) -> str:
    import pypdf

    reader = pypdf.PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() for page in reader.pages)


def parse_articles(full_text: str, law_id: str, law_title: str, source_url: str) -> list[Article]:
    """Splits the raw extracted PDF text into one Article per 'Art. N Title'
    heading. Page-footer noise (page numbers, the running law title/SR
    number printed on every page) is left in the surrounding text for now -
    good enough for a first retrieval pass, worth cleaning up later if it
    pollutes embeddings."""
    matches = list(_ARTICLE_HEADER_RE.finditer(full_text))
    articles = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        articles.append(
            Article(
                law_id=law_id,
                law_title=law_title,
                article_number=m.group(1),
                heading=m.group(2).strip(),
                text=body,
                source_url=source_url,
            )
        )
    return articles


def ingest_statute(
    law_id: str,
    law_title: str,
    year: int,
    num: int,
    lang: str = "de",
    display_url: str | None = None,
) -> list[Article]:
    """End-to-end: RDF lookup -> PDF download -> text extraction -> per-article split.

    year/num come from the law's eli/cc/{year}/{num} identifier (see
    config/sources.yaml). display_url is what gets stored as each Article's
    source_url for citation purposes - defaults to the human-facing Fedlex
    page even though the PDF is what was actually parsed.
    """
    with httpx.Client() as client:
        consolidation_date = _get_consolidation_date(year, num, client)
        pdf_bytes = _download_pdf(year, num, consolidation_date, lang, client)
    full_text = _extract_text(pdf_bytes)
    url = display_url or f"https://www.fedlex.admin.ch/eli/cc/{year}/{num}/{lang}"
    return parse_articles(full_text, law_id, law_title, url)


if __name__ == "__main__":
    # Quick manual check: python -m dsg_compliance.ingestion.fedlex
    articles = ingest_statute(
        law_id="dsg",
        law_title="Bundesgesetz über den Datenschutz (DSG)",
        year=2022,
        num=491,
    )
    print(f"{len(articles)} articles parsed")
    for a in articles[:3]:
        print(f"--- Art. {a.article_number}: {a.heading} ---")
        print(a.text[:200].replace("\n", " "))
        print()
