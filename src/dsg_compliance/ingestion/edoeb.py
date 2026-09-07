"""Ingestion of FDPIC/EDOEB (Eidgenoessischer Datenschutz- und
Oeffentlichkeitsbeauftragter) published dispositions (Verfuegungen) - the
concrete case-decision examples requested alongside the bare statute text.

https://www.edoeb.admin.ch/de/verfuegungen is server-rendered (not an
Angular shell like fedlex.admin.ch), grouped by year:

    <h2 id="{year}">{year}</h2>
    ...
    <ul class="list">
      <li><a class="download-item" href="{pdf_url}">
        <div>
          <h4 class="download-item__title">{title}</h4>
          <p class="download-item__description">{articles referenced, outcome}</p>
          <p class="download-item__meta-info">...<span>{publication date}</span></p>
        </div>
      </a></li>
      ...
    </ul>

The description field usefully already names the DSG articles at issue
(e.g. "Art. 6 DSG - Einhaltung der Datenbearbeitungsgrundsaetze"), which is
worth keeping alongside the full decision text for retrieval.

Historical (pre-01.09.2023, old-law) recommendations live on a separate,
similarly-structured page - see EDOEB_ADSG_URL below. Same a.download-item
markup, just without the .download-item__description element.

IMPORTANT - old vs. current law: the aDSG (old DSG, pre-revision) archive
uses different article numbers than the current DSG (e.g. old Art. 4 aDSG
is not the same provision as current Art. 6 DSG). Mixing the two without
labeling risks a RAG answer citing an old article number as if it were
still current. Every Decision here carries a `law_version` field
("DSG (ab 01.09.2023)" or "aDSG (bis 31.08.2023)") set automatically from
which page it came from - never drop this field before embedding/storing,
and surface it in any answer that cites one of these decisions.
"""
from __future__ import annotations

import re
from datetime import date
from io import BytesIO

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

EDOEB_VERFUEGUNGEN_URL = "https://www.edoeb.admin.ch/de/verfuegungen"
EDOEB_ADSG_URL = "https://www.edoeb.admin.ch/de/schlussberichte-empfehlungen-bis-31082023"

_GERMAN_MONTHS = {
    "Januar": 1, "Februar": 2, "März": 3, "April": 4, "Mai": 5, "Juni": 6,
    "Juli": 7, "August": 8, "September": 9, "Oktober": 10, "November": 11, "Dezember": 12,
}
_DATE_RE = re.compile(r"(\d{1,2})\.\s+(\w+)\s+(\d{4})")


class Decision(BaseModel):
    title: str
    description: str  # FDPIC's own one-line summary, often names the DSG articles at issue
    publication_date: date | None
    pdf_url: str
    law_version: str  # "DSG (ab 01.09.2023)" or "aDSG (bis 31.08.2023)" - see module docstring
    full_text: str | None = None  # populated by fetch_decision_text()


def _parse_german_date(text: str) -> date | None:
    m = _DATE_RE.search(text)
    if not m:
        return None
    day, month_name, year = m.groups()
    month = _GERMAN_MONTHS.get(month_name)
    if not month:
        return None
    return date(int(year), month, int(day))


def list_decisions(url: str = EDOEB_VERFUEGUNGEN_URL) -> list[Decision]:
    """Scrapes the metadata (title, description, date, PDF link) for every
    published decision on the given EDOEB listing page - does not download
    the PDFs themselves, see fetch_decision_text() for that.

    law_version is inferred from which URL was passed: EDOEB_ADSG_URL means
    the old (pre-revision) law, anything else is treated as current-DSG."""
    law_version = "aDSG (bis 31.08.2023)" if url == EDOEB_ADSG_URL else "DSG (ab 01.09.2023)"

    resp = httpx.get(url, timeout=30.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    decisions = []
    for item in soup.select("a.download-item"):
        href = item.get("href")
        title_el = item.select_one(".download-item__title")
        desc_el = item.select_one(".download-item__description")
        meta_el = item.select_one(".download-item__meta-info")
        if not href or not title_el:
            continue
        meta_text = meta_el.get_text(" ", strip=True) if meta_el else ""
        decisions.append(
            Decision(
                title=title_el.get_text(strip=True),
                description=desc_el.get_text(" ", strip=True) if desc_el else "",
                publication_date=_parse_german_date(meta_text),
                pdf_url=href,
                law_version=law_version,
            )
        )
    return decisions


def fetch_decision_text(decision: Decision, client: httpx.Client | None = None) -> str:
    """Downloads and extracts the full text of one decision's PDF."""
    import pypdf

    owns_client = client is None
    client = client or httpx.Client()
    try:
        resp = client.get(decision.pdf_url, timeout=60.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        reader = pypdf.PdfReader(BytesIO(resp.content))
        return "\n".join(page.extract_text() for page in reader.pages)
    finally:
        if owns_client:
            client.close()


def ingest_decisions(url: str = EDOEB_VERFUEGUNGEN_URL, with_text: bool = True) -> list[Decision]:
    decisions = list_decisions(url)
    if with_text:
        with httpx.Client() as client:
            for d in decisions:
                d.full_text = fetch_decision_text(d, client)
    return decisions


if __name__ == "__main__":
    # Quick manual check: python -m dsg_compliance.ingestion.edoeb
    current = list_decisions(EDOEB_VERFUEGUNGEN_URL)
    adsg = list_decisions(EDOEB_ADSG_URL)
    print(f"{len(current)} current-DSG decisions, {len(adsg)} aDSG (pre-revision) decisions")
    for d in current[:2] + adsg[:2]:
        print(f"--- [{d.law_version}] {d.publication_date} | {d.title} ---")
        print("articles/outcome:", (d.description or "(no description on this page)")[:200])
        print()
