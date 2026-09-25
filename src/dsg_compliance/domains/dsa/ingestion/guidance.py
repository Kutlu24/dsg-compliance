"""Ingestion of secondary/guidance sources: European Commission DSA policy
pages and Swiss-relevance commentary (config/sources.yaml's `guidance` and
`swiss_relevance` lists). Unlike EDOEB's decision archive (DSG Compliance),
these are single HTML pages, not a paginated index to crawl - so this module
is a plain "fetch one URL, extract main content text" fetcher rather than a
site-specific scraper.

Extraction is intentionally generic (try <main>, then <article>, then
<body>; strip nav/header/footer/script/style) rather than hand-tuned
per-domain selectors, since sources.yaml mixes a Drupal-based EC site with
law-firm CMS pages that don't share markup. Good enough for a first pass -
if a specific source's extracted text turns out noisy, add a per-id override
here rather than generalizing prematurely.
"""
from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

_STRIP_TAGS = ("script", "style", "nav", "header", "footer", "noscript", "svg", "form")


class GuidancePage(BaseModel):
    source_id: str
    title: str
    text: str
    source_url: str


def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    # Take the LARGEST <main>/<article> by extracted text length, not the
    # first one - a page can have several <article> elements (e.g. a "related
    # content" teaser card ahead of the real one in document order), and
    # picking the first blindly grabbed a sidebar teaser instead of the
    # actual page content on a real source (pwc.ch, confirmed 2026-09-09).
    candidates = soup.find_all(["main", "article"]) or [soup.body or soup]
    container = max(candidates, key=lambda c: len(c.get_text(strip=True)))
    text = container.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    # Collapse runs of short nav-like lines (menu items etc.) that survive
    # tag stripping - a real paragraph is rarely under ~25 chars.
    return "\n".join(lines)


def fetch_guidance_page(source_id: str, title: str, url: str) -> GuidancePage:
    resp = httpx.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
        timeout=30.0,
    )
    resp.raise_for_status()
    text = _extract_main_text(resp.text)
    if len(text) < 200:
        raise RuntimeError(
            f"Extracted suspiciously little text ({len(text)} chars) from {url} "
            f"(source_id={source_id!r}) - page structure may not match the generic "
            "extractor, check manually."
        )
    return GuidancePage(source_id=source_id, title=title, text=text, source_url=url)


def ingest_guidance_sources(entries: list[dict]) -> list[GuidancePage]:
    pages = []
    for entry in entries:
        pages.append(fetch_guidance_page(entry["id"], entry["title"], entry["url"]))
    return pages


if __name__ == "__main__":
    # Quick manual check: python -m dsa_compliance.ingestion.guidance
    page = fetch_guidance_page(
        "ec-dsa-main",
        "The Digital Services Act",
        "https://digital-strategy.ec.europa.eu/en/policies/digital-services-act",
    )
    print(f"{len(page.text)} chars extracted")
    print(page.text[:500])
