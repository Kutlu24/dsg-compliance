"""Ingestion of EU regulation text via the Publications Office's CELLAR
repository - NOT via eur-lex.europa.eu directly.

eur-lex.europa.eu sits behind an AWS WAF that returns HTTP 202 with
`x-amzn-waf-action: challenge` (a JS challenge) to every plain httpx/curl
request, including its own "machine-readable format" endpoints
(/legal-content/.../TXT/XML/, /TXT/PDF/) - confirmed 2026-09-09, so none of
those URLs are usable for unattended ingestion despite being documented as
the intended API.

What DOES work (same date): publications.europa.eu/resource/celex/{CELEX}
uses HTTP content negotiation (no WAF) and 303-redirects to a CELLAR
manifestation. Requesting `Accept: application/xhtml+xml` on
`.../celex/{CELEX}.{LANGCODE}` (3-letter ISO 639-2 codes: ENG, DEU, FRA, ...)
lands on the original Official Journal XHTML, with each article's heading in
a `<p class="oj-ti-art">Article N</p>` element - confirmed against CELEX
32022R2065 (the DSA): 93 Article headings, matching the real article count.

This is the ORIGINAL (as-published) text, not a live-tracked consolidated
version - the DSA has not been substantively amended since publication as of
2026-09, so this is fine for now. If that changes, re-derive the correct
CELEX/expression for the current consolidated version before trusting this
blindly (see the "branch" notice manifestation used for `_verify_exists`
below for how to check).
"""
from __future__ import annotations

import re

import httpx
from pydantic import BaseModel

CELLAR_BASE = "http://publications.europa.eu/resource/celex"
_ARTICLE_HEADER_RE = re.compile(
    r'<p[^>]*\bclass="oj-ti-art"[^>]*>\s*Article[\s\xa0]+(\d+[a-z]?)\s*</p>',
)
_TAG_RE = re.compile(r"<[^>]+>")


class Article(BaseModel):
    law_id: str
    law_title: str
    article_number: str
    text: str
    source_url: str


def _strip_tags(html: str) -> str:
    text = _TAG_RE.sub("\n", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _fetch_xhtml(celex: str, lang: str) -> str:
    resp = httpx.get(
        f"{CELLAR_BASE}/{celex}.{lang}",
        headers={"Accept": "application/xhtml+xml", "User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
        timeout=60.0,
    )
    resp.raise_for_status()
    if not resp.text.strip():
        raise RuntimeError(
            f"Empty response for CELEX {celex} ({lang}) - CELLAR content negotiation "
            "may have changed, check manually with curl -v."
        )
    return resp.text


def parse_articles(xhtml: str, law_id: str, law_title: str, source_url: str) -> list[Article]:
    headers = list(_ARTICLE_HEADER_RE.finditer(xhtml))
    articles = []
    for i, m in enumerate(headers):
        number = m.group(1)
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(xhtml)
        text = _strip_tags(xhtml[start:end])
        if not text:
            continue
        articles.append(
            Article(
                law_id=law_id,
                law_title=law_title,
                article_number=number,
                text=text,
                source_url=source_url,
            )
        )
    return articles


def ingest_statute(
    law_id: str, law_title: str, celex: str, lang: str = "ENG", display_url: str | None = None
) -> list[Article]:
    xhtml = _fetch_xhtml(celex, lang)
    url = display_url or f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
    return parse_articles(xhtml, law_id, law_title, url)


if __name__ == "__main__":
    # Quick manual check: python -m dsa_compliance.ingestion.eurlex
    articles = ingest_statute(law_id="dsa", law_title="Digital Services Act", celex="32022R2065")
    print(f"{len(articles)} articles parsed")
    for a in articles[:3]:
        print(f"--- Article {a.article_number} ---")
        print(a.text[:200].replace("\n", " "))
        print()
