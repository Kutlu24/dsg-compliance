"""Splits ingested statute articles and EDOEB decisions into a single,
uniformly-shaped Chunk type ready for embedding + storage in Chroma.

Statute articles are already naturally chunk-sized (one DSG article is
rarely more than a page); decisions are full PDF texts (20-50k characters)
and need real splitting. Every chunk carries enough metadata to build a
correct citation back to its source without re-fetching anything, and -
critically for the decisions - the `law_version` field so a chunk from the
old (pre-revision) aDSG archive can never be presented as current law.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from .ingestion.edoeb import Decision
from .ingestion.fedlex import Article


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_type: str  # "statute" | "decision"
    source_title: str  # law title, or decision title
    law_version: str  # e.g. "DSG (ab 01.09.2023)", "aDSG (bis 31.08.2023)"
    article_number: str | None = None
    decision_date: date | None = None
    decision_summary: str | None = None  # FDPIC's own description, if any (see edoeb.py)
    source_url: str


def chunk_articles(articles: list[Article], law_version: str = "DSG (ab 01.09.2023)") -> list[Chunk]:
    """One chunk per article - DSG/DSV articles are short enough that
    further splitting would just fragment a single legal provision across
    chunks, hurting retrieval more than it helps.

    Article numbers can repeat within a single parsed document: laws
    commonly end with an "Aenderung anderer Erlasse" annex that amends
    OTHER laws, and that annex reuses "Art. N" numbering for those other
    laws' articles - so parse_articles() can legitimately return more than
    one "Art. 3" etc. Rather than silently dropping/overwriting one on
    chunk_id collision, every occurrence is kept with a disambiguating
    suffix (first occurrence gets none, so normal citations still look
    like "dsg-art25")."""
    seen: dict[str, int] = {}
    chunks = []
    for a in articles:
        key = f"{a.law_id}-art{a.article_number}"
        n = seen.get(key, 0)
        seen[key] = n + 1
        chunk_id = key if n == 0 else f"{key}-dup{n}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=f"Art. {a.article_number} {a.heading}\n\n{a.text}",
                source_type="statute",
                source_title=a.law_title,
                law_version=law_version,
                article_number=a.article_number,
                source_url=a.source_url,
            )
        )
    return chunks


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Simple sliding-window splitter on paragraph boundaries where
    possible, falling back to a hard cut. Good enough for a first pass -
    revisit with a sentence-aware splitter if retrieval quality on
    decisions turns out weak."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = f"{current}\n{para}".strip()
            continue
        if current:
            chunks.append(current)
        if len(para) > max_chars:
            for i in range(0, len(para), max_chars - overlap):
                chunks.append(para[i : i + max_chars])
            current = ""
        else:
            current = para
    if current:
        chunks.append(current)

    # Add overlap between consecutive chunks so a fact split across a
    # boundary is still findable from either side.
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail}\n{c}")
    return overlapped


def chunk_decisions(decisions: list[Decision], max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    chunks = []
    for d in decisions:
        if not d.full_text:
            continue
        pieces = _split_text(d.full_text, max_chars, overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"decision-{abs(hash(d.pdf_url))}-{i}",
                    text=piece,
                    source_type="decision",
                    source_title=d.title,
                    law_version=d.law_version,
                    decision_date=d.publication_date,
                    decision_summary=d.description or None,
                    source_url=d.pdf_url,
                )
            )
    return chunks
