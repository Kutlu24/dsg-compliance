"""Splits ingested DSA articles and guidance pages into a single, uniformly-
shaped Chunk type ready for embedding + storage in Chroma. Mirrors DSG
Compliance's chunking.py structure (see that project for the original
rationale); the `source_type` values differ ("statute" | "guidance" here,
no "decision" - DSA has no EDOEB-equivalent published-decision archive yet,
see sources.yaml's "not yet sourced" note).
"""
from __future__ import annotations

import hashlib

from pydantic import BaseModel

from .ingestion.eurlex import Article
from .ingestion.guidance import GuidancePage


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_type: str  # "statute" | "guidance"
    source_title: str
    article_number: str | None = None
    source_url: str


def chunk_articles(articles: list[Article]) -> list[Chunk]:
    """One chunk per article - DSA articles run from a few lines to ~1.5
    pages; short enough that further splitting would just fragment a single
    obligation across chunks."""
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
                text=f"Article {a.article_number}\n\n{a.text}",
                source_type="statute",
                source_title=a.law_title,
                article_number=a.article_number,
                source_url=a.source_url,
            )
        )
    return chunks


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Same sliding-window paragraph splitter as DSG Compliance's
    chunking.py - see that module for the rationale."""
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

    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail}\n{c}")
    return overlapped


def chunk_guidance(pages: list[GuidancePage], max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    chunks = []
    for p in pages:
        pieces = _split_text(p.text, max_chars, overlap)
        # hashlib (not built-in hash()) for a chunk_id stable across runs -
        # see DSG Compliance's chunking.py for why this matters for
        # vector_store.upsert_chunks's id-based dedup.
        url_digest = hashlib.sha256(p.source_url.encode("utf-8")).hexdigest()[:16]
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"guidance-{url_digest}-{i}",
                    text=piece,
                    source_type="guidance",
                    source_title=p.title,
                    source_url=p.source_url,
                )
            )
    return chunks
