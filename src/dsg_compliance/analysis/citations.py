"""Builds the "which court precedents shape EDOEB's decisions" graph.

Real exploration (2026-09) found EDOEB decisions essentially never cite
each other - the "decision atlas" idea's original network-graph premise
(EDOEB decision -> EDOEB decision) doesn't hold in this corpus. What they
DO cite, repeatedly, is Swiss Federal Supreme Court case law (BGE -
"Bundesgerichtsentscheid"): a 6-decision manual sample found 5/6 citing at
least one BGE ruling, several appearing in more than one decision (e.g.
BGE 136 II 508, BGE 138 II 346). That's a real bipartite citation network -
EDOEB decisions on one side, BGE precedents on the other - and it answers a
genuinely useful question: which court precedents most shape EDOEB's data-
protection enforcement.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from ..ingestion.edoeb import Decision

# Matches "BGE 136 II 508" and close variants (spacing/OCR noise from PDF
# extraction) - deliberately does NOT try to capture the pinpoint "E. x.y"
# paragraph reference, just the case citation itself.
_BGE_RE = re.compile(r"BGE\s+(\d{2,3})\s+([IVX]+)\s+(\d{1,4})")


def extract_bge_citations(text: str) -> list[str]:
    """Returns normalized "BGE <vol> <part> <page>" citations found in
    `text`, in first-occurrence order, deduplicated."""
    seen: dict[str, None] = {}
    for m in _BGE_RE.finditer(text):
        citation = f"BGE {m.group(1)} {m.group(2)} {m.group(3)}"
        seen.setdefault(citation, None)
    return list(seen.keys())


class DecisionCitations(BaseModel):
    title: str
    pdf_url: str
    publication_date: str | None
    law_version: str
    bge_citations: list[str]


class CitationGraph(BaseModel):
    decisions: list[DecisionCitations]
    bge_frequency: dict[str, int]  # BGE citation -> number of distinct decisions citing it


def build_citation_graph(decisions: list[Decision]) -> CitationGraph:
    entries = []
    freq: dict[str, int] = {}
    for d in decisions:
        if not d.full_text:
            continue
        cites = extract_bge_citations(d.full_text)
        entries.append(
            DecisionCitations(
                title=d.title,
                pdf_url=d.pdf_url,
                publication_date=d.publication_date.isoformat() if d.publication_date else None,
                law_version=d.law_version,
                bge_citations=cites,
            )
        )
        for c in cites:
            freq[c] = freq.get(c, 0) + 1
    return CitationGraph(decisions=entries, bge_frequency=freq)
