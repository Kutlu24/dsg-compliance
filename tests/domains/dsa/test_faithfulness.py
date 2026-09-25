"""Regression test for the ragas-based faithfulness check itself (does the
metric actually discriminate a grounded answer from a fabricated one) -
not a test of the retrieval/generation pipeline, which is exercised by
`cli.py ask`/`eval-faithfulness` against the real index instead. Skipped
without GLM_API_KEY since it makes real LLM calls (the judge model)."""
from __future__ import annotations

import asyncio

import pytest

from dsg_compliance.config import settings
from dsg_compliance.domains.dsa.rag.chat import AnswerWithSources, SourceRef

pytest.importorskip("ragas")

pytestmark = pytest.mark.skipif(
    not settings.glm_api_key, reason="GLM_API_KEY not set - faithfulness judge needs a real model call"
)

_CONTEXT = (
    "Article 2(1) DSA: This Regulation applies to intermediary services offered to "
    "recipients of the service that have their place of establishment or residence in "
    "the Union, irrespective of where the providers of those intermediary services have "
    "their place of establishment."
)


def _answer(text: str) -> AnswerWithSources:
    return AnswerWithSources(
        question="Does the DSA apply to a Swiss company with no EU establishment?",
        answer=text,
        sources=[SourceRef(label="Article 2 DSA", source_url="https://example.invalid", distance=0.1, excerpt=_CONTEXT)],
        model_used="test",
    )


def test_faithful_answer_scores_high():
    from dsg_compliance.domains.dsa.rag.faithfulness import score_faithfulness

    score = asyncio.run(score_faithfulness(_answer(
        "Yes - a provider established outside the EU is still covered if it offers "
        "intermediary services to recipients located in the EU, per Article 2(1)."
    )))
    assert score >= 0.8


def test_fabricated_answer_scores_low():
    from dsg_compliance.domains.dsa.rag.faithfulness import score_faithfulness

    score = asyncio.run(score_faithfulness(_answer(
        "No, the DSA never applies to any company outside the EU, and violators face "
        "a maximum fine of exactly CHF 50000 as set by Swiss law."
    )))
    assert score <= 0.3


def test_score_faithfulness_requires_sources():
    from dsg_compliance.domains.dsa.rag.faithfulness import score_faithfulness

    empty = AnswerWithSources(question="q", answer="a", sources=[], model_used="test")
    with pytest.raises(ValueError):
        asyncio.run(score_faithfulness(empty))
