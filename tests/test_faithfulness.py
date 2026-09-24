"""Regression test for the ragas-based faithfulness check itself (does the
metric actually discriminate a grounded answer from a fabricated one) -
not a test of the retrieval/generation pipeline, which is exercised by
`cli.py ask`/`eval-faithfulness` against the real index instead. Skipped
without GLM_API_KEY since it makes real LLM calls (the judge model)."""
from __future__ import annotations

import asyncio

import pytest

from dsg_compliance.config import settings
from dsg_compliance.rag.chat import AnswerWithSources, SourceRef

pytest.importorskip("ragas")

pytestmark = pytest.mark.skipif(
    not settings.glm_api_key, reason="GLM_API_KEY not set - faithfulness judge needs a real model call"
)

_CONTEXT = (
    "Art. 8 DSG: Der Verantwortliche und der Auftragsbearbeiter gewährleisten durch geeignete "
    "technische und organisatorische Massnahmen eine dem Risiko angemessene Datensicherheit."
)


def _answer(text: str) -> AnswerWithSources:
    return AnswerWithSources(
        question="Welche Massnahmen zur Datensicherheit verlangt das DSG?",
        answer=text,
        sources=[SourceRef(label="Art. 8 DSG", law_version="DSG", source_url="https://example.invalid", distance=0.1, excerpt=_CONTEXT)],
        model_used="test",
    )


def test_faithful_answer_scores_high():
    from dsg_compliance.rag.faithfulness import score_faithfulness

    score = asyncio.run(score_faithfulness(_answer(
        "Verantwortliche und Auftragsbearbeiter müssen gemäss Art. 8 DSG geeignete technische "
        "und organisatorische Massnahmen treffen, die dem Risiko angemessen sind."
    )))
    assert score >= 0.8


def test_fabricated_answer_scores_low():
    from dsg_compliance.rag.faithfulness import score_faithfulness

    score = asyncio.run(score_faithfulness(_answer(
        "Das DSG verlangt keine Sicherheitsmassnahmen, solange die Firma weniger als 50 "
        "Mitarbeitende hat und eine Busse von maximal CHF 500 zahlt."
    )))
    assert score <= 0.3


def test_score_faithfulness_requires_sources():
    from dsg_compliance.rag.faithfulness import score_faithfulness

    empty = AnswerWithSources(question="q", answer="a", sources=[], model_used="test")
    with pytest.raises(ValueError):
        asyncio.run(score_faithfulness(empty))
