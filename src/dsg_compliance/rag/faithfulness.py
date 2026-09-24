"""Automated check for whether a cited answer's claims actually trace back
to the excerpts it cites - the thing this whole app is supposed to
guarantee, but which was previously only ever verified by a human reading
the answer next to its sources. Uses ragas's Faithfulness metric: it
decomposes the answer into individual statements and checks each one
against the retrieved context, rather than a single fuzzy similarity
score, which matches "cite every claim" being the actual system prompt
rule in chat.py. Not the same thing as analysis/citations.py's BGE
citation graph - that's a corpus-level "who cites whom" analysis; this is
a per-answer "does this specific answer's claims hold up" check.

The judge LLM is the same GLM provider/model this app already uses for
answer generation (see config.py) - no second API key, no new provider
dependency, and it keeps the eval self-hostable on the same free tier the
rest of the app runs on.

Pinned dependency note: ragas>=0.4 requires langchain-community, but
ragas 0.4.3's import of `langchain_community.chat_models.vertexai` breaks
against langchain-community>=0.4 (module moved/removed there) - see
pyproject.toml's `eval` extra, which pins langchain-community==0.3.31.

max_tokens=8192 below is not cosmetic: ragas's statement-generation step
defaults to max_tokens=1024, which silently truncates the model's JSON
output and raises `instructor.exceptions.IncompleteOutputException`.
Found by running this against real GLM calls, not from documentation --
1024 fails immediately, 4096 still fails against a realistic top_k=5
multi-source answer (~1.5k chars of answer + ~6k chars of context), 8192
is the value actually verified working against real retrieval output
(verified in the sibling dsa-compliance project's identical setup).
"""
from __future__ import annotations

from openai import AsyncOpenAI

from ..config import settings
from .chat import AnswerWithSources

# A small, representative spread of real DSG questions (not exhaustive) -
# used by `cli.py eval-faithfulness` when no question is given explicitly.
# Covers a statute-only question, a pre/post-revision distinction (a rule
# this app's system prompt specifically calls out), and an EDOEB-guidance
# question, so a regression in any one area is more likely to surface than
# repeating one question.
DEFAULT_EVAL_QUESTIONS = [
    "Welche technischen und organisatorischen Massnahmen (TOM) verlangt das DSG?",
    "Was hat sich bei den Melde- und Informationspflichten zwischen dem alten und dem revidierten DSG geändert?",
    "Was sagt der EDÖB zur grenzüberschreitenden Datenübermittlung?",
]


def _judge_llm():
    from ragas.llms import llm_factory

    if not settings.glm_api_key:
        raise RuntimeError("GLM_API_KEY not set in .env - the faithfulness judge uses the same GLM model as answer generation.")
    client = AsyncOpenAI(api_key=settings.glm_api_key, base_url=settings.glm_base_url)
    return llm_factory(settings.glm_model, client=client, max_tokens=8192)


async def score_faithfulness(result: AnswerWithSources) -> float:
    """Returns a 0..1 faithfulness score for one already-generated answer:
    1.0 means every claim in `result.answer` is supported by at least one
    of `result.sources`' excerpts; lower scores mean some claim isn't
    traceable to the cited context (a fabrication, or a claim the model
    added from outside knowledge despite the system prompt forbidding it)."""
    from ragas.metrics.collections import Faithfulness

    if not result.sources:
        raise ValueError("Cannot score faithfulness with no retrieved sources - nothing to check the answer against.")
    faithfulness = Faithfulness(llm=_judge_llm())
    score = await faithfulness.ascore(
        user_input=result.question,
        response=result.answer,
        retrieved_contexts=[s.excerpt for s in result.sources],
    )
    return score.value
