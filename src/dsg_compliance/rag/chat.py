"""Cited-answer chat layer on top of the Chroma retrieval index. Retrieval
itself needs no LLM (local sentence-transformers embeddings); this module
adds the generation step - answering a question in prose, grounded only in
the retrieved excerpts, with a real source list attached programmatically
(not just trusted from the model's own citations - see AnswerWithSources.sources,
which comes straight from the retrieval hits, not from LLM output).
"""
from __future__ import annotations

from pydantic import BaseModel

from ..config import settings
from ..embeddings import embed_texts
from . import vector_store

_SYSTEM_PROMPT = """You are a legal-research assistant answering questions about Swiss data \
protection law (the DSG) for a researcher. You will be given a question and a set of excerpts \
retrieved from the DSG/DSV statute text and from FDPIC (EDOEB) published decisions/recommendations.

Rules:
- Answer ONLY using the provided excerpts. Do not use outside knowledge of Swiss or other law.
- Cite the specific source for every claim, using the label given with each excerpt (e.g.
  "Art. 18 DSV" or a decision's title). Never state a rule without naming which excerpt it
  came from.
- Some excerpts are marked aDSG (bis 31.08.2023) - the OLD, pre-revision law, with different
  article numbers than the current DSG. If you cite one, say explicitly that it is from the old
  law, not current law.
- If the excerpts do not contain enough information to answer, say so plainly - do not guess or
  fill gaps with general knowledge.
- This is informational only, not legal advice - end your answer with a short note saying so.
"""

_LANG_NAMES = {"de": "German", "en": "English"}


class SourceRef(BaseModel):
    label: str
    law_version: str
    source_url: str
    distance: float


class AnswerWithSources(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef]
    model_used: str


def _format_hit_label(metadata: dict) -> str:
    if metadata["source_type"] == "statute":
        return f"Art. {metadata.get('article_number')} {metadata['source_title']}"
    return f"{metadata['source_title']} ({metadata.get('decision_date', '?')})"


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        label = _format_hit_label(m)
        blocks.append(f"[{i}] {label} [{m['law_version']}]\n{h['text']}")
    return "\n\n".join(blocks)


def _system_prompt_for(lang: str) -> str:
    lang_name = _LANG_NAMES.get(lang, "German")
    # Explicit and repeated on purpose: the excerpts block below is entirely
    # German-language statute/decision text and dominates the prompt by
    # volume, which was observed to pull weaker models (glm-4.5-flash) into
    # answering in German even when asked in English - inferring the answer
    # language from the question alone was not reliable, so the caller now
    # passes the UI's selected language explicitly instead.
    return (
        f"{_SYSTEM_PROMPT}\n"
        f"- Write your entire answer in {lang_name}, regardless of the language the excerpts "
        f"below are written in. Only the source labels/titles stay in their original language."
    )


def _call_glm(question: str, context: str, lang: str) -> str:
    from openai import OpenAI

    if not settings.glm_api_key:
        raise RuntimeError("GLM_API_KEY not set in .env")
    client = OpenAI(api_key=settings.glm_api_key, base_url=settings.glm_base_url)
    response = client.chat.completions.create(
        model=settings.glm_model,
        messages=[
            {"role": "system", "content": _system_prompt_for(lang)},
            {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


def _call_gemini(question: str, context: str, lang: str) -> str:
    from google import genai

    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = f"{_system_prompt_for(lang)}\n\nExcerpts:\n\n{context}\n\nQuestion: {question}"
    interaction = client.interactions.create(model=settings.gemini_model, input=prompt)
    return interaction.output_text


def _call_anthropic(question: str, context: str, lang: str) -> str:
    import anthropic

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1024,
        system=_system_prompt_for(lang),
        messages=[{"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"}],
    )
    return response.content[0].text


_CALLERS = {"glm": _call_glm, "gemini": _call_gemini, "anthropic": _call_anthropic}


def ask(question: str, top_k: int = 5, where: dict | None = None, lang: str = "de") -> AnswerWithSources:
    [query_vector] = embed_texts([question], task_type="RETRIEVAL_QUERY")
    hits = vector_store.query(query_vector, top_k=top_k, where=where)
    if not hits:
        no_hits = {
            "de": "Keine relevanten Textstellen im Index gefunden.",
            "en": "No relevant excerpts found in the index.",
        }
        return AnswerWithSources(
            question=question,
            answer=no_hits.get(lang, no_hits["de"]),
            sources=[],
            model_used="none",
        )

    context = _format_context(hits)
    caller = _CALLERS[settings.chat_provider]
    answer_text = caller(question, context, lang)
    model_label = {
        "glm": settings.glm_model,
        "gemini": settings.gemini_model,
        "anthropic": "claude-opus-5",
    }[settings.chat_provider]

    sources = [
        SourceRef(
            label=_format_hit_label(h["metadata"]),
            law_version=h["metadata"]["law_version"],
            source_url=h["metadata"]["source_url"],
            distance=h["distance"],
        )
        for h in hits
    ]
    return AnswerWithSources(
        question=question, answer=answer_text, sources=sources,
        model_used=f"{settings.chat_provider}:{model_label}",
    )
