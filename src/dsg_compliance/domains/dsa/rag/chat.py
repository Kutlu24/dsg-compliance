"""Cited-answer chat layer on top of the Chroma retrieval index. Same
structure as DSG Compliance's rag/chat.py - see that module for the
retrieval/generation split rationale."""
from __future__ import annotations

from pydantic import BaseModel

from ....config import settings
from ....embeddings import embed_texts
from . import vector_store

_SYSTEM_PROMPT = """You are a legal-research assistant answering questions about the EU Digital \
Services Act (DSA, Regulation (EU) 2022/2065) for a Swiss business trying to assess whether and \
how it applies to them. You will be given a question and a set of excerpts retrieved from the DSA \
regulation text, European Commission guidance, and Swiss-relevance commentary.

Rules:
- Answer ONLY using the provided excerpts. Do not use outside knowledge of EU or other law.
- Cite the specific source for every claim, using the label given with each excerpt (e.g.
  "Article 16 DSA" or a guidance page's title). Never state a rule without naming which excerpt
  it came from.
- The DSA applies extraterritorially: a Switzerland-domiciled provider can still be in scope if it
  offers intermediary/hosting/platform services to recipients located in the EU (Art. 2(1)). If a
  question turns on this, be explicit about that condition rather than assuming EU domicile.
- If the excerpts do not contain enough information to answer, say so plainly - do not guess or
  fill gaps with general knowledge.
- This is informational only, not legal advice - end your answer with a short note saying so.
"""

_LANG_NAMES = {"de": "German", "en": "English"}


class SourceRef(BaseModel):
    label: str
    source_url: str
    distance: float
    excerpt: str


class AnswerWithSources(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef]
    model_used: str


def _format_hit_label(metadata: dict) -> str:
    if metadata["source_type"] == "statute":
        return f"Article {metadata.get('article_number')} DSA"
    return metadata["source_title"]


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        label = _format_hit_label(m)
        blocks.append(f"[{i}] {label}\n{h['text']}")
    return "\n\n".join(blocks)


def _system_prompt_for(lang: str) -> str:
    lang_name = _LANG_NAMES.get(lang, "English")
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


def ask(question: str, top_k: int = 5, where: dict | None = None, lang: str = "en") -> AnswerWithSources:
    [query_vector] = embed_texts([question], task_type="RETRIEVAL_QUERY")
    hits = vector_store.query(query_vector, top_k=top_k, where=where)
    if not hits:
        no_hits = {
            "de": "Keine relevanten Textstellen im Index gefunden.",
            "en": "No relevant excerpts found in the index.",
        }
        return AnswerWithSources(
            question=question,
            answer=no_hits.get(lang, no_hits["en"]),
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
            source_url=h["metadata"]["source_url"],
            distance=h["distance"],
            excerpt=h["text"],
        )
        for h in hits
    ]
    return AnswerWithSources(
        question=question, answer=answer_text, sources=sources,
        model_used=f"{settings.chat_provider}:{model_label}",
    )
