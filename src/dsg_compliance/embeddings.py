"""Hosted, free-tier embeddings via the Gemini API - no local torch model to
load, which matters because this service runs on Render's free 512MB-RAM
plan. A local sentence-transformers model (paraphrase-multilingual-mpnet-
base-v2) was tried first and reliably OOM-killed the process on every /ask
request; switching to a hosted embedding call removed torch from the
runtime entirely.

gemini-embedding-001 was picked over the newer gemini-embedding-2 because it
supports per-string batch embedding plus a task_type hint (RETRIEVAL_DOCUMENT
at ingest time, RETRIEVAL_QUERY at query time), which the newer model does
not - task_type meaningfully improves retrieval quality for asymmetric
document/query pairs like this one.
"""
from __future__ import annotations

import time

from .config import settings

_BATCH_SIZE = 40  # the free tier enforces a per-request TOKEN limit, not just a
# request-count one: a batch of 100 real chunks (~1200 chars each) was
# rejected instantly (429, no processing delay) while 50 succeeded - 40
# leaves margin since chunk length varies.
_PACING_SECONDS = 2.0  # stay well under the free tier's per-minute request cap
_RETRY_DELAYS = (10.0, 30.0, 60.0, 60.0)  # backoff on transient/rate-limit errors


def _client():
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    if not texts:
        return []

    from google.genai import types

    client = _client()
    config = types.EmbedContentConfig(task_type=task_type)
    vectors: list[list[float]] = []

    for start in range(0, len(texts), _BATCH_SIZE):
        if start:
            time.sleep(_PACING_SECONDS)
        batch = texts[start : start + _BATCH_SIZE]
        last_error: Exception | None = None
        for attempt, delay in enumerate((0.0, *_RETRY_DELAYS)):
            if delay:
                time.sleep(delay)
            try:
                result = client.models.embed_content(
                    model=settings.embedding_model, contents=batch, config=config
                )
                vectors.extend(e.values for e in result.embeddings)
                last_error = None
                break
            except Exception as e:  # broad: the SDK's rate-limit exception path is unstable
                last_error = e
        if last_error is not None:
            raise last_error

    return vectors
