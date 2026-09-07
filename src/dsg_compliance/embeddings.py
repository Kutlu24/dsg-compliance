"""Local, free embeddings via sentence-transformers - no API key, no quota,
no per-call cost. Chosen over Claude (which has no embeddings endpoint) and
paid providers (OpenAI/Voyage) partly on principle for a project meant to be
runnable by any researcher without a paid account, and partly because the
Anthropic account backing this workspace was out of credit when this was
built (2026-09).

paraphrase-multilingual-mpnet-base-v2 was picked for solid German coverage
(it's trained on 50+ languages including German) - swap
DSG_EMBEDDING_MODEL in .env if a German-legal-specific model turns out to
retrieve better once there's a real query set to evaluate against.
"""
from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model().encode(texts, show_progress_bar=len(texts) > 20, normalize_embeddings=True)
    return vectors.tolist()
