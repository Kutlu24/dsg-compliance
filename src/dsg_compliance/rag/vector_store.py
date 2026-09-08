"""Chroma-backed storage/retrieval for Chunk objects. Embeddings are
supplied by the caller (see embeddings.py) rather than letting Chroma
compute them internally, so the same embedding function is guaranteed to be
used at both write and query time regardless of Chroma's default."""
from __future__ import annotations

from ..chunking import Chunk
from ..config import settings

COLLECTION_NAME = "dsg_compliance"


def _client():
    import chromadb

    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def _collection():
    return _client().get_or_create_collection(COLLECTION_NAME)


def _chunk_metadata(c: Chunk) -> dict:
    # Chroma metadata values must be str/int/float/bool - None fields are
    # dropped rather than passed through.
    meta = {
        "source_type": c.source_type,
        "source_title": c.source_title,
        "law_version": c.law_version,
        "source_url": c.source_url,
    }
    if c.article_number is not None:
        meta["article_number"] = c.article_number
    if c.decision_date is not None:
        meta["decision_date"] = c.decision_date.isoformat()
    if c.decision_summary is not None:
        meta["decision_summary"] = c.decision_summary
    return meta


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    if not chunks:
        return
    collection = _collection()
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[_chunk_metadata(c) for c in chunks],
    )


def query(query_vector: list[float], top_k: int = 5, where: dict | None = None) -> list[dict]:
    collection = _collection()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
    )
    hits = []
    for i in range(len(result["ids"][0])):
        hits.append(
            {
                "chunk_id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],
            }
        )
    return hits


def count() -> int:
    return _collection().count()


def existing_ids() -> set[str]:
    return set(_collection().get(include=[])["ids"])
