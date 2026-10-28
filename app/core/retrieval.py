from __future__ import annotations

import re

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.core.embedder import DENSE_VECTOR, SPARSE_VECTOR, BgeM3Embedder
from app.core.glossary import Glossary


QDRANT_URL = settings.qdrant_url
QDRANT_COLLECTION = "rag_docs_hybrid"
EMBED_MODEL = settings.EMBED_MODEL


RETRIEVAL_THRESHOLD = 0.35
RETRIEVAL_TOP_K = 10

PREFETCH_MULTIPLIER = 2
DEDUP_THRESHOLD = 0.85

UNIT_TYPE_BOOST: dict[str, float] = {
    "clause":    0.00,
    "table":     0.03,
    "glossary":  0.00,
    "paragraph": 0.00,
    "heading":  -0.05,
}

type DocResult = dict


def _check_embed_model(client, collection: str, expected_model: str) -> None:
    """Fail-fast: коллекция должна быть проиндексирована той же моделью."""
    try:
        records, _ = client.scroll(
            collection_name=collection, limit=1,
            with_payload=True, with_vectors=False,
        )
    except Exception:
        return
    if not records:
        return
    actual = (records[0].payload or {}).get("embed_model")
    if actual and str(actual).strip().lower() != expected_model.strip().lower():
        raise RuntimeError(
            f"Коллекция '{collection}' проиндексирована моделью '{actual}', "
            f"а сервис настроен на '{expected_model}'. "
            f"Переиндексируйте с --reset"
        )


qdrant = QdrantClient(url=QDRANT_URL)
_check_embed_model(qdrant, QDRANT_COLLECTION, EMBED_MODEL)
embed_model = BgeM3Embedder(
    EMBED_MODEL,
    local_files_only=settings.EMBED_LOCAL_FILES_ONLY,
    device=settings.EMBED_DEVICE,
)

glossary = Glossary(qdrant, QDRANT_COLLECTION)


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _chunk_score_boost(meta: dict) -> float:
    unit_type = meta.get("unit_type") or meta.get("chunk_type", "paragraph")
    return UNIT_TYPE_BOOST.get(unit_type, 0.0)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _deduplicate(results: list[DocResult]) -> list[DocResult]:
    if len(results) <= 1:
        return results

    kept: list[DocResult] = []
    for candidate in results:
        if any(candidate["source"] == e["source"]
               and _jaccard(candidate["text"], e["text"]) >= DEDUP_THRESHOLD
               for e in kept):
            continue
        kept.append(candidate)

    return kept


def _to_result(chunk_id, text: str, meta: dict, score: float) -> DocResult:
    return {
        "text":     text,
        "source":   meta.get("source", "unknown"),
        "chunk_id": chunk_id,
        "score":    score,
        "section":  meta.get("section", ""),
        "_meta":    meta,
    }


def _hybrid_search(query: str, limit: int) -> list[DocResult]:
    """Dense + sparse одним запросом к Qdrant, слияние — RRF на стороне сервера."""
    try:
        dense_vecs, sparse_vecs = embed_model.encode([query])
    except Exception as exc:
        print(f"[retrieval] Ошибка эмбеддинга запроса: {exc}")
        return []

    prefetch = [
        models.Prefetch(
            query=dense_vecs[0].tolist(),
            using=DENSE_VECTOR,
            limit=limit,
            score_threshold=RETRIEVAL_THRESHOLD,
        ),
    ]

    sparse = sparse_vecs[0] if sparse_vecs else {}
    if sparse:
        prefetch.append(models.Prefetch(
            query=models.SparseVector(
                indices=list(sparse.keys()),
                values=list(sparse.values()),
            ),
            using=SPARSE_VECTOR,
            limit=limit,
        ))

    try:
        raw = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:
        print(f"[retrieval] Qdrant query error: {exc}")
        return []

    results: list[DocResult] = []
    for point in raw.points:
        meta = point.payload or {}
        text = meta.get("text", "")
        if not text:
            continue
        results.append(_to_result(point.id, text, meta, float(point.score)))
    return results


def lookup_glossary(query: str) -> list[DocResult]:
    """Точное совпадение термина/аббревиатуры в глоссарии (для мини-роутинга)."""
    return glossary.lookup(query)


def expand_query(query: str) -> str:
    """Заменить известные аббревиатуры на расшифровки перед поиском."""
    return glossary.expand(query)


def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    candidates = _hybrid_search(query, top_k * PREFETCH_MULTIPLIER)

    for r in candidates:
        r["score"] += _chunk_score_boost(r.get("_meta", {}))

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    deduped = _deduplicate(ranked)

    for r in deduped:
        r.pop("_meta", None)

    return deduped[:top_k]
