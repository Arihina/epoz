from __future__ import annotations

import re
import threading

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.core.embedder import DENSE_VECTOR, SPARSE_VECTOR, BgeM3Embedder, Reranker
from app.core.glossary import Glossary


QDRANT_URL = settings.qdrant_url
QDRANT_COLLECTION = settings.QDRANT_COLLECTION
EMBED_MODEL = settings.EMBED_MODEL


RETRIEVAL_TOP_K = 10

PREFETCH_MULTIPLIER = 2
RERANK_CANDIDATE_MULTIPLIER = 4
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


# Ленивая инициализация тяжёлых зависимостей

_qdrant: QdrantClient | None = None
_embed_model: BgeM3Embedder | None = None
_reranker: Reranker | None = None
_glossary: Glossary | None = None

_qdrant_lock = threading.Lock()
_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()
_glossary_lock = threading.Lock()
_reranker_checked = False


def get_qdrant() -> QdrantClient:
    """Ленивый singleton Qdrant-клиента.

    Проверка «коллекция проиндексирована той же моделью» выполняется здесь, а не
    на импорте: при недоступном Qdrant сервис всё равно поднимается, а ошибка
    всплывает на первом поиске.
    """
    global _qdrant

    if _qdrant is not None:
        return _qdrant

    with _qdrant_lock:
        if _qdrant is None:
            client = QdrantClient(url=QDRANT_URL)
            _check_embed_model(client, QDRANT_COLLECTION, EMBED_MODEL)
            _qdrant = client
    return _qdrant


def get_embed_model() -> BgeM3Embedder:
    """Ленивая загрузка BGE-M3: ноль VRAM при импорте, одна копия на процесс."""
    global _embed_model

    if _embed_model is not None:
        return _embed_model

    with _embed_lock:
        if _embed_model is None:
            get_qdrant()
            print(
                f"[retrieval] Загрузка эмбеддера {EMBED_MODEL} "
                f"на {settings.EMBED_DEVICE}"
            )
            _embed_model = BgeM3Embedder(
                EMBED_MODEL,
                local_files_only=settings.EMBED_LOCAL_FILES_ONLY,
                device=settings.EMBED_DEVICE,
            )
    return _embed_model


def get_reranker() -> Reranker | None:
    """Ленивый реранкер; None, если отключён конфигом или не загрузился."""
    global _reranker, _reranker_checked

    if _reranker_checked:
        return _reranker

    with _rerank_lock:
        if not _reranker_checked:
            if settings.RERANK_ENABLED:
                try:
                    print(
                        f"[retrieval] Загрузка реранкера {settings.RERANK_MODEL} "
                        f"на {settings.EMBED_DEVICE}"
                    )
                    _reranker = Reranker(
                        settings.RERANK_MODEL,
                        local_files_only=settings.EMBED_LOCAL_FILES_ONLY,
                        device=settings.EMBED_DEVICE,
                    )
                except Exception as exc:
                    print(
                        "[retrieval] Reranker недоступен, "
                        f"переранжирование отключено: {exc}"
                    )
                    _reranker = None
            _reranker_checked = True
    return _reranker


def get_glossary() -> Glossary:
    """Ленивая загрузка глоссария (scroll по Qdrant), а не на импорте."""
    global _glossary

    if _glossary is not None:
        return _glossary

    with _glossary_lock:
        if _glossary is None:
            _glossary = Glossary(get_qdrant(), QDRANT_COLLECTION)
    return _glossary


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
    embed_model = get_embed_model()
    qdrant = get_qdrant()

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
            score_threshold=settings.RETRIEVAL_THRESHOLD,
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
    return get_glossary().lookup(query)


def expand_query(query: str) -> str:
    """Заменить известные аббревиатуры на расшифровки перед поиском."""
    return get_glossary().expand(query)


def _minmax(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _rerank(
    query: str,
    candidates: list[DocResult],
    reranker: Reranker,
) -> list[DocResult]:
    """Смесь RRF-скора и cross-encoder'а: RERANK_ALPHA — вес cross-encoder'а."""
    texts = [c["text"] for c in candidates]
    cross = reranker.score(query, texts)
    rrf = [c["score"] for c in candidates]

    cross_norm = _minmax(cross)
    rrf_norm = _minmax(rrf)

    alpha = settings.RERANK_ALPHA
    for c, cn, rn in zip(candidates, cross_norm, rrf_norm):
        c["score"] = alpha * cn + (1 - alpha) * rn
    return candidates


def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    get_embed_model()
    reranker = get_reranker()

    multiplier = RERANK_CANDIDATE_MULTIPLIER if reranker else PREFETCH_MULTIPLIER
    candidates = _hybrid_search(query, top_k * multiplier)

    if reranker is not None:
        candidates = _rerank(query, candidates, reranker)

    for r in candidates:
        r["score"] += _chunk_score_boost(r.get("_meta", {}))

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    deduped = _deduplicate(ranked)

    for r in deduped:
        r.pop("_meta", None)

    return deduped[:top_k]
