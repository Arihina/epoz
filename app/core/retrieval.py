from __future__ import annotations

import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from app.core.config import settings


QDRANT_URL = settings.qdrant_url
QDRANT_COLLECTION = "rag_docs_hybrid"
EMBED_MODEL = settings.EMBED_MODEL

RETRIEVAL_THRESHOLD = 0.35
RETRIEVAL_TOP_K = 10
RRF_K = 60
DEDUP_THRESHOLD = 0.85

UNIT_TYPE_BOOST: dict[str, float] = {
    "clause":    0.00,
    "table":     0.03,
    "glossary":  0.00,
    "paragraph": 0.00,
    "heading": -0.05,
}

type DocResult = dict


embed_model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
qdrant = QdrantClient(url=QDRANT_URL)

documents: list[str] = []
metadatas: list[dict] = []
doc_ids: list[str] = []
bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _enrich_text(text: str, meta: dict) -> str:
    section = meta.get("section", "").strip()
    if section and section not in text:
        return f"{section}\n{text}"
    return text


def _init_bm25() -> None:
    global documents, metadatas, doc_ids, bm25
    try:
        documents, metadatas, doc_ids = [], [], []
        offset = None
        while True:
            records, offset = qdrant.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
                text = payload.get("text", "")
                if not text:
                    continue
                doc_ids.append(rec.id)
                documents.append(text)
                metadatas.append(payload)
            if offset is None:
                break
        print(f"[retrieval] Загружено {len(documents)} документов из Qdrant")
    except Exception as exc:
        print(f"[retrieval] Ошибка загрузки из Qdrant: {exc}")
        documents, metadatas, doc_ids = [], [], []

    if documents:
        enriched = [_enrich_text(doc, meta)
                    for doc, meta in zip(documents, metadatas)]
        tokenized = [_tokenize(t) for t in enriched]
        bm25 = BM25Okapi(tokenized)
        print("[retrieval] BM25-индекс создан")
    else:
        bm25 = None
        print("[retrieval] Предупреждение: BM25-индекс не создан — нет документов")


_init_bm25()


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
        is_dup = False
        for i, existing in enumerate(kept):
            if candidate["source"] != existing["source"]:
                continue
            if _jaccard(candidate["text"], existing["text"]) >= DEDUP_THRESHOLD:
                if (candidate.get("source_format") == "json"
                        and existing.get("source_format") == "md"):
                    kept[i] = candidate
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)

    return kept


def _to_result(chunk_id, text: str, meta: dict, score: float) -> DocResult:
    return {
        "text":          text,
        "source":        meta.get("source", "unknown"),
        "chunk_id":      chunk_id,
        "score":         score,
        "source_format": meta.get("source_format", ""),
        "chunk_type":    meta.get("chunk_type", ""),
        "section":       meta.get("section", ""),
        "_meta":         meta,
    }


def _bm25_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    if bm25 is None or not documents:
        return []

    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:top_k]

    results: list[DocResult] = []
    for idx in ranked:
        score = float(scores[idx])
        if score <= 0:
            continue
        meta = metadatas[idx] if idx < len(metadatas) else {}
        results.append(_to_result(doc_ids[idx], documents[idx], meta, score))
    return results


def _vector_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    try:
        embedding = embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].tolist()

        raw = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=embedding,
            using="dense",
            limit=top_k * 2,
            with_payload=True,
        )

        results: list[DocResult] = []
        for point in raw.points:
            score = float(point.score)
            if score < RETRIEVAL_THRESHOLD:
                continue
            meta = point.payload or {}
            results.append(_to_result(
                point.id, meta.get("text", ""), meta, score))
        return results

    except Exception as exc:
        print(f"[retrieval] Qdrant query error: {exc}")
        return []


def _rrf_merge(
    vector_results: list[DocResult],
    bm25_results:   list[DocResult],
    top_k:          int,
) -> list[DocResult]:
    merged: dict[str, DocResult] = {}

    for rank, r in enumerate(vector_results):
        key = r["chunk_id"]
        rrf = 1.0 / (RRF_K + rank + 1)
        if key not in merged:
            merged[key] = {**r, "score": 0.0}
        merged[key]["score"] += rrf

    for rank, r in enumerate(bm25_results):
        key = r["chunk_id"]
        rrf = 1.0 / (RRF_K + rank + 1)
        if key not in merged:
            merged[key] = {**r, "score": 0.0}
        merged[key]["score"] += rrf

    for r in merged.values():
        r["score"] += _chunk_score_boost(r.get("_meta", {}))

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    deduped = _deduplicate(ranked)

    for r in deduped:
        r.pop("_meta", None)

    return deduped[:top_k]


def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    vector = _vector_search(query, top_k * 2)
    bm25_ = _bm25_search(query, top_k * 2)
    return _rrf_merge(vector, bm25_, top_k)
