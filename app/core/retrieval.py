from __future__ import annotations

import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb


CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "rag_docs_hybrid"
EMBED_MODEL = "intfloat/multilingual-e5-small"

QUERY_PREFIX = "query: "

RETRIEVAL_THRESHOLD = 0.35
RETRIEVAL_TOP_K = 5

RRF_K = 60

DEDUP_THRESHOLD = 0.92

JSON_SCORE_BOOST = 0.02

embed_model = SentenceTransformer(EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(CHROMA_COLLECTION)

documents: list[str] = []
metadatas: list[dict] = []
bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _init_bm25() -> None:
    global documents, metadatas, bm25
    try:
        result = collection.get(include=["documents", "metadatas"])
        if result and result["documents"]:
            documents = result["documents"]
            metadatas = result["metadatas"]
            print(
                f"[retrieval] Загружено {len(documents)} документов из ChromaDB")
        else:
            print("[retrieval] Коллекция пуста")
    except Exception as exc:
        print(f"[retrieval] Ошибка загрузки из ChromaDB: {exc}")
        documents, metadatas = [], []

    if documents:
        tokenized = [_tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized)
        print("[retrieval] BM25-индекс создан")
    else:
        bm25 = None
        print("[retrieval] Предупреждение: BM25-индекс не создан — нет документов")


_init_bm25()


# {"text": str, "source": str, "chunk_id": int|str, "score": float}
type DocResult = dict


def _doc_key(meta: dict, fallback_idx: int) -> tuple:
    return (
        meta.get("source", "unknown"),
        meta.get("anchor") or meta.get("chunk_id") or str(fallback_idx),
        meta.get("source_format", ""),
    )


def _deduplicate(results: list[DocResult]) -> list[DocResult]:
    if len(results) <= 1:
        return results

    def _similarity(a: str, b: str) -> float:
        sa, sb = set(_tokenize(a)), set(_tokenize(b))
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / len(sa | sb)

    kept: list[DocResult] = []
    for candidate in results:
        is_dup = False
        for existing in kept:
            if candidate["source"] != existing["source"]:
                continue
            sim = _similarity(candidate["text"], existing["text"])
            if sim >= DEDUP_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)

    return kept


def _bm25_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    if bm25 is None or not documents:
        return []

    scores = bm25.get_scores(_tokenize(query))
    ranked = np.argsort(scores)[::-1][:top_k]

    results: list[DocResult] = []
    for idx in ranked:
        score = float(scores[idx])
        if score <= 0:
            continue
        meta = metadatas[idx] if idx < len(metadatas) else {}
        results.append({
            "text":          documents[idx],
            "source":        meta.get("source", "unknown"),
            "chunk_id":      _doc_key(meta, int(idx)),
            "score":         score,
            "source_format": meta.get("source_format", ""),
            "_meta":         meta,
        })
    return results


def _vector_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    try:
        embedding = embed_model.encode(
            [QUERY_PREFIX + query] if QUERY_PREFIX else [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].tolist()

        raw = collection.query(
            query_embeddings=[embedding],
            n_results=top_k * 2,
            include=["documents", "metadatas", "distances"],
        )

        results: list[DocResult] = []
        if raw["documents"] and raw["documents"][0]:
            for i, doc in enumerate(raw["documents"][0]):
                score = 1.0 - float(raw["distances"][0][i])
                if score < RETRIEVAL_THRESHOLD:
                    continue
                meta = (raw["metadatas"][0][i] or {}
                        ) if raw["metadatas"] else {}
                results.append({
                    "text":          doc,
                    "source":        meta.get("source", "unknown"),
                    "chunk_id":      _doc_key(meta, i),
                    "score":         score,
                    "source_format": meta.get("source_format", ""),
                    "_meta":         meta,
                })
        return results

    except Exception as exc:
        print(f"[retrieval] ChromaDB query error: {exc}")
        return []


def _rrf_merge(
    vector_results: list[DocResult],
    bm25_results:   list[DocResult],
    top_k:          int,
) -> list[DocResult]:
    merged: dict[tuple, DocResult] = {}

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
        if r.get("source_format") == "json":
            r["score"] += JSON_SCORE_BOOST

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    deduped = _deduplicate(ranked)

    for r in deduped:
        r.pop("_meta", None)

    return deduped[:top_k]


def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    vector = _vector_search(query, top_k * 2)
    bm25_ = _bm25_search(query, top_k * 2)
    return _rrf_merge(vector, bm25_, top_k)
