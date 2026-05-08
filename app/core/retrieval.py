from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb


CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_THRESHOLD = 0.4
RETRIEVAL_TOP_K = 3


embed_model = SentenceTransformer(EMBED_MODEL)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(CHROMA_COLLECTION)

documents: list[str] = []
metadatas: list[dict] = []
bm25: BM25Okapi | None = None


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

    if documents:
        tokenized = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized)
        print("[retrieval] BM25-индекс создан")
    else:
        bm25 = None
        print("[retrieval] Предупреждение: BM25-индекс не создан — нет документов")


_init_bm25()


# {"text": str, "source": str, "chunk_id": int|str, "score": float}
type DocResult = dict


def _bm25_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    if bm25 is None or not documents:
        return []

    scores = bm25.get_scores(query.lower().split())
    ranked = np.argsort(scores)[::-1][:top_k]

    results: list[DocResult] = []
    for idx in ranked:
        score = float(scores[idx])
        if score <= 0:
            continue
        meta = metadatas[idx] if idx < len(metadatas) else {}
        results.append({
            "text": documents[idx],
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", idx),
            "score": score,
        })
    return results


def _vector_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    try:
        embedding = embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].tolist()

        raw = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
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
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "chunk_id": meta.get("chunk_id", i),
                    "score": score,
                })
        return results

    except Exception as exc:
        print(f"[retrieval] ChromaDB query error: {exc}")
        return []


def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[DocResult]:
    combined = _vector_search(query, top_k) + _bm25_search(query, top_k)

    unique: dict[tuple, DocResult] = {}
    for r in combined:
        key = (r["source"], r["chunk_id"])
        if key not in unique or unique[key]["score"] < r["score"]:
            unique[key] = r

    return sorted(unique.values(), key=lambda x: x["score"], reverse=True)[:top_k]
