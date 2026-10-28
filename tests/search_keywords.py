"""
Лексический (sparse) поиск по базе знаний: веса BGE-M3 напрямую в Qdrant.

В отличие от BM25, веса считает сама BGE-M3 (обучаемая sparse-голова),
поэтому запрос и чанк сравниваются по одним и тем же token_id.

Использование:
    python search_keywords.py "закупочная комиссия состав"

    # больше результатов
    python search_keywords.py "председатель комиссии" --top 10

    # фильтр по документу
    python search_keywords.py "председатель" --source "doc_name"

    # фильтр по типу чанка
    python search_keywords.py "председатель" --type table

    # показать, какие токены запроса совпали с чанком и их вклад
    python search_keywords.py "состав комиссии" --explain
"""

import argparse
from collections import defaultdict

import torch
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient, models

EMBED_MODEL = "BAAI/bge-m3"
QDRANT_URL = "http://localhost:6336"
QDRANT_COLLECTION = "rag_docs_hybrid"
DEFAULT_TOP_K = 5
PREVIEW_LEN = 300


def load_embedder(model_name: str = EMBED_MODEL) -> BGEM3FlagModel:
    """Локальный снапшот, если он уже в кэше, иначе — загрузка по имени."""
    try:
        model_path = snapshot_download(model_name, local_files_only=True)
    except Exception:
        model_path = model_name

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return BGEM3FlagModel(
        model_path,
        use_fp16=device.startswith("cuda"),
        devices=device,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )


def query_weights(embedder: BGEM3FlagModel, query: str) -> dict[int, float]:
    out = embedder.encode(
        [query], return_dense=False, return_sparse=True, return_colbert_vecs=False)
    weights = out["lexical_weights"][0]
    return {int(k): float(v) for k, v in weights.items() if float(v) > 0}


def build_filter(source: str | None, chunk_type: str | None) -> models.Filter | None:
    conditions = []
    if source:
        conditions.append(models.FieldCondition(
            key="source", match=models.MatchValue(value=source)))
    if chunk_type:
        conditions.append(models.FieldCondition(
            key="unit_type", match=models.MatchValue(value=chunk_type)))
    return models.Filter(must=conditions) if conditions else None


def preview(text: str, max_len: int = PREVIEW_LEN) -> str:
    flat = text.replace("\n", " ")
    return flat if len(flat) <= max_len else flat[:max_len] + "..."


def token_text(embedder: BGEM3FlagModel, token_id: int) -> str:
    token = embedder.tokenizer.convert_ids_to_tokens(token_id)
    return token.replace("\u2581", " ").strip() or token


def explain_hits(
    embedder: BGEM3FlagModel,
    query: dict[int, float],
    chunk_vector,
) -> str:
    """Наибольшие произведения q_w * c_w по общим token_id."""
    if not isinstance(chunk_vector, models.SparseVector):
        return "—"
    chunk = dict(zip(chunk_vector.indices, chunk_vector.values))
    hits = [
        (weight * chunk[token_id], token_id, weight, chunk[token_id])
        for token_id, weight in query.items()
        if token_id in chunk
    ]
    if not hits:
        return "—"
    hits.sort(reverse=True)
    return ", ".join(
        f"«{token_text(embedder, token_id)}» {qw:.2f}×{cw:.2f}={score:.2f}"
        for score, token_id, qw, cw in hits[:5]
    )


def search(
    query:      str,
    client:     QdrantClient,
    embedder:   BGEM3FlagModel,
    collection: str,
    top_k:      int,
    source:     str | None,
    chunk_type: str | None,
    explain:    bool,
) -> None:
    weights = query_weights(embedder, query)
    if not weights:
        print(f"В запросе «{query}» нет токенов для лексического поиска.")
        return

    sparse = models.SparseVector(
        indices=list(weights.keys()), values=list(weights.values()))

    response = client.query_points(
        collection_name=collection,
        query=sparse,
        using="sparse",
        query_filter=build_filter(source, chunk_type),
        limit=top_k,
        with_payload=True,
        with_vectors=True,
    )

    results = response.points
    if not results:
        print(f"По запросу «{query}» ничего не найдено.")
        return

    print(f"\nЗапрос  : «{query}»")
    print(f"Токенов : {len(weights)}")
    if source or chunk_type:
        active = []
        if source:
            active.append(f"source={source}")
        if chunk_type:
            active.append(f"type={chunk_type}")
        print(f"Фильтры : {', '.join(active)}")
    print(f"Найдено : {len(results)} результатов\n")
    print("═" * 80)

    for rank, point in enumerate(results, 1):
        meta = point.payload or {}
        text = meta.get("text", "")

        print(f"\n  #{rank}  sparse-score={point.score:.4f}")
        print(f"  документ : {meta.get('source', '—')}")
        print(f"  раздел   : {meta.get('section', '—')}")
        print(f"  тип      : {meta.get('unit_type', meta.get('chunk_type', '—'))}")
        print(f"  длина    : {len(text)} символов")

        if explain:
            chunk_vector = (point.vector or {}).get("sparse")
            print(f"  совпадения: {explain_hits(embedder, weights, chunk_vector)}")

        print("  текст    :")
        print(f"    {preview(text)}")
        print("─" * 80)

    if not source:
        print("\nРаспределение по документам:")
        doc_hits: dict[str, list[float]] = defaultdict(list)
        for point in results:
            meta = point.payload or {}
            doc_hits[meta.get("source", "unknown")].append(float(point.score))
        for src in sorted(doc_hits, key=lambda s: max(doc_hits[s]), reverse=True):
            scores = doc_hits[src]
            print(f"  {src:<50}  чанков: {len(scores)}  max: {max(scores):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse (BGE-M3) поиск по Qdrant")
    parser.add_argument("query", help="Поисковый запрос (ключевые слова)")
    parser.add_argument("--top",      type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--source",   default=None,
                        help="Фильтр по имени документа")
    parser.add_argument("--type",     default=None,
                        choices=["glossary", "clause", "table", "paragraph"],
                        help="Фильтр по unit_type")
    parser.add_argument("--explain",  action="store_true",
                        help="Показать совпавшие токены запроса и их вклад")
    parser.add_argument("--qdrant_url", default=QDRANT_URL)
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--model", default=EMBED_MODEL)
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url)
    embedder = load_embedder(args.model)

    search(
        query=args.query,
        client=client,
        embedder=embedder,
        collection=args.collection,
        top_k=args.top,
        source=args.source,
        chunk_type=args.type,
        explain=args.explain,
    )


if __name__ == "__main__":
    main()
