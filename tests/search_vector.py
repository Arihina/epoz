"""
Семантический (dense) поиск по базе знаний: dense-векторы BGE-M3 в Qdrant.

Использование:
    python search_vector.py "порядок создания закупочной комиссии"

    # больше результатов
    python search_vector.py "состав комиссии" --top 10

    # фильтр по документу
    python search_vector.py "состав комиссии" --source "doc_name"

    # фильтр по типу юнита
    python search_vector.py "состав комиссии" --type table

    # минимальный порог схожести (0..1)
    python search_vector.py "состав комиссии" --threshold 0.5
"""

import argparse

import torch
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient, models

EMBED_MODEL = "BAAI/bge-m3"
QDRANT_URL = "http://localhost:6336"
QDRANT_COLLECTION = "rag_docs_hybrid"
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.0    # без порога по умолчанию — показываем всё
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
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )


def query_embedding(embedder: BGEM3FlagModel, query: str) -> list[float]:
    out = embedder.encode(
        [query], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return out["dense_vecs"][0].tolist()


def build_filter(source: str | None, unit_type: str | None) -> models.Filter | None:
    conditions = []
    if source:
        conditions.append(models.FieldCondition(
            key="source", match=models.MatchValue(value=source)))
    if unit_type:
        conditions.append(models.FieldCondition(
            key="unit_type", match=models.MatchValue(value=unit_type)))
    return models.Filter(must=conditions) if conditions else None


def search(
    query:      str,
    client:     QdrantClient,
    embedder:   BGEM3FlagModel,
    collection: str,
    top_k:      int,
    threshold:  float,
    source:     str | None,
    unit_type:  str | None,
) -> None:
    embedding = query_embedding(embedder, query)

    response = client.query_points(
        collection_name=collection,
        query=embedding,
        using="dense",
        query_filter=build_filter(source, unit_type),
        limit=top_k,
        with_payload=True,
    )

    results = []
    for point in response.points:
        score = float(point.score)
        if score < threshold:
            continue
        meta = point.payload or {}
        results.append((score, meta.get("text", ""), meta))

    if not results:
        print(f"Нет результатов выше порога {threshold:.2f}.")
        return

    print(f"\nЗапрос : «{query}»")
    print(f"Найдено: {len(results)} результатов\n")
    print("═" * 80)

    for rank, (score, doc, meta) in enumerate(results, 1):
        print(f"\n  #{rank}  score={score:.4f}")
        print(f"  документ : {meta.get('source', '—')}")
        print(f"  раздел   : {meta.get('section', '—')}")
        print(f"  тип      : {meta.get('unit_type', '—')}")
        print(f"  длина    : {len(doc)} символов")
        print(f"  текст    :")

        preview = doc.replace("\n", " ")[:PREVIEW_LEN]
        suffix = "..." if len(doc) > PREVIEW_LEN else ""
        print(f"    {preview}{suffix}")
        print("─" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense (BGE-M3) поиск по Qdrant")
    parser.add_argument("query",                help="Поисковый запрос")
    parser.add_argument("--top",       type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Минимальный cosine-score (0..1)")
    parser.add_argument("--source",    default=None,
                        help="Фильтр по имени документа")
    parser.add_argument("--type",      default=None,
                        choices=["glossary", "clause", "table"],
                        help="Фильтр по unit_type")
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
        threshold=args.threshold,
        source=args.source,
        unit_type=args.type,
    )


if __name__ == "__main__":
    main()
