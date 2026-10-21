"""
search_vector.py
────────────────
Семантический (векторный) поиск по базе знаний (Qdrant).

Использование:
    python search_vector.py "порядок создания закупочной комиссии"

    # больше результатов
    python search_vector.py "состав комиссии" --top 10

    # фильтр по документу
    python search_vector.py "состав комиссии" --source "doc_name"

    # фильтр по типу чанка
    python search_vector.py "состав комиссии" --type paragraph

    # минимальный порог схожести (0..1)
    python search_vector.py "состав комиссии" --threshold 0.5
"""

import argparse

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6336"
QDRANT_COLLECTION = "rag_docs_hybrid"
EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.0    # без порога по умолчанию — показываем всё
PREVIEW_LEN = 300


def search(
    query:      str,
    client,
    embedder:   SentenceTransformer,
    top_k:      int,
    threshold:  float,
    source:     str | None,
    chunk_type: str | None,
) -> None:
    embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].tolist()

    conditions = []
    if source:
        conditions.append(models.FieldCondition(
            key="source", match=models.MatchValue(value=source)))
    if chunk_type:
        conditions.append(models.FieldCondition(
            key="chunk_type", match=models.MatchValue(value=chunk_type)))
    query_filter = models.Filter(must=conditions) if conditions else None

    raw = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=embedding,
        using="dense",
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    results = []
    for point in raw.points:
        score = float(point.score)
        if score < threshold:
            continue
        meta = point.payload or {}
        doc = meta.get("text", "")
        results.append((score, doc, meta))

    if not results:
        print(f"Нет результатов выше порога {threshold:.2f}.")
        return

    print(f"\nЗапрос : «{query}»")
    print(f"Найдено: {len(results)} результатов\n")
    print("═" * 80)

    for rank, (score, doc, meta) in enumerate(results, 1):
        source_val = meta.get("source", "—")
        section_val = meta.get("section", "—")
        ctype_val = meta.get("chunk_type", "—")
        anchor_val = meta.get("anchor", "")

        print(f"\n  #{rank}  score={score:.4f}")
        print(f"  документ : {source_val}")
        print(f"  раздел   : {section_val}")
        print(f"  тип      : {ctype_val}", end="")
        if anchor_val:
            print(f"  |  anchor: {anchor_val}", end="")
        print()
        print(f"  длина    : {len(doc)} символов")
        print(f"  текст    :")

        preview = doc.replace("\n", " ")[:PREVIEW_LEN]
        suffix = "..." if len(doc) > PREVIEW_LEN else ""
        print(f"    {preview}{suffix}")
        print("─" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Векторный поиск по Qdrant")
    parser.add_argument("query",                help="Поисковый запрос")
    parser.add_argument("--top",       type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Минимальный cosine-score (0..1)")
    parser.add_argument("--source",    default=None,
                        help="Фильтр по имени документа")
    parser.add_argument("--type",      default=None,
                        choices=["heading", "paragraph", "table"],
                        help="Фильтр по типу чанка")
    parser.add_argument("--qdrant_url", default=QDRANT_URL)
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    args = parser.parse_args()

    print(f"Загрузка модели {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url=args.qdrant_url)

    search(
        query=args.query,
        client=client,
        embedder=embedder,
        top_k=args.top,
        threshold=args.threshold,
        source=args.source,
        chunk_type=args.type,
    )


if __name__ == "__main__":
    main()
