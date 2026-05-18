"""
search_vector.py
────────────────
Семантический (векторный) поиск по базе знаний.

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

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "../chroma_db"
CHROMA_COLLECTION = "rag_docs_hybrid"
EMBED_MODEL = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.0    # без порога по умолчанию — показываем всё
PREVIEW_LEN = 300


def search(
    query:      str,
    collection,
    embedder:   SentenceTransformer,
    top_k:      int,
    threshold:  float,
    source:     str | None,
    chunk_type: str | None,
) -> None:
    embedding = embedder.encode(
        [QUERY_PREFIX + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].tolist()

    filters = []
    if source:
        filters.append({"source": source})
    if chunk_type:
        filters.append({"chunk_type": chunk_type})

    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    kwargs = dict(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    if not raw["documents"] or not raw["documents"][0]:
        print("Ничего не найдено.")
        return

    results = []
    for i, doc in enumerate(raw["documents"][0]):
        score = 1.0 - float(raw["distances"][0][i])
        if score < threshold:
            continue
        meta = raw["metadatas"][0][i] or {}
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
    parser = argparse.ArgumentParser(description="Векторный поиск по ChromaDB")
    parser.add_argument("query",                help="Поисковый запрос")
    parser.add_argument("--top",       type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Минимальный cosine-score (0..1)")
    parser.add_argument("--source",    default=None,
                        help="Фильтр по имени документа")
    parser.add_argument("--type",      default=None,
                        choices=["heading", "paragraph", "table"],
                        help="Фильтр по типу чанка")
    parser.add_argument("--chroma_dir",  default=CHROMA_PATH)
    parser.add_argument("--collection",  default=CHROMA_COLLECTION)
    args = parser.parse_args()

    print(f"Загрузка модели {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.collection)

    search(
        query=args.query,
        collection=collection,
        embedder=embedder,
        top_k=args.top,
        threshold=args.threshold,
        source=args.source,
        chunk_type=args.type,
    )


if __name__ == "__main__":
    main()
