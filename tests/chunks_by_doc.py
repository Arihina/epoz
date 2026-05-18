"""
Просмотр чанков по документам в ChromaDB.

Использование:
    # список всех документов
    python chunks_by_doc.py

    # все чанки конкретного документа
    python chunks_by_doc.py --source "doc_name"

    # только определённый тип чанков
    python chunks_by_doc.py --source "doc_name" --type heading
    python chunks_by_doc.py --source "doc_name" --type paragraph
    python chunks_by_doc.py --source "doc_name" --type table

    # показать полный текст чанков (без обрезки)
    python chunks_by_doc.py --source "doc_name" --full
"""

import argparse
from collections import defaultdict

import chromadb

CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "rag_docs_hybrid"
PREVIEW_LEN = 200


def list_sources(collection) -> None:
    result = collection.get(include=["metadatas"])
    if not result["metadatas"]:
        print("Коллекция пуста.")
        return

    stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "heading": 0, "paragraph": 0, "table": 0})
    for meta in result["metadatas"]:
        src = meta.get("source", "unknown")
        ctype = meta.get("chunk_type", "unknown")
        stats[src]["total"] += 1
        if ctype in stats[src]:
            stats[src][ctype] += 1

    print(
        f"\n{'Документ':<50} {'Всего':>6} {'heading':>8} {'paragraph':>10} {'table':>6}")
    print("─" * 84)
    for src in sorted(stats):
        s = stats[src]
        print(
            f"{src:<50} {s['total']:>6} {s['heading']:>8} {s['paragraph']:>10} {s['table']:>6}")
    print(f"\nИтого документов: {len(stats)}")


def show_chunks(collection, source: str, chunk_type: str | None, full: bool) -> None:
    where = {"source": source}
    if chunk_type:
        where = {"$and": [{"source": source}, {"chunk_type": chunk_type}]}

    result = collection.get(
        where=where,
        include=["documents", "metadatas"],
    )

    if not result["documents"]:
        print(f"Чанков не найдено для source='{source}'" +
              (f", chunk_type='{chunk_type}'" if chunk_type else ""))
        return

    docs = result["documents"]
    metadatas = result["metadatas"]

    # группируем по chunk_type для удобного вывода
    grouped: dict[str, list] = defaultdict(list)
    for doc, meta in zip(docs, metadatas):
        grouped[meta.get("chunk_type", "unknown")].append((doc, meta))

    total = len(docs)
    print(f"\nДокумент: {source}  |  чанков: {total}" +
          (f"  |  фильтр: {chunk_type}" if chunk_type else ""))
    print("═" * 80)

    type_order = ["heading", "paragraph", "table"]
    all_types = type_order + [t for t in grouped if t not in type_order]

    for ctype in all_types:
        if ctype not in grouped:
            continue
        items = grouped[ctype]
        print(f"\n[{ctype.upper()}] — {len(items)} шт.")
        print("─" * 80)

        for i, (text, meta) in enumerate(items, 1):
            section = meta.get("section", "")
            anchor = meta.get("anchor", "")

            print(f"\n  #{i}")
            if section:
                print(f"  section : {section}")
            if anchor:
                print(f"  anchor  : {anchor}")
            print(f"  длина   : {len(text)} символов")
            print(f"  текст   :")

            if full:
                # с отступом для читаемости
                for line in text.splitlines():
                    print(f"    {line}")
            else:
                preview = text.replace("\n", " ")[:PREVIEW_LEN]
                suffix = "..." if len(text) > PREVIEW_LEN else ""
                print(f"    {preview}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Просмотр чанков в ChromaDB")
    parser.add_argument("--source", default=None,
                        help="Имя документа (значение поля source в метаданных)")
    parser.add_argument("--type",   default=None,
                        choices=["heading", "paragraph", "table"],
                        help="Фильтр по типу чанка")
    parser.add_argument("--full",   action="store_true",
                        help="Показывать полный текст без обрезки")
    parser.add_argument("--chroma_dir",  default=CHROMA_PATH)
    parser.add_argument("--collection",  default=CHROMA_COLLECTION)
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.collection)

    if args.source is None:
        list_sources(collection)
    else:
        show_chunks(collection, args.source, args.type, args.full)


if __name__ == "__main__":
    main()
