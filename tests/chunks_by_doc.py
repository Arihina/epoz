"""
Просмотр чанков по документам в Qdrant.

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

from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6336"
QDRANT_COLLECTION = "rag_docs_hybrid"
PREVIEW_LEN = 200


def _load(client):
    records = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(batch)
        if offset is None:
            break
    return [(r.payload or {}, (r.payload or {}).get("text", ""))
            for r in records]


def list_sources(client) -> None:
    items = _load(client)
    if not items:
        print("Коллекция пуста.")
        return

    stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "heading": 0, "paragraph": 0, "table": 0})
    for meta, _ in items:
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


def show_chunks(client, source: str, chunk_type: str | None, full: bool) -> None:
    items = _load(client)
    docs = [(meta, text) for meta, text in items
            if meta.get("source") == source
            and (chunk_type is None or meta.get("chunk_type") == chunk_type)]

    if not docs:
        print(f"Чанков не найдено для source='{source}'" +
              (f", chunk_type='{chunk_type}'" if chunk_type else ""))
        return

    grouped: dict[str, list] = defaultdict(list)
    for meta, text in docs:
        grouped[meta.get("chunk_type", "unknown")].append((text, meta))

    total = len(docs)
    print(f"\nДокумент: {source}  |  чанков: {total}" +
          (f"  |  фильтр: {chunk_type}" if chunk_type else ""))
    print("═" * 80)

    type_order = ["heading", "paragraph", "table"]
    all_types = type_order + [t for t in grouped if t not in type_order]

    for ctype in all_types:
        if ctype not in grouped:
            continue
        items_of_type = grouped[ctype]
        print(f"\n[{ctype.upper()}] — {len(items_of_type)} шт.")
        print("─" * 80)

        for i, (text, meta) in enumerate(items_of_type, 1):
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
                for line in text.splitlines():
                    print(f"    {line}")
            else:
                preview = text.replace("\n", " ")[:PREVIEW_LEN]
                suffix = "..." if len(text) > PREVIEW_LEN else ""
                print(f"    {preview}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Просмотр чанков в Qdrant")
    parser.add_argument("--source", default=None,
                        help="Имя документа (значение поля source в payload)")
    parser.add_argument("--type",   default=None,
                        choices=["heading", "paragraph", "table"],
                        help="Фильтр по типу чанка")
    parser.add_argument("--full",   action="store_true",
                        help="Показывать полный текст без обрезки")
    parser.add_argument("--qdrant_url", default=QDRANT_URL)
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url)

    if args.source is None:
        list_sources(client)
    else:
        show_chunks(client, args.source, args.type, args.full)


if __name__ == "__main__":
    main()
