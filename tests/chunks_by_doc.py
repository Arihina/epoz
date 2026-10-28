"""
Просмотр юнитов по документам в Qdrant.

Использование:
    # список всех документов
    python chunks_by_doc.py

    # все юниты конкретного документа
    python chunks_by_doc.py --source "doc_name"

    # только определённый тип юнитов
    python chunks_by_doc.py --source "doc_name" --type glossary
    python chunks_by_doc.py --source "doc_name" --type clause
    python chunks_by_doc.py --source "doc_name" --type table

    # показать полный текст юнитов (без обрезки)
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
        lambda: {"total": 0, "glossary": 0, "clause": 0, "table": 0})
    for meta, _ in items:
        src = meta.get("source", "unknown")
        utype = meta.get("unit_type", meta.get("chunk_type", "unknown"))
        stats[src]["total"] += 1
        if utype in stats[src]:
            stats[src][utype] += 1

    print(
        f"\n{'Документ':<50} {'Всего':>6} {'glossary':>9} {'clause':>7} {'table':>6}")
    print("─" * 82)
    for src in sorted(stats):
        s = stats[src]
        print(
            f"{src:<50} {s['total']:>6} {s['glossary']:>9} {s['clause']:>7} {s['table']:>6}")
    print(f"\nИтого документов: {len(stats)}")


def show_chunks(client, source: str, unit_type: str | None, full: bool) -> None:
    items = _load(client)
    docs = [(meta, text) for meta, text in items
            if meta.get("source") == source
            and (unit_type is None
                 or meta.get("unit_type", meta.get("chunk_type")) == unit_type)]

    if not docs:
        print(f"Юнитов не найдено для source='{source}'" +
              (f", unit_type='{unit_type}'" if unit_type else ""))
        return

    grouped: dict[str, list] = defaultdict(list)
    for meta, text in docs:
        grouped[meta.get("unit_type", meta.get("chunk_type", "unknown"))].append(
            (text, meta))

    total = len(docs)
    print(f"\nДокумент: {source}  |  юнитов: {total}" +
          (f"  |  фильтр: {unit_type}" if unit_type else ""))
    print("═" * 80)

    type_order = ["glossary", "clause", "table"]
    all_types = type_order + [t for t in grouped if t not in type_order]

    for utype in all_types:
        if utype not in grouped:
            continue
        items_of_type = grouped[utype]
        print(f"\n[{utype.upper()}] — {len(items_of_type)} шт.")
        print("─" * 80)

        for i, (text, meta) in enumerate(items_of_type, 1):
            section = meta.get("section", "")
            anchor = meta.get("anchor", "")
            term = meta.get("term", "")

            print(f"\n  #{i}")
            if section:
                print(f"  section : {section}")
            if term:
                print(f"  term    : {term}")
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
    parser = argparse.ArgumentParser(description="Просмотр юнитов в Qdrant")
    parser.add_argument("--source", default=None,
                        help="Имя документа (значение поля source в payload)")
    parser.add_argument("--type",   default=None,
                        choices=["glossary", "clause", "table"],
                        help="Фильтр по unit_type")
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
