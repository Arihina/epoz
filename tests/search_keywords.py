"""
Поиск по ключевым словам (BM25) по базе знаний.

Использование:
    python search_keywords.py "закупочная комиссия состав"

    # больше результатов
    python search_keywords.py "председатель комиссии" --top 10

    # фильтр по документу
    python search_keywords.py "председатель" --source "doc_name"

    # фильтр по типу чанка
    python search_keywords.py "председатель" --type table

    # показать все слова запроса и их BM25-веса в найденных чанках
    python search_keywords.py "состав комиссии" --explain
"""

import argparse
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
import chromadb

CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "rag_docs_hybrid"
DEFAULT_TOP_K = 5
PREVIEW_LEN = 300


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def highlight(text: str, query_tokens: set[str], max_len: int = PREVIEW_LEN) -> str:
    """Обрезает текст и помечает совпадения символами >>..<<."""
    flat = text.replace("\n", " ")
    if len(flat) > max_len:
        best_start = 0
        best_hits = 0
        step = max_len // 2
        for start in range(0, max(1, len(flat) - max_len), step):
            window = flat[start: start + max_len]
            hits = sum(1 for t in query_tokens if t in window.lower())
            if hits > best_hits:
                best_hits = hits
                best_start = start
        flat = "..." + flat[best_start: best_start + max_len] + "..."

    words = flat.split()
    marked = []
    for w in words:
        clean = re.sub(r"[^\w]", "", w.lower())
        if clean in query_tokens:
            marked.append(f">>{w}<<")
        else:
            marked.append(w)
    return " ".join(marked)


def explain_scores(doc: str, query_tokens: list[str], bm25: BM25Okapi, doc_idx: int) -> None:
    """Выводит вклад каждого токена запроса в итоговый score."""
    print("  вклад токенов:")
    doc_tokens = tokenize(doc)
    for token in query_tokens:
        tf = doc_tokens.count(token)
        print(f"    «{token}» — встречается {tf} раз")


def search(
    query:      str,
    collection,
    top_k:      int,
    source:     str | None,
    chunk_type: str | None,
    explain:    bool,
) -> None:
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

    get_kwargs = dict(include=["documents", "metadatas"])
    if where:
        get_kwargs["where"] = where

    result = collection.get(**get_kwargs)

    if not result["documents"]:
        print("Коллекция пуста или фильтр не дал результатов.")
        return

    documents = result["documents"]
    metadatas = result["metadatas"]

    tokenized = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append((score, idx, documents[idx], metadatas[idx]))

    if not results:
        print(f"По запросу «{query}» ничего не найдено.")
        return

    query_set = set(query_tokens)

    print(f"\nЗапрос  : «{query}»")
    print(f"Токены  : {query_tokens}")
    if source or chunk_type:
        active = []
        if source:
            active.append(f"source={source}")
        if chunk_type:
            active.append(f"type={chunk_type}")
        print(f"Фильтры : {', '.join(active)}")
    print(f"Найдено : {len(results)} результатов\n")
    print("═" * 80)

    for rank, (score, idx, doc, meta) in enumerate(results, 1):
        source_val = meta.get("source", "—")
        section_val = meta.get("section", "—")
        ctype_val = meta.get("chunk_type", "—")

        print(f"\n  #{rank}  BM25-score={score:.4f}")
        print(f"  документ : {source_val}")
        print(f"  раздел   : {section_val}")
        print(f"  тип      : {ctype_val}")
        print(f"  длина    : {len(doc)} символов")

        if explain:
            explain_scores(doc, query_tokens, bm25, idx)

        print(f"  текст    :")
        print(f"    {highlight(doc, query_set)}")
        print("─" * 80)

    # статистика по документам
    if not source:
        print("\nРаспределение по документам:")
        doc_hits: dict[str, list[float]] = defaultdict(list)
        for score, _, _, meta in results:
            doc_hits[meta.get("source", "unknown")].append(score)
        for src in sorted(doc_hits, key=lambda s: max(doc_hits[s]), reverse=True):
            scores_list = doc_hits[src]
            print(
                f"  {src:<50}  чанков: {len(scores_list)}  max: {max(scores_list):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 поиск по ChromaDB")
    parser.add_argument(
        "query",               help="Поисковый запрос (ключевые слова)")
    parser.add_argument("--top",      type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--source",   default=None,
                        help="Фильтр по имени документа")
    parser.add_argument("--type",     default=None,
                        choices=["heading", "paragraph", "table"],
                        help="Фильтр по типу чанка")
    parser.add_argument("--explain",  action="store_true",
                        help="Показать вклад каждого токена запроса")
    parser.add_argument("--chroma_dir",  default=CHROMA_PATH)
    parser.add_argument("--collection",  default=CHROMA_COLLECTION)
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.collection)

    search(
        query=args.query,
        collection=collection,
        top_k=args.top,
        source=args.source,
        chunk_type=args.type,
        explain=args.explain,
    )


if __name__ == "__main__":
    main()
