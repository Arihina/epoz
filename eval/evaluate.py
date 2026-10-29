"""Оценка качества retrieval: recall@k, hit@k, mrr, nDCG@k.

Релевантность бинарная, на уровне документа (`source`) + опциональный фильтр
по `section` (подстрока пути раздела). Это и есть «рельсы» для калибровки
порогов/бустов (#10), reranker'а (#4) и query rewriting (#6).

Использование (из корня репозитория):
    python eval/evaluate.py --top 10

Требования: поднятый Qdrant с проиндексированной коллекцией и модель BGE-M3 в кэше.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _is_relevant(doc: dict, q: dict) -> bool:
    if doc.get("source") not in q["sources"]:
        return False
    section = q.get("section")
    if section and section.lower() not in (doc.get("section") or "").lower():
        return False
    return True


def _recall_at_k(docs: list[dict], q: dict) -> float:
    found = {d["source"] for d in docs if _is_relevant(d, q)}
    return len(found) / len(q["sources"]) if q["sources"] else 0.0


def _hit_at_k(docs: list[dict], q: dict) -> float:
    return 1.0 if any(_is_relevant(d, q) for d in docs) else 0.0


def _mrr(docs: list[dict], q: dict) -> float:
    for i, d in enumerate(docs, 1):
        if _is_relevant(d, q):
            return 1.0 / i
    return 0.0


def _ndcg_at_k(docs: list[dict], q: dict) -> float:
    # Бинарная релевантность с дедупликацией источника (первое вхождение).
    seen: set[str] = set()
    dcg = 0.0
    for i, d in enumerate(docs, 1):
        src = d.get("source", "")
        if _is_relevant(d, q) and src not in seen:
            seen.add(src)
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, len(q["sources"]) + 1))
    return dcg / ideal if ideal else 0.0


METRICS = {
    "recall@k": _recall_at_k,
    "hit@k":   _hit_at_k,
    "mrr":     _mrr,
    "ndcg@k":  _ndcg_at_k,
}


def evaluate(queries: list[dict], retrieve_fn, top_k: int = 10, debug: bool = False) -> tuple[dict, list[dict]]:
    """Прогнать запросы через retrieve_fn и посчитать метрики.

    Возвращает (сводка {метрика: среднее}, построчные результаты).
    При debug=True в строку дополнительно кладётся список извлечённых чанков.
    """
    summary = {name: [] for name in METRICS}
    rows: list[dict] = []
    for q in queries:
        docs = retrieve_fn(q["query"], top_k)
        row = {"query": q["query"], "retrieved": len(docs)}
        for name, fn in METRICS.items():
            value = fn(docs, q)
            row[name] = round(value, 4)
            summary[name].append(value)
        if debug:
            row["docs"] = docs
        rows.append(row)
    return {name: round(sum(v) / len(v), 4) for name, v in summary.items()}, rows


def print_report(summary: dict, rows: list[dict], debug: bool = False) -> None:
    print("\n=== Сводка (среднее) ===")
    for name, value in summary.items():
        print(f"  {name:9s} {value:.4f}")
    print("\n=== По запросам ===")
    for r in rows:
        metrics = " ".join(f"{m}={r[m]}" for m in METRICS)
        print(f"  [{r['retrieved']:2d}] {r['query'][:55]:55s} {metrics}")

    if debug:
        for r in rows:
            if r["recall@k"] >= 1.0:
                continue
            print(f"\n--- «{r['query']}» (recall={r['recall@k']}) ---")
            for i, d in enumerate(r.get("docs", [])[:5], 1):
                src = d.get("source", "")
                sec = (d.get("section") or "")[:45]
                txt = (d.get("text") or "").replace("\n", " ")
                print(f"  {i}. [{d.get('score', 0):.3f}] {src} | {sec:45s} | {txt[:70]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества retrieval")
    parser.add_argument("--golden", default=str(Path(__file__).parent / "golden.json"))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--debug", action="store_true",
                        help="Показать извлечённые чанки для запросов с recall<1")
    args = parser.parse_args()

    queries = json.loads(Path(args.golden).read_text(encoding="utf-8"))

    # Скрипт запускается как `python eval/evaluate.py` — при этом в sys.path
    # попадает каталог `eval/`, а не корень репозитория. Добавим корень явно,
    # чтобы импортировался пакет `app`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Ленивый импорт: нужны поднятый Qdrant и загруженная модель.
    from app.core.retrieval import retrieve

    summary, rows = evaluate(queries, retrieve, args.top, debug=args.debug)
    print_report(summary, rows, debug=args.debug)


if __name__ == "__main__":
    main()
