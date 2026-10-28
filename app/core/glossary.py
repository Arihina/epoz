"""Глоссарий (термины и аббревиатуры) для роутинга.

Извлекает unit_type=glossary из Qdrant в память и даёт:
  - lookup() — точное определение термина, если он явно назван в вопросе
    («что такое НМЦ», «расшифруй ЕИС» или просто «НМЦ»);
  - expand() — замена коротких аббревиатур на расшифровки перед поиском
    («какой срок на НМЦ» → «какой срок на начальная (максимальная) цена»).
"""

from __future__ import annotations

import re

from qdrant_client import QdrantClient, models

_GLOSSARY_UNIT_TYPE = "glossary"

_LOOKUP_RE = re.compile(
    r"(?:что такое|кто такой|кто такие|расшифруй|расшифровка|что значит|что означает|дай определение)\s+(.+)",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", s.lower())


def _definition(text: str) -> str:
    for sep in ("–", "—", "-"):
        if sep in text:
            _, _, d = text.partition(sep)
            d = d.strip()
            if d:
                return d
    return text.strip()


def _is_abbreviation(term: str) -> bool:
    """Короткий однословный код: НМЦ, ЕИС, ЗК, ГОЗ (но не «Заказчик»)."""
    t = term.strip()
    if not t or " " in t or len(t) > 12:
        return False
    body = re.sub(r"[\W_]+", "", t)
    return bool(body) and (body.isupper() or any(c.isdigit() for c in body))


class Glossary:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.terms: dict[str, dict] = {}          # norm_term -> entry
        self.abbreviations: dict[str, str] = {}   # norm_term -> definition
        self._load(client, collection)

    def _load(self, client: QdrantClient, collection: str) -> None:
        try:
            offset = None
            while True:
                records, offset = client.scroll(
                    collection_name=collection,
                    limit=512,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=models.Filter(must=[
                        models.FieldCondition(
                            key="unit_type",
                            match=models.MatchValue(value=_GLOSSARY_UNIT_TYPE),
                        ),
                    ]),
                )
                for rec in records:
                    payload = rec.payload or {}
                    term = (payload.get("term") or "").strip()
                    text = payload.get("text") or ""
                    if not term or not text:
                        continue
                    key = _norm(term)
                    if not key or key in self.terms:
                        continue
                    entry = {
                        "term":       term,
                        "definition": _definition(text),
                        "text":       text,
                        "source":     payload.get("source", ""),
                        "section":    payload.get("section", ""),
                    }
                    self.terms[key] = entry
                    if _is_abbreviation(term):
                        self.abbreviations[key] = entry["definition"] or text
                if offset is None:
                    break
        except Exception as exc:
            print(f"[glossary] Ошибка загрузки глоссария: {exc}")

    def lookup(self, query: str) -> list[dict]:
        """0 или 1 определение термина, если он явно назван в вопросе."""
        m = _LOOKUP_RE.search(query)
        if m:
            entry = self.terms.get(_norm(m.group(1)))
            if entry:
                return [self._to_result(entry)]

        # весь запрос — это и есть термин/аббревиатура («НМЦ», «закупочная комиссия»)
        entry = self.terms.get(_norm(query))
        return [self._to_result(entry)] if entry else []

    def expand(self, query: str) -> str:
        """Заменить известные аббревиатуры на расшифровки."""
        out = query
        for key in sorted(self.abbreviations, key=lambda k: -len(self.terms[k]["term"])):
            term = self.terms[key]["term"]
            out = re.sub(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                self.abbreviations[key],
                out,
                flags=re.IGNORECASE,
            )
        return out

    @staticmethod
    def _to_result(entry: dict) -> dict:
        return {
            "text":     entry["text"],
            "source":   entry["source"],
            "chunk_id": f"glossary:{_norm(entry['term'])}",
            "score":    1.0,
            "section":  entry["section"],
        }
