"""
Индексирует результаты парсинга MinerU в Qdrant по новой схеме разбиения.

Источник (по приоритету):
  - *content_list_v2.json  ← структурный парсинг (основной)
  - *model.json            ← запасной
  - *content_list.json     ← запасной (плоский)

Единицы хранения (unit_type):
  - glossary — аббревиатуры и термины («НМЦ – начальная максимальная цена…»),
               с отдельным полем term для точного поиска;
  - clause   — нумерованные пункты норм и обычные абзацы;
  - table    — строки таблиц (с повторённой шапкой).

Заголовки разделов не индексируются отдельно — попадают в поле section.
Оглавление («СОДЕРЖАНИЕ») и колонтитулы/номера страниц отбрасываются.

Коллекция Qdrant хранит два именованных вектора:
  - dense  — 1024-мерный эмбеддинг BGE-M3 (cosine);
  - sparse — лексические веса BGE-M3 (token_id -> вес) для гибридного поиска.

Использование:
    pip install qdrant-client FlagEmbedding beautifulsoup4

    python create_collection_hybrid.py \
        --input_dir ./EPoZ_data \
        --qdrant_url http://localhost:6336 \
        [--collection rag_docs_hybrid] \
        [--model BAAI/bge-m3] \
        [--reset]
"""

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

from bs4 import BeautifulSoup
from qdrant_client import QdrantClient, models

from app.core.embedder import (
    DENSE_DIM,
    DENSE_VECTOR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    SPARSE_VECTOR,
    BgeM3Embedder,
)


DEFAULT_MODEL = "BAAI/bge-m3"

# BGE-M3 не требует префиксов query:/passage: — эмбеддим текст как есть.
PASSAGE_PREFIX = ""

_DEFAULT_QDRANT_URL = (
    f"http://{os.getenv('QDRANT_HOST', 'localhost')}"
    f":{os.getenv('QDRANT_PORT', '6336')}"
)

# Семантика title-уровней MinerU: 1..3 — заголовки разделов,
# 4..7 — пункты норм, 10 — колонтитулы/номера страниц.
HEADING_MAX_LEVEL = 3
CLAUSE_MIN_LEVEL = 4
NOISE_LEVEL = 10

TOC_TITLE = "СОДЕРЖАНИЕ"
GLOSSARY_MARKERS = ("СОКРАЩЕНИЯ", "ТЕРМИН")

# Точек в одном upsert — чтобы не превысить лимит размера запроса Qdrant (32 МБ).
UPLOAD_BATCH_SIZE = 256


def _find_by_suffix(directory: Path, suffix: str) -> "Path | None":
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.name.endswith(suffix):
            return f
    return None


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > size and current:
            chunks.append(current.strip())
            current = (current[-overlap:] + " " + sent) if overlap else sent
        else:
            current = (current + " " + sent).strip()
    if current:
        chunks.append(current.strip())
    return chunks or [text]


def _strip_xml_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


# Таблицы

def _table_grid_from_html(html: str) -> list[list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []
    grid = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            grid.append(cells)
    return grid


def _table_grid_from_content(content: dict) -> list[list[str]]:
    grid = []
    for row in content.get("table_content", []):
        cells = []
        for cell in row:
            txt = "".join(p.get("content", "")
                          for p in cell.get("paragraph_content", []))
            cells.append(txt.strip())
        grid.append([c for c in cells if c])
    return grid


def _with_section(section: str, text: str) -> str:
    """Дописать путь раздела в текст чанка, чтобы заголовки участвовали в поиске."""
    return f"{section}\n{text}".strip() if section else text


def _doc_title(blocks: list[dict]) -> str:
    """Заголовок приложения: первый содержательный абзац после титульной страницы."""
    markers = ("Приложение", "Единому Положению", "Государственной корпорации")
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b["kind"] == "text" and any(m in b.get("text", "") for m in markers):
            i += 1
            continue
        break
    for b in blocks[i:]:
        if b["kind"] in ("table", "list"):
            break
        if b["kind"] in ("text", "title"):
            t = (b.get("text") or "").strip()
            if t:
                return t
    return ""


def _doc_prefix(source_name: str, title: str = "") -> str:
    """Тема приложения (имя папки + заголовок) — дописываем её в текст чанка."""
    if not source_name.startswith("Приложение"):
        return ""
    prefix = source_name.replace("_", " ").strip()
    if title and title.lower() not in prefix.lower():
        prefix = f"{prefix} {title}".strip()
    return prefix + "\n"


def _table_units(grid: list[list[str]], section: str, anchor: str) -> list[dict]:
    if not grid:
        return []
    if len(grid) == 1:
        return [{"text": _with_section(section, " | ".join(grid[0])),
                 "section": section, "anchor": anchor, "unit_type": "table"}]
    header = " | ".join(grid[0])
    units = []
    for row in grid[1:]:
        units.append({"text": _with_section(section, header + "\n" + " | ".join(row)),
                      "section": section, "anchor": anchor, "unit_type": "table"})
    return units


# Парсеры JSON 

def _v2_title_text(content: dict) -> str:
    return "".join(p.get("content", "")
                   for p in content.get("title_content", [])).strip()


def _v2_paragraph_text(content: dict) -> str:
    return "".join(p.get("content", "")
                   for p in content.get("paragraph_content", [])).strip()


def _v2_list_items(content: dict) -> list[str]:
    out = []
    for li in content.get("list_items", []):
        parts = li.get("item_content", [])
        text = " ".join((p.get("content") or "").strip()
                        for p in parts if (p.get("content") or "").strip())
        text = text.strip()
        if text:
            out.append(text)
    return out


def parse_content_list_v2(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks: list[dict] = []
    for page in data:
        if not isinstance(page, list):
            continue
        for item in page:
            itype = item.get("type", "")
            content = item.get("content", {})
            anchor = item.get("anchor", "")
            if itype == "title":
                text = _v2_title_text(content)
                if text:
                    blocks.append({"kind": "title", "level": content.get("level", 0),
                                   "text": text, "anchor": anchor})
            elif itype == "paragraph":
                text = _v2_paragraph_text(content)
                if text:
                    blocks.append({"kind": "text", "text": text, "anchor": anchor})
            elif itype == "table":
                grid = (_table_grid_from_html(content.get("html", ""))
                        or _table_grid_from_content(content))
                if grid:
                    blocks.append({"kind": "table", "grid": grid, "anchor": anchor})
            elif itype == "list":
                items = _v2_list_items(content)
                if items:
                    blocks.append({"kind": "list", "items": items})
    return blocks


def parse_model_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks: list[dict] = []
    for page in data:
        if not isinstance(page, list):
            continue
        for item in page:
            itype = item.get("type", "")
            anchor = item.get("anchor", "")
            if itype == "title":
                text = _strip_xml_tags(item.get("content", ""))
                if text:
                    blocks.append({"kind": "title", "level": item.get("level", 0),
                                   "text": text, "anchor": anchor})
            elif itype in ("text", "paragraph"):
                text = _strip_xml_tags(item.get("content", ""))
                if text:
                    blocks.append({"kind": "text", "text": text, "anchor": anchor})
            elif itype == "table":
                grid = _table_grid_from_html(item.get("content", ""))
                if grid:
                    blocks.append({"kind": "table", "grid": grid, "anchor": anchor})
    return blocks


def parse_content_list_flat(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks: list[dict] = []
    for item in data:
        text = (item.get("text") or "").strip()
        if text:
            blocks.append({"kind": "text", "text": text,
                           "anchor": item.get("anchor", "")})
    return blocks


def load_json_blocks(office_dir: Path) -> "tuple[list[dict], str] | tuple[None, None]":
    for suffix, parser in [
        ("content_list_v2.json", parse_content_list_v2),
        ("model.json",           parse_model_json),
        ("content_list.json",    parse_content_list_flat),
    ]:
        found = _find_by_suffix(office_dir, suffix)
        if found:
            return parser(found), found.name
    return None, None


# Сборка юнитов 

def _is_page_noise(text: str, level: int) -> bool:
    if level != NOISE_LEVEL:
        return False
    stripped = text.strip()
    if re.fullmatch(r"\d+", stripped):       # номер страницы
        return True
    return bool(re.search(r"\t+\d+$", stripped))  # «Заголовок\t\t9»


def _split_term(text: str) -> "tuple[str, str]":
    for sep in ("–", "—", "-"):
        if sep in text:
            term, _, definition = text.partition(sep)
            term, definition = term.strip(), definition.strip()
            if term and definition and len(term.split()) <= 6:
                return term, definition
    return "", text


def _is_glossary_list(items: list[str]) -> bool:
    if not items:
        return False
    with_term = sum(1 for t in items if _split_term(t)[0])
    return with_term >= max(1, len(items) // 2)


def build_units(blocks: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    breadcrumb: dict[int, str] = {}
    clause_breadcrumb: dict[int, str] = {}
    skip_toc = False
    glossary_mode = False
    buffer_text = ""
    buffer_anchor = ""
    units: list[dict] = []

    def section() -> str:
        return " > ".join(v for _, v in sorted(breadcrumb.items()) if v)

    def flush() -> None:
        nonlocal buffer_text, buffer_anchor
        text = buffer_text.strip()
        if text:
            sec = section()
            for part in _split_long_text(text, chunk_size, chunk_overlap):
                units.append({"text": _with_section(sec, part), "section": sec,
                              "anchor": buffer_anchor, "unit_type": "clause"})
        buffer_text, buffer_anchor = "", ""

    def add_to_buffer(text: str, anchor: str = "") -> None:
        nonlocal buffer_text, buffer_anchor
        buffer_text = (buffer_text + "\n" + text).strip() if buffer_text else text
        if not buffer_anchor and anchor:
            buffer_anchor = anchor
        if len(buffer_text) > chunk_size:
            flush()

    for blk in blocks:
        kind = blk["kind"]

        if skip_toc:
            if kind == "title":
                level = blk["level"]
                text = blk["text"].strip()
                if (not _is_page_noise(text, level)
                        and level <= HEADING_MAX_LEVEL
                        and text.upper() != TOC_TITLE):
                    skip_toc = False
                else:
                    continue
            else:
                continue

        if kind == "title":
            level = blk["level"]
            text = blk["text"].strip()
            if _is_page_noise(text, level):
                continue
            if text.upper() == TOC_TITLE and level <= HEADING_MAX_LEVEL:
                skip_toc = True
                continue
            if level <= HEADING_MAX_LEVEL:
                flush()
                breadcrumb = {k: v for k, v in breadcrumb.items() if k < level}
                breadcrumb[level] = text
                clause_breadcrumb = {}   # новый раздел — сброс цепочки пунктов
                glossary_mode = text.upper().startswith(GLOSSARY_MARKERS)
            elif level >= CLAUSE_MIN_LEVEL:
                flush()
                context = " ".join(clause_breadcrumb[k]
                                   for k in sorted(clause_breadcrumb) if k < level)
                full_text = f"{context} {text}".strip() if context else text
                clause_breadcrumb = {k: v for k, v in clause_breadcrumb.items() if k < level}
                clause_breadcrumb[level] = text
                sec = section()
                for part in _split_long_text(full_text, chunk_size, chunk_overlap):
                    units.append({"text": _with_section(sec, part), "section": sec,
                                  "anchor": blk.get("anchor", ""),
                                  "unit_type": "clause"})

        elif kind == "text":
            add_to_buffer(blk["text"], blk.get("anchor", ""))

        elif kind == "table":
            flush()
            units.extend(_table_units(blk["grid"], section(), blk.get("anchor", "")))

        elif kind == "list":
            if glossary_mode and _is_glossary_list(blk["items"]):
                flush()
                for item_text in blk["items"]:
                    term, _ = _split_term(item_text)
                    units.append({"text": item_text, "term": term,
                                  "section": section(), "anchor": "",
                                  "unit_type": "glossary"})
            else:
                for item_text in blk["items"]:
                    add_to_buffer(item_text)

    flush()
    return units


# Запись в Qdrant 

def _add_to_collection(
    client,
    collection_name: str,
    units:        list[dict],
    source_name:  str,
    model_name:   str,
    embedder:     BgeM3Embedder,
    title:        str = "",
) -> int:
    if not units:
        return 0

    prefix = _doc_prefix(source_name, title)
    texts = [PASSAGE_PREFIX + prefix + u["text"] for u in units]
    dense_vecs, sparse_vecs = embedder.encode(texts)

    points = []
    for u, dense, sparse in zip(units, dense_vecs, sparse_vecs):
        payload = {
            "text":        u["text"],
            "source":      source_name,
            "section":     u.get("section", ""),
            "anchor":      u.get("anchor", ""),
            "unit_type":   u.get("unit_type", "clause"),
            "embed_model": model_name,
        }
        if u.get("term"):
            payload["term"] = u["term"]

        vector: dict = {DENSE_VECTOR: dense.tolist()}
        if sparse:
            vector[SPARSE_VECTOR] = models.SparseVector(
                indices=list(sparse.keys()),
                values=list(sparse.values()),
            )

        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        ))

    for i in range(0, len(points), UPLOAD_BATCH_SIZE):
        client.upsert(
            collection_name=collection_name,
            points=points[i:i + UPLOAD_BATCH_SIZE],
        )
    return len(units)


def _check_embed_model(client, collection_name: str, expected_model: str) -> None:
    """Проверить, что коллекция проиндексирована той же моделью."""
    try:
        records, _ = client.scroll(
            collection_name=collection_name, limit=1,
            with_payload=True, with_vectors=False,
        )
    except Exception:
        return
    if not records:
        return
    actual = (records[0].payload or {}).get("embed_model")
    if actual and str(actual).strip().lower() != expected_model.strip().lower():
        sys.exit(
            f"Коллекция '{collection_name}' проиндексирована моделью '{actual}', "
            f"а сейчас используется '{expected_model}'. Пересоздайте: --reset"
        )


def _ensure_collection(client, collection_name: str, model_name: str) -> None:
    """Создать коллекцию dense+sparse или проверить, что схема и модель совпадают."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=DENSE_DIM,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
        )
        print(f"Коллекция '{collection_name}' создана (dense + sparse)")
        return

    params = client.get_collection(collection_name).config.params
    vectors = params.vectors if isinstance(params.vectors, dict) else {}
    sparse_vectors = params.sparse_vectors or {}
    dense_size = getattr(vectors.get(DENSE_VECTOR), "size", None)

    if dense_size != DENSE_DIM or SPARSE_VECTOR not in sparse_vectors:
        sys.exit(
            f"Схема коллекции '{collection_name}' не соответствует dense+sparse "
            f"(dense: {dense_size}, sparse: {sorted(sparse_vectors)}). "
            f"Пересоздайте коллекцию: --reset"
        )

    _check_embed_model(client, collection_name, model_name)


def index_documents(
    input_dir:      Path,
    qdrant_url:     str,
    collection_name: str,
    model_name:     str,
    chunk_size:     int,
    chunk_overlap:  int,
    reset:          bool,
    batch_size:     int = DEFAULT_BATCH_SIZE,
    max_length:     int = DEFAULT_MAX_LENGTH,
    local_files_only: bool = False,
) -> None:
    print(f"Загрузка модели: {model_name}")
    embedder = BgeM3Embedder(
        model_name,
        local_files_only=local_files_only,
        batch_size=batch_size,
        max_length=max_length,
    )
    print(f"  → устройство: {embedder.device}, "
          f"батч: {batch_size}, max_length: {max_length}")

    client = QdrantClient(url=qdrant_url)

    if reset and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Коллекция '{collection_name}' удалена (--reset)")

    _ensure_collection(client, collection_name, model_name)

    doc_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / "office").exists()
    ])

    if not doc_dirs:
        print(f"Не найдено папок с подпапкой 'office' в {input_dir}")
        sys.exit(1)

    print(f"Найдено документов: {len(doc_dirs)}\n")

    total = 0

    for doc_dir in doc_dirs:
        office_dir = doc_dir / "office"
        source_name = doc_dir.name
        print(f"[{source_name}]")

        blocks, fname = load_json_blocks(office_dir)
        if not blocks:
            print("  → JSON не найден, пропускаем")
            print()
            continue

        print(f"  → источник: {fname}")
        units = build_units(blocks, chunk_size, chunk_overlap)
        title = _doc_title(blocks)
        added = _add_to_collection(
            client, collection_name, units, source_name, model_name, embedder, title)
        total += added
        print(f"  → юнитов добавлено: {added}")
        print()

    print(f"Готово. Всего юнитов в '{collection_name}': {total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексация MinerU (JSON) → Qdrant")
    parser.add_argument("--input_dir",     required=True,  type=Path)
    parser.add_argument("--qdrant_url",    default=_DEFAULT_QDRANT_URL)
    parser.add_argument("--collection",    default="rag_docs_hybrid")
    parser.add_argument("--model",         default=DEFAULT_MODEL)
    parser.add_argument("--chunk_size",    type=int,
                        default=800)
    parser.add_argument("--chunk_overlap", type=int,
                        default=80)
    parser.add_argument("--batch_size",    type=int, default=DEFAULT_BATCH_SIZE,
                        help="Текстов на один проход модели")
    parser.add_argument("--max_length",    type=int, default=DEFAULT_MAX_LENGTH,
                        help="Максимум токенов на текст (BGE-M3: до 8192)")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Не обращаться к HF Hub, брать модель из кэша")
    parser.add_argument("--reset",         action="store_true",
                        help="Удалить коллекцию перед индексацией")

    args = parser.parse_args()
    index_documents(
        input_dir=args.input_dir,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
