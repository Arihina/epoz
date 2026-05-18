"""
Индексирует результаты парсинга MinerU в ChromaDB.
Использует оба источника из каждой папки:
  - *model.json / *content_list_v2.json / *content_list.json — структурный парсинг
  - *.md — полная конвертация документа

Чанки из обоих источников попадают в одну коллекцию;
поле source_format=json|md позволяет различать их при поиске.

Структура входных данных:
    <input_dir>/
        <doc_name>/
            office/
                *model.json            ← приоритетный JSON-источник
                *content_list_v2.json  ← запасной
                *content_list.json     ← запасной (плоский)
                *.md                   ← MD-источник

Использование:
    pip install chromadb sentence-transformers

    python create_collection_hybrid.py \
        --input_dir ./mineru_output \
        --chroma_dir ./chroma_db \
        [--collection rag_docs_v2] \
        [--model all-MiniLM-L6-v2] \
        [--chunk_size 800] \
        [--chunk_overlap 80] \
        [--reset]
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Generator

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80

# all-MiniLM-L6-v2 не требует префиксов
PASSAGE_PREFIX = ""

Block = dict  # {"type": str, "level": int, "text": str, "anchor": str}


def _find_by_suffix(directory: Path, suffix: str) -> "Path | None":
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.name.endswith(suffix):
            return f
    return None


def _find_by_extension(directory: Path, ext: str) -> "Path | None":
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() == ext:
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


# JSON-парсеры 

def _strip_xml_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _extract_text_model(item: dict) -> str:
    raw = item.get("content", "")
    return _strip_xml_tags(raw) if isinstance(raw, str) else ""


def _extract_text_v2(item: dict) -> str:
    itype = item.get("type", "")
    content = item.get("content", {})
    if itype == "paragraph":
        parts = content.get("paragraph_content", [])
    elif itype == "title":
        parts = content.get("title_content", [])
    elif itype == "table":
        rows = content.get("table_content", [])
        cells = []
        for row in rows:
            for cell in row:
                for part in cell.get("paragraph_content", []):
                    cells.append(part.get("content", "").strip())
        return " | ".join(c for c in cells if c)
    else:
        return ""
    return "".join(p.get("content", "") for p in parts).strip()


def _extract_text_flat(item: dict) -> str:
    return item.get("text", "").strip()


def parse_model_json(path: Path) -> list[Block]:
    data, blocks = json.loads(path.read_text(encoding="utf-8")), []
    for page in data:
        if not isinstance(page, list):
            continue
        for item in page:
            text = _extract_text_model(item)
            if text:
                blocks.append({"type": item.get("type", "text"),
                               "level": item.get("level", 0),
                               "text": text, "anchor": item.get("anchor", "")})
    return blocks


def parse_content_list_v2(path: Path) -> list[Block]:
    data, blocks = json.loads(path.read_text(encoding="utf-8")), []
    for page in data:
        if not isinstance(page, list):
            continue
        for item in page:
            text = _extract_text_v2(item)
            if not text:
                continue
            itype = item.get("type", "text")
            level = item.get("content", {}).get(
                "level", 0) if itype == "title" else 0
            blocks.append({"type": itype, "level": level,
                           "text": text, "anchor": item.get("anchor", "")})
    return blocks


def parse_content_list_flat(path: Path) -> list[Block]:
    data, blocks = json.loads(path.read_text(encoding="utf-8")), []
    for item in data:
        text = _extract_text_flat(item)
        if text:
            blocks.append({"type": item.get("type", "text"), "level": 0,
                           "text": text, "anchor": item.get("anchor", "")})
    return blocks


def load_json_blocks(office_dir: Path) -> "tuple[list[Block], str] | tuple[None, None]":
    """Возвращает (blocks, filename) или (None, None) если ничего не найдено."""
    for suffix, parser in [
        ("model.json",           parse_model_json),
        ("content_list_v2.json", parse_content_list_v2),
        ("content_list.json",    parse_content_list_flat),
    ]:
        found = _find_by_suffix(office_dir, suffix)
        if found:
            return parser(found), found.name
    return None, None

# MD-парсер

_MD_HEADING = re.compile(r'^(#{1,10})\s+(.+)', re.MULTILINE)

_MD_TABLE_ROW = re.compile(r'^\|.+\|', re.MULTILINE)

_MD_TABLE_SEP = re.compile(r'^\|[-| :]+\|', re.MULTILINE)


def _parse_md_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        if _MD_TABLE_SEP.match(line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(" | ".join(c for c in cells if c))
    return "\n".join(rows)


_MD_HEADING = re.compile(r'^(#{1,10})\s+(.+)', re.MULTILINE)
_MD_TABLE_ROW = re.compile(r'^\|.+\|', re.MULTILINE)
_MD_TABLE_SEP = re.compile(r'^\|[-| :]+\|', re.MULTILINE)

_HTML_TABLE_OPEN  = re.compile(r'<table[\s>]', re.IGNORECASE)
_HTML_TABLE_CLOSE = re.compile(r'</table>', re.IGNORECASE)


def _parse_html_table(html: str) -> str:
    def cell_text(cell_html: str) -> str:
        text = re.sub(r'<br\s*/?>', ' ', cell_html, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    rows_text = []
    for tr in re.split(r'<tr[\s>]', html, flags=re.IGNORECASE):
        cells = re.findall(
            r'<t[dh][^>]*>(.*?)</t[dh]>',
            tr, flags=re.IGNORECASE | re.DOTALL
        )
        row_parts = [cell_text(c) for c in cells if cell_text(c)]
        if row_parts:
            rows_text.append(' | '.join(row_parts))

    return '\n'.join(rows_text)


def _parse_md_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        if _MD_TABLE_SEP.match(line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(" | ".join(c for c in cells if c))
    return "\n".join(rows)


def parse_md_blocks(path: Path) -> list[Block]:
    """
    Парсит MD-файл в список блоков.
    Стратегия:
      - <table>...</table> (HTML) → block type=table
      - строки '# ...' → block type=title, level=глубина '#'
      - блоки таблиц (строки начинающиеся с '|') → block type=table
      - всё остальное → block type=text
    """
    text = path.read_text(encoding="utf-8")

    html_tables: list[str] = []
    PLACEHOLDER = "\x00TABLE{}\x00"

    def _replace_table(m: re.Match) -> str:
        html_tables.append(m.group(0))
        return PLACEHOLDER.format(len(html_tables) - 1) + "\n"

    text_no_tables = re.sub(
        r'<table[\s>].*?</table>',
        _replace_table,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    lines = text_no_tables.splitlines()
    blocks: list[Block] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        ph_match = re.match(r'\x00TABLE(\d+)\x00', line)
        if ph_match:
            idx = int(ph_match.group(1))
            table_text = _parse_html_table(html_tables[idx])
            if table_text.strip():
                blocks.append({"type": "table", "level": 0,
                               "text": table_text, "anchor": ""})
            i += 1
            continue

        m = _MD_HEADING.match(line)
        if m:
            level = len(m.group(1))
            heading = re.sub(r'[*_`]', '', m.group(2)).strip()
            if heading:
                blocks.append({"type": "title", "level": level,
                               "text": heading, "anchor": ""})
            i += 1
            continue

        if _MD_TABLE_ROW.match(line.strip()):
            table_lines = []
            while i < len(lines) and (
                _MD_TABLE_ROW.match(lines[i].strip()) or
                _MD_TABLE_SEP.match(lines[i].strip())
            ):
                table_lines.append(lines[i])
                i += 1
            table_text = _parse_md_table(table_lines)
            if table_text.strip():
                blocks.append({"type": "table", "level": 0,
                               "text": table_text, "anchor": ""})
            continue

        para_lines = []
        while i < len(lines):
            l = lines[i]
            if (_MD_HEADING.match(l) or _MD_TABLE_ROW.match(l.strip())
                    or re.match(r'\x00TABLE\d+\x00', l)):
                break
            stripped = re.sub(r'[*_`]', '', l.strip())
            stripped = re.sub(r'!\[.*?\]\(.*?\)', '', stripped)
            stripped = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', stripped)
            if stripped:
                para_lines.append(stripped)
            elif para_lines:
                break
            i += 1

        para_text = " ".join(para_lines).strip()
        if para_text:
            blocks.append({"type": "text", "level": 0,
                           "text": para_text, "anchor": ""})

    return blocks


# Иерархический чанкинг (общий для JSON и MD)

def build_chunks(
    blocks:       list[Block],
    chunk_size:   int,
    chunk_overlap: int,
    source_format: str,   # "json" | "md"
) -> Generator[dict, None, None]:
    breadcrumb:   dict[int, str] = {}
    buffer_text = ""
    buffer_anchor = ""

    def flush(btext: str, banchor: str) -> "dict | None":
        btext = btext.strip()
        if not btext:
            return None
        section = " > ".join(v for _, v in sorted(breadcrumb.items()) if v)
        return {"text": btext, "section": section, "anchor": banchor,
                "chunk_type": "paragraph", "source_format": source_format}

    for block in blocks:
        btype = block["type"]
        level = block["level"]
        text = block["text"]
        anchor = block["anchor"]

        if btype == "title":
            if buffer_text:
                result = flush(buffer_text, buffer_anchor)
                if result:
                    yield result
                buffer_text, buffer_anchor = "", ""

            breadcrumb = {k: v for k, v in breadcrumb.items() if k < level}
            breadcrumb[level] = text
            section = " > ".join(v for _, v in sorted(breadcrumb.items()) if v)
            yield {"text": text, "section": section, "anchor": anchor,
                   "chunk_type": "heading", "source_format": source_format}

        elif btype == "table":
            if buffer_text:
                result = flush(buffer_text, buffer_anchor)
                if result:
                    yield result
                buffer_text, buffer_anchor = "", ""

            section = " > ".join(v for _, v in sorted(breadcrumb.items()) if v)
            yield {"text": text, "section": section, "anchor": anchor,
                   "chunk_type": "table", "source_format": source_format}

        else:
            if not buffer_anchor and anchor:
                buffer_anchor = anchor

            if len(buffer_text) + len(text) + 1 > chunk_size and buffer_text:
                result = flush(buffer_text, buffer_anchor)
                if result:
                    for part in _split_long_text(result["text"], chunk_size, chunk_overlap):
                        yield {**result, "text": part}
                buffer_text = (
                    buffer_text[-chunk_overlap:] + " " + text) if chunk_overlap else text
                buffer_anchor = anchor
            else:
                buffer_text = (buffer_text + "\n" + text).strip()

    if buffer_text:
        result = flush(buffer_text, buffer_anchor)
        if result:
            for part in _split_long_text(result["text"], chunk_size, chunk_overlap):
                yield {**result, "text": part}


def _add_to_collection(
    collection,
    chunks:      list[dict],
    source_name: str,
    embedder:    SentenceTransformer,
) -> int:
    if not chunks:
        return 0

    texts = [PASSAGE_PREFIX + c["text"] for c in chunks]
    metadatas = [
        {
            "source":        source_name,
            "section":       c["section"],
            "anchor":        c["anchor"],
            "chunk_type":    c["chunk_type"],
            "source_format": c["source_format"],
        }
        for c in chunks
    ]
    ids = [str(uuid.uuid4()) for _ in chunks]

    embeddings = []
    for i in range(0, len(texts), 64):
        batch = texts[i: i + 64]
        embeddings.extend(
            embedder.encode(batch, normalize_embeddings=True).tolist()
        )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=metadatas,
    )
    return len(chunks)


def index_documents(
    input_dir:     Path,
    chroma_dir:    Path,
    collection_name: str,
    model_name:    str,
    chunk_size:    int,
    chunk_overlap: int,
    reset:         bool,
) -> None:
    print(f"Загрузка модели: {model_name}")
    embedder = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    if reset and collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
        print(f"Коллекция '{collection_name}' удалена (--reset)")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    doc_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / "office").exists()
    ])

    if not doc_dirs:
        print(f"Не найдено папок с подпапкой 'office' в {input_dir}")
        sys.exit(1)

    print(f"Найдено документов: {len(doc_dirs)}\n")

    total_json = 0
    total_md = 0

    for doc_dir in doc_dirs:
        office_dir = doc_dir / "office"
        source_name = doc_dir.name
        print(f"[{source_name}]")

        # JSON
        json_blocks, json_fname = load_json_blocks(office_dir)
        if json_blocks:
            print(f"  JSON → источник: {json_fname}")
            json_chunks = list(build_chunks(
                json_blocks, chunk_size, chunk_overlap, "json"))
            added = _add_to_collection(
                collection, json_chunks, source_name, embedder)
            total_json += added
            print(f"  JSON → чанков добавлено: {added}")
        else:
            print(f"  JSON → не найден, пропускаем")

        # MD
        md_file = _find_by_extension(office_dir, ".md")
        if md_file:
            print(f"  MD   → источник: {md_file.name}")
            md_blocks = parse_md_blocks(md_file)
            md_chunks = list(build_chunks(
                md_blocks, chunk_size, chunk_overlap, "md"))
            added = _add_to_collection(
                collection, md_chunks, source_name, embedder)
            total_md += added
            print(f"  MD   → чанков добавлено: {added}")
        else:
            print(f"  MD   → не найден, пропускаем")

        print()

    total = total_json + total_md
    print(f"Готово.")
    print(f"  JSON-чанков : {total_json}")
    print(f"  MD-чанков   : {total_md}")
    print(f"  Итого        : {total} в коллекции '{collection_name}'")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексация MinerU (JSON + MD) → ChromaDB")
    parser.add_argument("--input_dir",     required=True,  type=Path)
    parser.add_argument("--chroma_dir",    required=True,  type=Path)
    parser.add_argument("--collection",    default="rag_docs_hibrid")
    parser.add_argument("--model",         default=DEFAULT_MODEL)
    parser.add_argument("--chunk_size",    type=int,
                        default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int,
                        default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--reset",         action="store_true",
                        help="Удалить коллекцию перед индексацией")

    args = parser.parse_args()
    index_documents(
        input_dir=args.input_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
