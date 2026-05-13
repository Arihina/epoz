"""
Индексирует результаты парсинга MinerU в ChromaDB.

Структура входных данных:
    <input_dir>/
        <doc_name>/
            office/
                *model.json            ← приоритетный источник
                *content_list_v2.json  ← запасной
                *content_list.json     ← запасной (плоский)

Использование:
    pip install chromadb sentence-transformers

    python index_to_chroma.py \
        --input_dir ./mineru_output \
        --chroma_dir ./chroma_db \
        [--collection rag_docs] \
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

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80
PASSAGE_PREFIX = "passage: "

Block = dict  # {"type": str, "level": int, "text": str, "anchor": str}


def _find_by_suffix(directory: Path, suffix: str) -> "Path | None":
    """Возвращает первый файл, имя которого оканчивается на suffix."""
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.name.endswith(suffix):
            return f
    return None


def _strip_xml_tags(text: str) -> str:
    """Убирает <text style=...>...</text> из model.json."""
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


def parse_model_json(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = []
    for page in data:
        if not isinstance(page, list):
            continue
        for item in page:
            text = _extract_text_model(item)
            if not text:
                continue
            blocks.append({
                "type":   item.get("type", "text"),
                "level":  item.get("level", 0),
                "text":   text,
                "anchor": item.get("anchor", ""),
            })
    return blocks


def parse_content_list_v2(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = []
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
            blocks.append({
                "type":   itype,
                "level":  level,
                "text":   text,
                "anchor": item.get("anchor", ""),
            })
    return blocks


def parse_content_list_flat(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = []
    for item in data:
        text = _extract_text_flat(item)
        if not text:
            continue
        blocks.append({
            "type":   item.get("type", "text"),
            "level":  0,
            "text":   text,
            "anchor": item.get("anchor", ""),
        })
    return blocks


def load_blocks(office_dir: Path) -> list:
    """
    Ищет файлы по суффиксу (имя может содержать произвольный префикс).
    Приоритет: *model.json → *content_list_v2.json → *content_list.json
    """
    candidates = [
        ("model.json",           parse_model_json),
        ("content_list_v2.json", parse_content_list_v2),
        ("content_list.json",    parse_content_list_flat),
    ]
    for suffix, parser in candidates:
        found = _find_by_suffix(office_dir, suffix)
        if found:
            print(f"  → источник: {found.name}")
            return parser(found)

    raise FileNotFoundError(f"Не найден подходящий JSON в {office_dir}")


def _split_long_text(text: str, size: int, overlap: int) -> list:
    if len(text) <= size:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""

    for sent in sentences:
        if len(current) + len(sent) + 1 > size and current:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + sent if overlap else sent
        else:
            current = (current + " " + sent).strip()

    if current:
        chunks.append(current.strip())

    return chunks or [text]


def build_chunks(blocks: list, chunk_size: int, chunk_overlap: int) -> Generator:
    """
    - title  → обновляет breadcrumb, индексируется как chunk_type=heading
    - table  → отдельный чанк, chunk_type=table
    - text   → агрегируется в буфер до chunk_size, chunk_type=paragraph
    """
    breadcrumb = {}
    buffer_text = ""
    buffer_anchor = ""

    def flush(btext, banchor):
        btext = btext.strip()
        if not btext:
            return None
        section = " > ".join(v for _, v in sorted(breadcrumb.items()) if v)
        return {"text": btext, "section": section,
                "anchor": banchor, "chunk_type": "paragraph"}

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
            yield {"text": text, "section": section,
                   "anchor": anchor, "chunk_type": "heading"}

        elif btype == "table":
            if buffer_text:
                result = flush(buffer_text, buffer_anchor)
                if result:
                    yield result
                buffer_text, buffer_anchor = "", ""

            section = " > ".join(v for _, v in sorted(breadcrumb.items()) if v)
            yield {"text": text, "section": section,
                   "anchor": anchor, "chunk_type": "table"}

        else:
            if not buffer_anchor and anchor:
                buffer_anchor = anchor

            if len(buffer_text) + len(text) + 1 > chunk_size and buffer_text:
                result = flush(buffer_text, buffer_anchor)
                if result:
                    for part in _split_long_text(result["text"], chunk_size, chunk_overlap):
                        yield {**result, "text": part}
                buffer_text = (text[-chunk_overlap:] + " " +
                               text) if chunk_overlap else text
                buffer_anchor = anchor
            else:
                buffer_text = (buffer_text + "\n" + text).strip()

    if buffer_text:
        result = flush(buffer_text, buffer_anchor)
        if result:
            for part in _split_long_text(result["text"], chunk_size, chunk_overlap):
                yield {**result, "text": part}


def index_documents(
    input_dir: Path,
    chroma_dir: Path,
    collection_name: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    reset: bool,
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

    total_chunks = 0

    for doc_dir in doc_dirs:
        office_dir = doc_dir / "office"
        source_name = doc_dir.name
        print(f"[{source_name}]")

        try:
            blocks = load_blocks(office_dir)
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            continue

        chunks = list(build_chunks(blocks, chunk_size, chunk_overlap))
        print(f"  → чанков: {len(chunks)}")

        if not chunks:
            continue

        texts = [PASSAGE_PREFIX + c["text"] for c in chunks]
        metadatas = [
            {
                "source":     source_name,
                "section":    c["section"],
                "anchor":     c["anchor"],
                "chunk_type": c["chunk_type"],
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

        total_chunks += len(chunks)
        print(f"  ✓ добавлено в коллекцию")

    print(f"\nГотово. Всего чанков в '{collection_name}': {total_chunks}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексация MinerU → ChromaDB")
    parser.add_argument("--input_dir",     required=True, type=Path)
    parser.add_argument("--chroma_dir",    required=True, type=Path)
    parser.add_argument("--collection",    default="rag_docs")
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
