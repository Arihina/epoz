import pypandoc
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from sentence_transformers import SentenceTransformer
import chromadb
import os
from datetime import datetime

def docx_to_md(input_file, output_file):
    output = pypandoc.convert_file(
        input_file,
        'md',
        format='docx',
        extra_args=[
            '--wrap=none',
            '--extract-media=media'
        ]
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

def split_with_headers(md_text):
    splitter = get_splitter()
    
    sections = re.split(r'(?=\n# )', md_text)
    all_chunks = []

    for section in sections:
        header_match = re.match(r'(#+ .+)', section)
        header = header_match.group(1) if header_match else ""

        chunks = splitter.split_text(section)
        
        for chunk in chunks:
            enriched_chunk = f"{header}\n{chunk}"
            all_chunks.append(enriched_chunk)

    return all_chunks

def save_chunks_info(chunks, filename="chunks_info.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ИНФОРМАЦИЯ О ЧАНКАХ ДОКУМЕНТА\n")
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Всего чанков: {len(chunks)}\n")
        f.write(f"Источник: input.docx\n")
        f.write(f"Модель эмбеддингов: all-MiniLM-L6-v2\n")
        f.write(f"Размер чанка: 800 символов (перекрытие 150)\n")
        f.write("\n" + "="*80 + "\n\n")
        
        
        f.write("ПОДРОБНАЯ ИНФОРМАЦИЯ ПО КАЖДОМУ ЧАНКУ:\n")
        f.write("="*80 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"ЧАНК #{i+1}\n")
            f.write("-"*80 + "\n")
            f.write(f"ID: chunk_{i}\n")
            f.write(f"Длина: {len(chunk)} символов\n")
            f.write(f"Количество строк: {len(chunk.split(chr(10)))}\n")
            
            headers = re.findall(r'#+\s+.+', chunk)
            if headers:
                f.write(f"Заголовки: {', '.join(headers[:3])}\n")

            numbered_items = re.findall(r'\d+\)', chunk)
            if numbered_items:
                unique_items = list(set(numbered_items))
                f.write(f"Нумерованные пункты: {', '.join(unique_items[:5])}\n")
            
            nested_items = re.findall(r'\(\d+\)', chunk)
            if nested_items:
                unique_nested = list(set(nested_items))
                f.write(f"Вложенная нумерация: {', '.join(unique_nested[:5])}\n")
            
            bullet_items = re.findall(r'^\s*[-*•]\s+.+', chunk, re.MULTILINE)
            if bullet_items:
                f.write(f"Маркированные списки: {len(bullet_items)} элементов\n")
            
            f.write("\nТЕКСТ ЧАНКА:\n")
            f.write("-"*80 + "\n")
            f.write(chunk)
            f.write("\n\n" + "-"*80 + "\n")
            f.write(f"КОНЕЦ ЧАНКА #{i+1}\n")
            f.write("="*80 + "\n\n")

        f.write("\n\n" + "="*80 + "\n")
        f.write("СПИСОК ЧАНКОВ (ДЛЯ БЫСТРОГО ПОИСКА):\n")
        f.write("="*80 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            preview = chunk[:150].replace('\n', ' ')
            f.write(f"{i+1}. {preview}...\n")
    
    print(f"✓ Информация о чанках сохранена в {filename}")

def save_chunks_summary(chunks, filename="chunks_summary.txt"):
    """Сохраняет краткую сводку о чанках"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("КРАТКАЯ СВОДКА ПО ЧАНКАМ\n")
        f.write("="*80 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            preview = chunk[:100].replace('\n', ' ').strip()
            if len(chunk) > 100:
                preview += "..."
            
            chunk_type = "Обычный"
            if re.search(r'#+\s+', chunk):
                chunk_type = "С заголовком"
            if re.search(r'\d+\)', chunk):
                chunk_type = "С нумерацией"
            if re.search(r'\(\d+\)', chunk):
                chunk_type = "С вложенной нумерацией"
            
            f.write(f"{i+1:3d}. [{chunk_type:20}] {preview}\n")
    
    print(f"✓ Краткая сводка сохранена в {filename}")

print("="*80)
print("ОБРАБОТКА ДОКУМЕНТА")
print("="*80)

print("1. Конвертация DOCX в MD...")
print("...Пропущено...")
# docx_to_md("input.docx", "output.md")

print("2. Чтение MD файла...")
with open("output_clean.md", "r", encoding="utf-8") as f:
    md_text = f.read()

print("3. Разбивка на чанки...")
chunks = split_with_headers(md_text)
print(f"   Получено {len(chunks)} чанков")

print("4. Сохранение информации о чанках...")
save_chunks_info(chunks, "chunks_info.txt")
save_chunks_summary(chunks, "chunks_summary.txt")

print("5. Загрузка модели эмбеддингов...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("6. Создание эмбеддингов...")
embeddings = model.encode(chunks, show_progress_bar=True)

print("7. Создание постоянного хранилища ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("docs")
    print("   Старая коллекция удалена")
except:
    pass

collection = client.create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"}
)

print("8. Добавление документов в коллекцию...")
ids = [f"chunk_{i}" for i in range(len(chunks))]

metadatas = []
for i, chunk in enumerate(chunks):
    metadata = {
        "chunk_id": i,
        "source": "input.docx",
        "length": len(chunk),
        "has_headers": bool(re.search(r'#+\s+', chunk)),
        "has_numbering": bool(re.search(r'\d+\)', chunk)),
        "has_nested": bool(re.search(r'\(\d+\)', chunk))
    }
    metadatas.append(metadata)

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=ids,
    metadatas=metadatas
)

print("\n" + "="*80)
print("✓ ГОТОВО!")
print("="*80)
print(f"   Добавлено {len(chunks)} документов в коллекцию 'docs'")
print(f"   База данных сохранена в папке ./chroma_db")
print("\n   СОЗДАННЫЕ ФАЙЛЫ:")
print(f"   - chunks_info.txt     (подробная информация о каждом чанке)")
print(f"   - chunks_summary.txt  (краткая сводка по всем чанкам)")
print(f"   - output.md           (сконвертированный Markdown файл)")
print("\n   МЕТАДАННЫЕ В CHROMADB:")
print(f"   - chunk_id: номер чанка")
print(f"   - source: input.docx")
print(f"   - length: длина чанка в символах")
print(f"   - has_headers: наличие заголовков")
print(f"   - has_numbering: наличие нумерации (1), 2) и т.д.)")
print(f"   - has_nested: наличие вложенной нумерации ((1), (2) и т.д.)")
print("="*80)

print("\nСТАТИСТИКА ПО ЧАНКАМ:")
print(f"  Всего: {len(chunks)}")
print(f"  С заголовками: {sum(1 for c in chunks if re.search(r'#+\s+', c))}")
print(f"  С нумерацией: {sum(1 for c in chunks if re.search(r'\d+\)', c))}")
print(f"  С вложенной нумерацией: {sum(1 for c in chunks if re.search(r'\(\d+\)', c))}")
print(f"  Средняя длина: {sum(len(c) for c in chunks) // len(chunks)} символов")
print("="*80)
