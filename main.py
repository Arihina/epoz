import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, Body
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import ollama
import uvicorn
from rank_bm25 import BM25Okapi
import chromadb


embed_model = SentenceTransformer("all-MiniLM-L6-v2")

ollama_client = ollama.Client(host="http://localhost:11434")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("docs")

documents = []
metadatas = []

try:
    all_results = collection.get(include=["documents", "metadatas"])
    
    if all_results and all_results['documents']:
        documents = all_results['documents']
        metadatas = all_results['metadatas']
        print(f"Загружено {len(documents)} документов из ChromaDB")
    else:
        print("Коллекция пуста")
        
except Exception as e:
    print(f"Ошибка при загрузке документов из ChromaDB: {e}")
    documents = []
    metadatas = []

chat_history = []

if documents:
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    print("BM25 индекс создан")
else:
    bm25 = None
    print("Предупреждение: нет документов для создания BM25 индекса")

RETRIEVAL_THRESHOLD = 0.4

SMALL_TALK_PATTERNS = [
    "привет",
    "здравств",
    "как тебя",
    "кто ты",
    "кто я",
    "как меня зовут",
    "меня зовут",
    "hello",
    "hi"
]


def is_small_talk(text):
    text = text.lower()
    return any(p in text for p in SMALL_TALK_PATTERNS)


def format_chat_history(history, max_turns=10):
    history = history[-max_turns * 2:]
    formatted = ""
    for role, text in history:
        if role == "user":
            formatted += f"Пользователь: {text}\n"
        else:
            formatted += f"Ассистент: {text}\n"
    return formatted.strip()


def build_general_prompt(history, user_question):
    chat = format_chat_history(history)

    return f"""
    Ты русскоязычный AI ассистент.
    Отвечай ТОЛЬКО на русском языке.
    Это общий вопрос, не связанный с документами.
    Отвечай свободно, как обычный ассистент.
    Источники указывать НЕ НУЖНО.

    Предыдущий диалог:
    {chat}

    Вопрос пользователя:
    {user_question}

    Ответ:
    """.strip()


def build_rag_prompt(retrieved_docs, history, user_question):
    context_blocks = []

    for i, doc in enumerate(retrieved_docs, 1):
        context_blocks.append(
            f"[Источник {i}]\n{doc['text']}"
        )

    context = "\n\n".join(context_blocks)
    chat = format_chat_history(history)

    return f"""
    Ты русскоязычный AI ассистент.
    Отвечай ТОЛЬКО на русском языке.
    Используй ТОЛЬКО ту информацию из контекста,
    которая действительно нужна для ответа.
    Если информация не использовалась — НЕ УПОМИНАЙ источник.
    
    Контекст:
    {context}
    
    Предыдущий диалог:
    {chat}
    
    Вопрос пользователя:
    {user_question}
    
    Ответ:
    """.strip()


def bm25_search(query, top_k=3):
    if bm25 is None or not documents:
        return []
        
    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in ranked_indices:
        score = float(scores[idx])

        if score <= 0:
            continue

        results.append({
            "text": documents[idx],
            "source": metadatas[idx].get("source", "unknown") if idx < len(metadatas) else "unknown",
            "chunk_id": metadatas[idx].get("chunk_id", idx) if idx < len(metadatas) else idx,
            "score": score
        })

    return results


def retrieve_docs(query, top_k=3):
    try:
        query_embedding = embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        vector_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = float(results['distances'][0][i])
                score = 1 - distance
                
                if score >= RETRIEVAL_THRESHOLD:
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    
                    vector_results.append({
                        "text": results['documents'][0][i],
                        "source": metadata.get("source", "unknown"),
                        "chunk_id": metadata.get("chunk_id", i),
                        "score": score
                    })
    except Exception as e:
        print(f"ChromaDB query error: {e}")
        vector_results = []

    bm25_results = bm25_search(query, top_k=top_k)

    combined = vector_results + bm25_results

    unique = {}

    for r in combined:
        key = (r["source"], r["chunk_id"])

        if key not in unique or unique[key]["score"] < r["score"]:
            unique[key] = r

    results = list(unique.values())

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def find_used_sources(answer, retrieved_docs):
    used = set()
    answer_lower = answer.lower()

    for doc in retrieved_docs:
        words = doc["text"].lower().split()
        overlap = sum(1 for w in words if w in answer_lower)

        if overlap > 5:
            used.add(doc["source"])

    return used


def rag_ollama_answer(user_question, chat_history):
    if is_small_talk(user_question):
        prompt = build_general_prompt(chat_history, user_question)
    else:
        retrieved = retrieve_docs(user_question, top_k=3)
        if not retrieved or retrieved[0]["score"] < 0.5:
            prompt = build_general_prompt(chat_history, user_question)
        else:
            prompt = build_rag_prompt(retrieved, chat_history, user_question)

    response = ollama_client.chat(
        model='gemma2:2b',
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response['message']['content'].strip()

    if not is_small_talk(user_question) and retrieved:
        used_sources = find_used_sources(answer, retrieved)
        if used_sources:
            answer += "\n\nИсточники:\n" + \
                "\n".join(f"- {s}" for s in used_sources)

    chat_history.append(("user", user_question))
    chat_history.append(("assistant", answer))

    return answer, chat_history

app = FastAPI()
app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

origins = ['http://localhost:5173', 'https://localhost:5173']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/gif",
    "image/tiff",
    "image/webp"
}


@app.post("/upload/text")
async def upload_text(item: dict = Body(...)):
    answer, _ = rag_ollama_answer(item['message'].replace("?", ""), chat_history)
    return {
        "sources": None,
        "answer": answer
    }


@app.get("/")
async def serve_frontend():
    return FileResponse("dist/index.html")


@app.get("/reset")
async def reset():
    global chat_history
    chat_history = []
    return FileResponse("dist/index.html")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8443,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
        reload=True
    )