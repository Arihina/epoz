import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi

print("Загрузка модели...")
model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db") 
collection = client.get_collection("docs")

print(f"Коллекция загружена, документов: {collection.count()}")

print("Загрузка документов для BM25...")
all_docs = collection.get(include=["documents", "metadatas"])
documents = all_docs['documents']
metadatas = all_docs['metadatas'] if all_docs['metadatas'] else [{}] * len(documents)

tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
print("Готов к поиску!\n")

def hybrid_search(query):
    query_embedding = model.encode([query])[0].tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "distances"]
    )
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:10]
    
    combined = {}
    
    if vector_results['documents'] and vector_results['documents'][0]:
        for i, (doc, distance) in enumerate(zip(vector_results['documents'][0], vector_results['distances'][0])):
            vector_score = 1 - distance
            combined[i] = {"text": doc, "vector_score": vector_score, "bm25_score": 0}
    
    for idx in bm25_indices:
        if idx in combined:
            combined[idx]["bm25_score"] = bm25_scores[idx]
        else:
            combined[idx] = {"text": documents[idx], "vector_score": 0, "bm25_score": bm25_scores[idx]}
    
    max_vector = max([v["vector_score"] for v in combined.values()]) if combined else 1
    max_bm25 = max([v["bm25_score"] for v in combined.values()]) if combined else 1
    
    results = []
    for item in combined.values():
        norm_vector = item["vector_score"] / max_vector if max_vector > 0 else 0
        norm_bm25 = item["bm25_score"] / max_bm25 if max_bm25 > 0 else 0
        final_score = (norm_vector + norm_bm25) / 2
        
        results.append({
            "text": item["text"],
            "score": final_score
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]

while True:
    query = input("Введите запрос (или 'exit' для выхода): ").strip()
    
    if query.lower() == 'exit':
        print("До свидания!")
        break
    
    if not query:
        continue
    
    results = hybrid_search(query)
    
    print("\n" + "="*60)
    print(f"Результаты поиска: '{query}'")
    print("="*60)
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   {result['text'][:200]}..." if len(result['text']) > 200 else f"   {result['text']}")
    else:
        print("Ничего не найдено")
    
    print()
