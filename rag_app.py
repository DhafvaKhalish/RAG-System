from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from fastapi import FastAPI, Request
import nest_asyncio
import uvicorn
import uuid

nest_asyncio.apply()

client = QdrantClient(":memory:")
COLLECTION_NAME = "rag_docs"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

def add_documents(documents):
    vectors = embedding_model.encode(documents).tolist()
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": doc})
        for vec, doc in zip(vectors, documents)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

# Contoh dokumen
documents = [
    "Indonesia adalah negara kepulauan terbesar di dunia.",
    "Python adalah bahasa pemrograman populer untuk data science.",
    "Qdrant adalah vektor database untuk pencarian semantik."
]
add_documents(documents)

llm = pipeline("text-generation", model="gpt2")

def rag_answer(question: str):
    question_vec = embedding_model.encode([question])[0].tolist()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vec,
        limit=3
    )
    context = "\n".join([hit.payload["text"] for hit in hits])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    output = llm(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]
    return output.split("Answer:")[-1].strip()

app = FastAPI()

@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = body.get("question", "")
    answer = rag_answer(question)
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
