"""CartVista support RAG: chunk -> embed -> retrieve -> generate.
   Every run is an MLflow experiment -- chunking is a hyperparameter."""
import argparse, pathlib, mlflow, ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim
LLM = "llama3.2:1b"

def chunk_docs(kb_dir, chunk_size, overlap):
    chunks = []
    for f in sorted(pathlib.Path(kb_dir).glob("*.md")):
        words = f.read_text(encoding='utf-8').split()
        step = chunk_size - overlap
        for i in range(0, max(len(words) - overlap, 1), step):
            chunks.append({"text": " ".join(words[i:i + chunk_size]),
                           "source": f.name})
    return chunks

def build_index(chunks, client):
    client.recreate_collection("cartvista_kb",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    vecs = EMBEDDER.encode([c["text"] for c in chunks])
    client.upsert("cartvista_kb", points=[
        PointStruct(id=i, vector=v.tolist(), payload=c)
        for i, (v, c) in enumerate(zip(vecs, chunks))])

def retrieve(query, client, top_k):
    qv = EMBEDDER.encode([query])[0]
    hits = client.search("cartvista_kb", query_vector=qv.tolist(),
                         limit=top_k)
    return [(h.payload, h.score) for h in hits]

PROMPT = """You are CartVista's support assistant.
Answer ONLY from the context below. If the context does not contain
the answer, say \"I don't have that information.\"

Context:
{context}

Question: {question}
Answer:"""

def answer(question, client, top_k):
    hits = retrieve(question, client, top_k)
    ctx = "\n---\n".join(
        f"[{p['source']}] {p['text']}" for p, _ in hits)
    resp = ollama.chat(model=LLM, messages=[{
        "role": "user",
        "content": PROMPT.format(context=ctx, question=question)}])
    return resp["message"]["content"], hits

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=60)  # words
    ap.add_argument("--overlap",    type=int, default=15)
    ap.add_argument("--top-k",      type=int, default=3)
    ap.add_argument("--question",   default="How long do I have to return milk that arrived spoiled?")
    a = ap.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cartvista-rag")

    client = QdrantClient(path="./qdrant_local")   # embedded, offline
    chunks = chunk_docs("kb", a.chunk_size, a.overlap)
    build_index(chunks, client)

    with mlflow.start_run(run_name=f"cs{a.chunk_size}-k{a.top_k}"):
        mlflow.log_params({"chunk_size": a.chunk_size,
                           "overlap": a.overlap, "top_k": a.top_k,
                           "llm": LLM, "n_chunks": len(chunks)})
        ans, hits = answer(a.question, client, a.top_k)
        mlflow.log_metric("top_score", hits[0][1])
        mlflow.log_text(ans, "answer.txt")
        print("Q:", a.question)
        print("Top source:", hits[0][0]["source"],
              f"(score {hits[0][1]:.3f})")
        print("A:", ans)
