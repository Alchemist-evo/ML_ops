"""Lab 6's RAG answer path with Langfuse tracing: one trace per question,
   with a retrieve span and a generate generation (prompt, output, tokens)."""
import ollama
from langfuse import Langfuse
from qdrant_client import QdrantClient
from rag_pipeline import chunk_docs, build_index, retrieve, PROMPT, LLM

lf = Langfuse()          # reads the env vars

def answer_traced(question, client, top_k):
    trace = lf.trace(name="support-query", input=question)

    span = trace.span(name="retrieve", input=question)
    hits = retrieve(question, client, top_k)
    span.end(output=[{"src": p["source"], "score": round(s, 3)}
                     for p, s in hits])

    ctx = "\n---\n".join(f"[{p['source']}] {p['text']}" for p, _ in hits)
    prompt = PROMPT.format(context=ctx, question=question)

    gen = trace.generation(name="generate", model=LLM, input=prompt)
    resp = ollama.chat(model=LLM,
        messages=[{"role": "user", "content": prompt}])
    out = resp["message"]["content"]
    gen.end(output=out, usage={
        "input": resp.get("prompt_eval_count", 0),
        "output": resp.get("eval_count", 0)})

    trace.update(output=out)
    return out, hits

QUESTIONS = [
    "How long do I have to return milk that arrived spoiled?",
    "What does CartVista Plus cost?",
    "Is COD available on express orders?",
    "Can I cancel once the rider has left?",
    "Do you sell mobile phones?",            # out-of-scope
]

if __name__ == "__main__":
    client = QdrantClient(path="./qdrant_local")
    build_index(chunk_docs("kb", 60, 15), client)
    for q in QUESTIONS:
        out, hits = answer_traced(q, client, top_k=3)
        print(f"Q: {q}\n   [{hits[0][0]['source']}] A: {out.strip()[:160]}\n")
    lf.flush()
    print("Traces sent -- open Langfuse -> Traces.")
