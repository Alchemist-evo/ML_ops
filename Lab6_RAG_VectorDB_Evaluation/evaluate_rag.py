"""Score the pipeline on the golden set; log everything to MLflow."""
import argparse
import mlflow, ollama
from qdrant_client import QdrantClient
from rag_pipeline import chunk_docs, build_index, answer
from golden_set import GOLDEN

# config under test (defaults 60 / 15 / 3; override on the CLI for Step B3)
ap = argparse.ArgumentParser()
ap.add_argument("--chunk-size", type=int, default=60)
ap.add_argument("--overlap",    type=int, default=15)
ap.add_argument("--top-k",      type=int, default=3)
_a = ap.parse_args()
CHUNK, OVERLAP, TOPK = _a.chunk_size, _a.overlap, _a.top_k

JUDGE_PROMPT = """Context:\n{ctx}\n\nAnswer:\n{ans}\n
Does the Answer contain any claim NOT supported by the Context?
Reply with exactly one word: FAITHFUL or UNFAITHFUL."""

def judge(ctx, ans):
    r = ollama.chat(model="llama3.2:1b", messages=[{
        "role": "user", "content": JUDGE_PROMPT.format(ctx=ctx, ans=ans)}])
    return "FAITHFUL" in r["message"]["content"].upper()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("cartvista-rag-eval")

client = QdrantClient(path="./qdrant_local")
build_index(chunk_docs("kb", CHUNK, OVERLAP), client)

hit = ans_ok = faithful = 0
with mlflow.start_run(run_name=f"eval-cs{CHUNK}-k{TOPK}"):
    mlflow.log_params({"chunk_size": CHUNK, "top_k": TOPK})
    for g in GOLDEN:
        a, hits = answer(g["q"], client, TOPK)
        srcs = [p["source"] for p, _ in hits]
        ctx = " ".join(p["text"] for p, _ in hits)
        r_hit = (g["src"] is None) or (g["src"] in srcs)
        a_ok  = any(m.lower() in a.lower() for m in g["must"])
        f_ok  = judge(ctx, a)
        hit += r_hit; ans_ok += a_ok; faithful += f_ok
        flag = "" if (r_hit and a_ok) else "  <-- REVIEW"
        print(f"[R:{int(r_hit)} A:{int(a_ok)} F:{int(f_ok)}] {g['q']}{flag}")
    n = len(GOLDEN)
    mlflow.log_metrics({"retrieval_hit_rate": hit / n,
                        "answer_accuracy": ans_ok / n,
                        "faithfulness": faithful / n})
    print(f"\nretrieval={hit}/{n}  answers={ans_ok}/{n}  faithful={faithful}/{n}")
