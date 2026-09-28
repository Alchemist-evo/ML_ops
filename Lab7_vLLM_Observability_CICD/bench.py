"""Benchmark an OpenAI-compatible endpoint: TTFT, tokens/sec, concurrency."""
import argparse, time, statistics, concurrent.futures as cf
from openai import OpenAI

QUESTIONS = ["Explain refund policies for groceries in 3 sentences.",
             "What is a delivery slot? Answer briefly.",
             "Summarise why memberships help frequent shoppers."] * 4

def one_call(client, model, q):
    t0 = time.perf_counter(); first = None; toks = 0
    stream = client.chat.completions.create(
        model=model, stream=True, max_tokens=120,
        messages=[{"role": "user", "content": q}])
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            toks += 1
            if first is None: first = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return first, toks / total

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="not-needed")

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        results = list(ex.map(
            lambda q: one_call(client, a.model, q), QUESTIONS))
    wall = time.perf_counter() - t0
    ttft = [r[0] for r in results]; tps = [r[1] for r in results]
    print(f"c={a.concurrency:2d}  requests={len(QUESTIONS)}  wall={wall:5.1f}s"
          f"  TTFT p50={statistics.median(ttft)*1000:6.0f}ms"
          f"  tok/s per-req p50={statistics.median(tps):5.1f}")
