"""A minimal two-tool agent with a hand-rolled decide-act-observe loop,
   fully traced in Langfuse. No framework -- so every moving part is
   visible and yours to instrument."""
import json, re, ollama
from langfuse import Langfuse
from qdrant_client import QdrantClient
from rag_pipeline import chunk_docs, build_index, retrieve
from orders_db import lookup_order

MODEL = "llama3.2:1b"          # or llama3.2:3b if pulled
MAX_STEPS = 5                  # the leash: agents must not run free
lf = Langfuse()

# ---- Tool 2: the Lab 6 retriever, wrapped ----
_client = QdrantClient(path="./qdrant_local")
build_index(chunk_docs("kb", 60, 15), _client)

def search_policy(query: str) -> str:
    hits = retrieve(query, _client, top_k=2)
    return "\n".join(f"[{p['source']}] {p['text']}" for p, _ in hits)

TOOLS = {"lookup_order": lookup_order, "search_policy": search_policy}

SYSTEM = """You are CartVista's support agent. You have two tools:
1. lookup_order(order_id)   -- fetch order facts
2. search_policy(query)     -- search company policy documents

Rules: To use a tool, reply with ONLY a JSON object:
  {"tool": "<name>", "arg": "<argument>"}
When you have enough information, reply with ONLY:
  {"answer": "<final answer to the customer>"}
Use each tool only when needed. Never invent order data or policy."""

def parse(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)   # tolerate chatter
    if not m: return None
    try: return json.loads(m.group(0))
    except json.JSONDecodeError: return None

def run_agent(question: str) -> dict:
    trace = lf.trace(name="agent-run", input=question)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user",   "content": question}]
    for step in range(1, MAX_STEPS + 1):
        gen = trace.generation(name=f"decide-{step}", model=MODEL,
                               input=msgs[-1]["content"])
        resp = ollama.chat(model=MODEL, messages=msgs)
        text = resp["message"]["content"]
        gen.end(output=text)
        act = parse(text)

        if act is None:                      # model rambled
            msgs.append({"role": "user", "content":
                "Reply with ONLY the JSON object as instructed."})
            continue
        if "answer" in act:
            trace.update(output=act["answer"],
                         metadata={"steps": step})
            return {"answer": act["answer"], "steps": step}
        if act.get("tool") in TOOLS:
            span = trace.span(name=f"tool:{act['tool']}",
                              input=act.get("arg"))
            result = TOOLS[act["tool"]](act.get("arg", ""))
            span.end(output=result)
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user",
                "content": f"Tool result: {json.dumps(result, default=str)}"})
        else:
            msgs.append({"role": "user", "content":
                f"Unknown tool {act.get('tool')}. Use lookup_order or search_policy."})
    trace.update(output="STEP LIMIT REACHED", metadata={"steps": MAX_STEPS})
    return {"answer": None, "steps": MAX_STEPS}

if __name__ == "__main__":
    q = ("Customer Meera asks: my mangoes in order ORD-1001 arrived "
         "bruised. Can I still get a refund, and how much?")
    out = run_agent(q)
    print(f"steps={out['steps']}\n{out['answer']}")
    lf.flush()
