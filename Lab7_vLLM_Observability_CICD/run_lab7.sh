#!/usr/bin/env bash
# Lab 7 -- Deploy an LLM with vLLM, Benchmark Inference, Observability + CI/CD
# Part A: vLLM server -> bench.py at c=1/4/8 -> Ollama baseline
#         (CPU route when no NVIDIA GPU / vLLM: bench Ollama at c=1 and c=4)
# Part B: rag_traced.py (Langfuse traces) -> ci_gate.py healthy -> sabotaged prompt -> restored
#
# Usage:  bash run_lab7.sh
#   ROUTE=vllm|cpu     force a Part A route (default: auto-detect)
#   INSTALL=1          pip install requirements (+ vllm on the vllm route)
#   LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY must be exported for tracing
#     (self-host: git clone https://github.com/langfuse/langfuse.git && cd langfuse && docker compose up -d)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

# ---- environment -------------------------------------------------------------
if   [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate
elif [ -f .venv/bin/activate ];     then source .venv/bin/activate
fi
PY="${PYTHON:-$(command -v python || command -v python3)}"
step() { printf '\n===== %s =====\n' "$*"; }
wait_for() {   # wait_for URL SECONDS
  for _ in $(seq 1 "${2:-90}"); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "Timed out waiting for $1"; return 1
}
VLLM_PID=""
cleanup() {
  [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
  if [ -f rag_pipeline.py.bak ]; then mv -f rag_pipeline.py.bak rag_pipeline.py; echo "Prompt restored."; fi
}
trap cleanup EXIT

if [ "${INSTALL:-0}" = "1" ]; then
  step "Installing requirements"
  "$PY" -m pip install -r requirements.txt
fi

curl -sf http://localhost:11434/api/tags >/dev/null \
  || { echo "Ollama is not running -- start the Ollama app or 'ollama serve'."; exit 1; }

# ============================ PART A ==========================================
if [ -z "${ROUTE:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && "$PY" -c "import vllm" >/dev/null 2>&1; then
    ROUTE=vllm; else ROUTE=cpu; fi
fi
step "Part A route: $ROUTE"

if [ "$ROUTE" = "vllm" ]; then
  [ "${INSTALL:-0}" = "1" ] && "$PY" -m pip install vllm
  step "Step A1: Start the vLLM server on :8010"
  "$PY" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-model-len 2048 --port 8010 &
  VLLM_PID=$!
  wait_for http://localhost:8010/v1/models 900     # first start downloads ~1 GB

  step "Step A2: Benchmark vLLM"
  "$PY" bench.py --concurrency 1
  "$PY" bench.py --concurrency 4
  "$PY" bench.py --concurrency 8
  kill "$VLLM_PID" 2>/dev/null || true; VLLM_PID=""

  step "Step A2: Ollama baseline (Lab 6 model)"
  "$PY" bench.py --base-url http://localhost:11434/v1 --model llama3.2:1b --concurrency 4
else
  step "Step A2 (CPU route): Benchmark Ollama at two concurrency levels"
  "$PY" bench.py --base-url http://localhost:11434/v1 --model llama3.2:1b --concurrency 1
  "$PY" bench.py --base-url http://localhost:11434/v1 --model llama3.2:1b --concurrency 4
fi

# ============================ PART B ==========================================
step "Step B2: Trace the RAG pipeline in Langfuse"
if [ -n "${LANGFUSE_PUBLIC_KEY:-}" ] && [ -n "${LANGFUSE_SECRET_KEY:-}" ]; then
  export LANGFUSE_HOST="${LANGFUSE_HOST:-http://localhost:3080}"
  "$PY" rag_traced.py
else
  echo "Skipped: export LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY first (Step B1)."
fi

step "Step B3a: CI gate on the healthy prompt (expect pass)"
if "$PY" ci_gate.py; then echo ">>> gate exit 0"; else echo ">>> gate exit 1"; fi

step "Step B3b: Sabotage the prompt (delete the 'Answer ONLY from the context' line)"
cp rag_pipeline.py rag_pipeline.py.bak
sed -i '/Answer ONLY from the context below/d' rag_pipeline.py
grep -n -A3 '^PROMPT' rag_pipeline.py
if "$PY" ci_gate.py; then echo ">>> gate exit 0 (sabotage not caught this time -- see E4)"; else echo ">>> gate exit 1 (sabotage caught)"; fi

step "Step B3c: Restore the prompt and re-run the gate"
mv -f rag_pipeline.py.bak rag_pipeline.py
if "$PY" ci_gate.py; then echo ">>> gate exit 0"; else echo ">>> gate exit 1"; fi

step "Lab 7 complete"
