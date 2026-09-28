#!/usr/bin/env bash
# Lab 6 -- Build, Deploy and Evaluate a RAG Pipeline with a Vector Database
# Part A: rag_pipeline.py (milk question + out-of-scope refusal) -> chunk-size sweep 20/60/200
# Part B: evaluate_rag.py on the golden set for chunk sizes 20/60/200
#
# Needs Ollama running locally (https://ollama.com) with llama3.2:1b pulled.
# Usage:  bash run_lab6.sh        (INSTALL=1 to pip install + pull models first)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

# ---- environment -------------------------------------------------------------
if   [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate
elif [ -f .venv/bin/activate ];     then source .venv/bin/activate
fi
PY="${PYTHON:-$(command -v python || command -v python3)}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
EMBEDDER_MODEL="${EMBEDDER_MODEL:-all-MiniLM-L6-v2}"
step() { printf '\n===== %s =====\n' "$*"; }

if [ "${INSTALL:-0}" = "1" ]; then
  step "One-time setup: packages, LLM and embedder (~1.5 GB)"
  "$PY" -m pip install -r requirements.txt
  ollama pull "$OLLAMA_MODEL"
  "$PY" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDER_MODEL'); print('embedder cached')"
fi

curl -sf http://localhost:11434/api/tags >/dev/null \
  || { echo "Ollama is not running -- start the Ollama app or 'ollama serve'."; exit 1; }

if ! ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL}"; then
  echo "Pulling missing model: ${OLLAMA_MODEL}"
  ollama pull "$OLLAMA_MODEL"
fi

rm -rf "$(dirname "$0")/qdrant_local"

# ============================ PART A ==========================================
step "Step A2: The pipeline -- milk question (expect 7 days, refunds.md)"
"$PY" rag_pipeline.py

step "Step A2: Out-of-scope question (expect 'I don't have that information')"
"$PY" rag_pipeline.py --question "Do you deliver to Pune airport?"

step "Step A3: Chunking as an experiment"
"$PY" rag_pipeline.py --chunk-size 20  --overlap 5
"$PY" rag_pipeline.py --chunk-size 60  --overlap 15
"$PY" rag_pipeline.py --chunk-size 200 --overlap 30

# ============================ PART B ==========================================
step "Step B2: Evaluation harness (chunk 60, the default config)"
"$PY" evaluate_rag.py --chunk-size 60 --overlap 15

step "Step B3: Harness for chunk sizes 20 and 200"
"$PY" evaluate_rag.py --chunk-size 20  --overlap 5
"$PY" evaluate_rag.py --chunk-size 200 --overlap 30

echo
echo "Compare runs in experiments 'cartvista-rag' and 'cartvista-rag-eval':"
echo "  $PY -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"

step "Lab 6 complete"
