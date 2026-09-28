#!/usr/bin/env bash
# Lab 8 -- Trace and Evaluate an Agentic Workflow
# agent.py (single Meera/ORD-1001 run) -> evaluate_agent.py (6 cases, task success + avg steps)
#
# Needs: Ollama running with llama3.2:1b, and the Lab 7 Langfuse stack up with
#        LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY exported.
# Usage:  bash run_lab8.sh        (INSTALL=1 to pip install first)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

# ---- environment -------------------------------------------------------------
if   [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate
elif [ -f .venv/bin/activate ];     then source .venv/bin/activate
fi
PY="${PYTHON:-$(command -v python || command -v python3)}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
LANGFUSE_HOST="${LANGFUSE_HOST:-http://localhost:3000}"
step() { printf '\n===== %s =====\n' "$*"; }

if [ "${INSTALL:-0}" = "1" ]; then
  step "Installing requirements"
  "$PY" -m pip install -r requirements.txt
  ollama pull "$OLLAMA_MODEL"
fi

curl -sf http://localhost:11434/api/tags >/dev/null \
  || { echo "Ollama is not running -- start the Ollama app or 'ollama serve'."; exit 1; }
if ! ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL}"; then
  echo "Pulling missing model: ${OLLAMA_MODEL}"
  ollama pull "$OLLAMA_MODEL"
fi
if [ -z "${LANGFUSE_PUBLIC_KEY:-}" ] || [ -z "${LANGFUSE_SECRET_KEY:-}" ]; then
  echo "WARNING: Langfuse keys not exported -- the agent will run but no traces will be recorded."
fi
export LANGFUSE_HOST

step "Step 1: Order system sanity check"
"$PY" -c "from orders_db import lookup_order; print(lookup_order('ORD-1001')); print(lookup_order('ORD-9999'))"

step "Step 3: Run the agent (Meera, ORD-1001)"
"$PY" agent.py

step "Step 4: Evaluate task success on 6 scenarios"
"$PY" evaluate_agent.py

echo
echo "Open Langfuse ($LANGFUSE_HOST) -> Traces -> agent-run to walk each trajectory."

step "Lab 8 complete"
