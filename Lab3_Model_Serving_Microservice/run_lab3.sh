#!/usr/bin/env bash
# Lab 3 -- Deploy and Serve the ML Model as a Microservice
# Starts serve_churn.py (uvicorn, background) -> /health, /model-info
# -> at-risk + loyal /predict -> 422 validation check -> load_test.py -> stops the server.
#
# Usage:  bash run_lab3.sh          (INSTALL=1 to pip install first, PORT=8001 to change port)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

# ---- environment -------------------------------------------------------------
if   [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate
elif [ -f .venv/bin/activate ];     then source .venv/bin/activate
fi
PY="${PYTHON:-$(command -v python || command -v python3)}"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
step() { printf '\n===== %s =====\n' "$*"; }
wait_for() {   # wait_for URL -- poll until the service answers (max ~90 s)
  for _ in $(seq 1 90); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "Timed out waiting for $1"; return 1
}

if [ "${INSTALL:-0}" = "1" ]; then
  step "Installing requirements"
  "$PY" -m pip install -r requirements.txt
fi

# ---- prerequisite: Lab 1 registry with @champion -----------------------------
has_champion() {
  [ -f mlflow.db ] && "$PY" - >/dev/null 2>&1 <<'EOF'
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.MlflowClient().get_model_version_by_alias("cartvista-churn", "champion")
EOF
}
if ! has_champion; then
  step "Prerequisite (Lab 1): build mlflow.db and register @champion"
  [ -f churn_data.csv ] || "$PY" generate_data.py
  "$PY" train_tracked.py --model logreg --C 1.0
  "$PY" train_tracked.py --model rf --n-estimators 200 --max-depth 8
  "$PY" register_model.py
fi

# ---- Step 2: start the service ----------------------------------------------
step "Step 2: Start the service on port $PORT"
"$PY" -m uvicorn serve_churn:app --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
wait_for "$BASE/health"
echo "API docs: $BASE/docs"
curl -s "$BASE/health";     echo
curl -s "$BASE/model-info"; echo

# ---- Step 3: call it like the app team --------------------------------------
step "Step 3a: At-risk customer (expect churn_prediction 1)"
curl -s -X POST "$BASE/predict" -H "Content-Type: application/json" \
  -d '{"tenure_months": 2, "monthly_spend": 2500, "orders_per_month": 1,
       "support_tickets": 6, "uses_discounts": 0, "app_sessions": 2}'; echo

step "Step 3b: Loyal customer (expect churn_prediction 0)"
curl -s -X POST "$BASE/predict" -H "Content-Type: application/json" \
  -d '{"tenure_months": 60, "monthly_spend": 1800, "orders_per_month": 8,
       "support_tickets": 0, "uses_discounts": 1, "app_sessions": 20}'; echo

# ---- Step 4: validation ------------------------------------------------------
step "Step 4: Invalid input (expect HTTP 422)"
curl -s -w '\nHTTP %{http_code}\n' -X POST "$BASE/predict" -H "Content-Type: application/json" \
  -d '{"tenure_months": -5, "monthly_spend": 2500, "orders_per_month": 1,
       "support_tickets": 6, "uses_discounts": 0, "app_sessions": 2}'

# ---- Step 5: latency ---------------------------------------------------------
step "Step 5: Latency profile (300 requests)"
if [ "$PORT" != "8000" ]; then
  sed "s#127.0.0.1:8000#127.0.0.1:${PORT}#" load_test.py | "$PY" -
else
  "$PY" load_test.py
fi

step "Lab 3 complete (stopping the service)"
