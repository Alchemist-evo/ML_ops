#!/usr/bin/env bash
# Lab 4 -- Monitoring, Drift Detection and Test-in-Production
# Part A: serve_churn_monitored.py -> Prometheus+Grafana (docker compose)
#         -> traffic_generator.py (normal, then --drift) -> psi_check.py
# Part B: train challenger -> register_challenger.py -> serve_ab.py (90/10)
#         -> mixed traffic -> read per-variant results from Prometheus
#
# Usage:  bash run_lab4.sh
#   INSTALL=1       pip install requirements first
#   MINUTES=3       length of each Part A traffic act (Part B uses MINUTES_AB=2)
#   KEEP_STACK=1    leave Prometheus/Grafana running at the end (for screenshots)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

# ---- environment -------------------------------------------------------------
if   [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate
elif [ -f .venv/bin/activate ];     then source .venv/bin/activate
fi
PY="${PYTHON:-$(command -v python || command -v python3)}"
MINUTES="${MINUTES:-3}"
MINUTES_AB="${MINUTES_AB:-2}"
BASE="http://127.0.0.1:8000"          # fixed: prometheus.yml scrapes :8000
step() { printf '\n===== %s =====\n' "$*"; }
wait_for() {
  for _ in $(seq 1 90); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "Timed out waiting for $1"; return 1
}
SERVER_PID=""
start_server() { "$PY" -m uvicorn "$1" --port 8000 & SERVER_PID=$!; wait_for "$BASE/health"; }
stop_server()  { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; sleep 2; }
cleanup() {
  stop_server
  if [ "${KEEP_STACK:-0}" != "1" ]; then docker compose down >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
prom_query() {   # prom_query "<PromQL>"
  curl -s -G "http://localhost:9090/api/v1/query" --data-urlencode "query=$1" | "$PY" -c '
import json, sys
for r in json.load(sys.stdin)["data"]["result"]:
    print(" ", r["metric"], "->", round(float(r["value"][1]), 3))'
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

# ============================ PART A ==========================================
step "Step A1: Start the instrumented service"
start_server serve_churn_monitored:app
curl -s "$BASE/metrics" | grep '^churn_' || true

step "Step A2: Start Prometheus + Grafana"
docker compose up -d
docker compose ps
echo "Prometheus: http://localhost:9090   Grafana: http://localhost:3000 (admin / cartvista)"
echo "Step A3: in Grafana add data source http://prometheus:9090 and build the 4 panels:"
echo "  rate(churn_predictions_total[1m])"
echo "  histogram_quantile(0.95, rate(churn_inference_seconds_bucket[1m]))"
echo "  rate(churn_predictions_total{outcome=\"1\"}[2m]) / rate(churn_predictions_total[2m])"
echo "  churn_input_support_tickets"

step "Step A4 act 1: normal traffic for ${MINUTES} min"
"$PY" traffic_generator.py --minutes "$MINUTES"
echo "Predicted churn-rate (normal):"
prom_query 'sum(rate(churn_predictions_total{outcome="1"}[2m])) / sum(rate(churn_predictions_total[2m]))'

step "Step A4 act 2: coupon-campaign drift for ${MINUTES} min"
"$PY" traffic_generator.py --minutes "$MINUTES" --drift
echo "Predicted churn-rate (drift):"
prom_query 'sum(rate(churn_predictions_total{outcome="1"}[2m])) / sum(rate(churn_predictions_total[2m]))'

step "Step A5: PSI"
"$PY" psi_check.py

stop_server

# ============================ PART B ==========================================
step "Step B1: Train and register the challenger"
"$PY" train_tracked.py --model rf --n-estimators 300 --max-depth 10
"$PY" register_challenger.py

step "Step B2: Start the A/B router (90/10)"
start_server serve_ab:app
"$PY" traffic_generator.py --minutes "$MINUTES_AB"
"$PY" traffic_generator.py --minutes "$MINUTES_AB" --drift
sleep 10   # let Prometheus take a final scrape

step "Step B3: Read the verdict"
echo "Requests per variant:"
prom_query 'sum by (variant) (ab_predictions_total)'
echo "Predicted churn-rate per variant:"
prom_query 'sum by (variant) (ab_predictions_total{outcome="1"}) / sum by (variant) (ab_predictions_total)'

step "Lab 4 complete"
if [ "${KEEP_STACK:-0}" = "1" ]; then
  echo "Prometheus/Grafana left running -- stop with: docker compose down"
fi
