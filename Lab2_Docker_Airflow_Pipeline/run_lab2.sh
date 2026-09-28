#!/usr/bin/env bash
# Lab 2 -- Containerize and Orchestrate a Model Deployment Pipeline (Docker + Airflow)
# Part A: export_champion -> docker build -> docker run score_batch.py (volume-mounted data)
# Part B: Airflow DAG cartvista_weekly_retrain (refresh -> train x2 -> gate -> promote),
#         then the same DAG with an impossible gate (0.95) to prove promote is blocked.
#
# Usage:  bash run_lab2.sh
#   INSTALL=1       pip install requirements (+ Airflow if Part B runs)
#   SKIP_DOCKER=1   skip Part A
#   SKIP_AIRFLOW=1  skip Part B (skipped automatically on native Windows -- use WSL2)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

# ---- environment -------------------------------------------------------------
if   [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate
elif [ -f .venv/bin/activate ];     then source .venv/bin/activate
fi
PY="${PYTHON:-$(command -v python || command -v python3)}"
step() { printf '\n===== %s =====\n' "$*"; }

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

# ---- Part A ------------------------------------------------------------------
if [ "${SKIP_DOCKER:-0}" != "1" ]; then
  step "Step A1: Export the champion out of the registry"
  "$PY" export_champion.py

  step "Step A3: Build the Docker image"
  docker build -t cartvista-scorer:v1 .
  docker images cartvista-scorer

  step "Step A4: Run the container with a data volume"
  HOST_DIR="$(pwd -W 2>/dev/null || pwd)"      # pwd -W gives a Windows path under Git Bash
  MSYS_NO_PATHCONV=1 docker run --rm -v "${HOST_DIR}:/data" \
    cartvista-scorer:v1 /data/churn_data.csv /data/scored.csv
  head -3 scored.csv
fi

# ---- Part B ------------------------------------------------------------------
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;; *) IS_WINDOWS=0 ;; esac
if [ "${SKIP_AIRFLOW:-0}" = "1" ] || [ "$IS_WINDOWS" = "1" ]; then
  step "Part B skipped"
  [ "$IS_WINDOWS" = "1" ] && echo "Airflow does not run on native Windows -- re-run this script inside WSL2 Ubuntu."
  exit 0
fi

export AIRFLOW_HOME="$(pwd)/airflow_home"

if [ "${INSTALL:-0}" = "1" ]; then
  step "Step B1: Install Airflow 2.9.2"
  PYVER="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  "$PY" -m pip install "apache-airflow==2.9.2" --constraint \
    "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.2/constraints-${PYVER}.txt"
fi

step "Step B1: Initialise the Airflow metadata DB"
airflow db migrate

step "Step B2: Check the DAG parses"
"$PY" airflow_home/dags/cartvista_pipeline.py
airflow dags list | grep cartvista || true

step "Step B3: Run the full pipeline (gate = 0.72)"
CARTVISTA_AUC_GATE=0.72 airflow dags test cartvista_weekly_retrain

step "Step B4: Prove the gate works (gate = 0.95 -> validate_best fails, promote never runs)"
CARTVISTA_AUC_GATE=0.95 airflow dags test cartvista_weekly_retrain \
  || echo ">>> Gate failed as expected; promote was upstream_failed."

echo
echo "For the Grid-view screenshots, run 'airflow standalone' (UI at http://localhost:8080)"
echo "with AIRFLOW_HOME=$AIRFLOW_HOME and trigger:  airflow dags trigger cartvista_weekly_retrain"

step "Lab 2 complete"
