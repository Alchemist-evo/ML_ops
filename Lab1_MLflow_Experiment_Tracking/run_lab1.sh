#!/usr/bin/env bash
# Lab 1 -- Experiment Tracking and Model Registry with MLflow
# Runs: generate_data -> train_no_tracking -> 6-run sweep (train_tracked)
#       -> register_model -> consume_model
#
# Usage:  bash run_lab1.sh            (INSTALL=1 bash run_lab1.sh to pip install first)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

# ---- environment -------------------------------------------------------------
if   [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate   # Windows
elif [ -f .venv/bin/activate ];     then source .venv/bin/activate       # macOS/Linux
fi
PY="${PYTHON:-$(command -v python || command -v python3)}"
step() { printf '\n===== %s =====\n' "$*"; }

if [ "${INSTALL:-0}" = "1" ]; then
  step "Installing requirements"
  "$PY" -m pip install -r requirements.txt
fi

step "Verify packages"
"$PY" -c "import mlflow, sklearn; print('MLflow', mlflow.__version__, '| sklearn', sklearn.__version__)"

# ---- Step 1 ------------------------------------------------------------------
step "Step 1: Generate the CartVista dataset"
"$PY" generate_data.py

# ---- Step 2 ------------------------------------------------------------------
step "Step 2: Train WITHOUT tracking"
"$PY" train_no_tracking.py

# ---- Step 3 + 4 --------------------------------------------------------------
step "Step 3-4: Tracked experiment sweep (3 x logreg, 3 x rf)"
"$PY" train_tracked.py --model logreg --C 0.1
"$PY" train_tracked.py --model logreg --C 1.0
"$PY" train_tracked.py --model logreg --C 10.0
"$PY" train_tracked.py --model rf --n-estimators 50  --max-depth 4
"$PY" train_tracked.py --model rf --n-estimators 200 --max-depth 8
"$PY" train_tracked.py --model rf --n-estimators 400 --max-depth 12

# ---- Step 6 ------------------------------------------------------------------
step "Step 6: Register the best model as @champion"
"$PY" register_model.py

# ---- Step 7 ------------------------------------------------------------------
step "Step 7: Consume the champion from a separate program"
"$PY" consume_model.py

# ---- Step 5 (interactive) ----------------------------------------------------
step "Step 5: Compare runs in the MLflow UI"
if [ "${OPEN_UI:-0}" = "1" ]; then
  echo "Starting MLflow UI on http://127.0.0.1:5000 (Ctrl+C to stop)"
  "$PY" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
else
  echo "Run this to open the UI (or re-run with OPEN_UI=1):"
  echo "  $PY -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
fi

step "Lab 1 complete"
