#!/usr/bin/env bash
# Lab 5 -- Manage the MLOps Lifecycle Using a Managed Cloud Service (AWS SageMaker track)
# export_champion -> prep_sagemaker (tar + S3 upload) -> deploy_endpoint (ml.t2.medium)
# -> invoke_endpoint -> create model package group -> MANDATORY TEARDOWN
#
# COSTS REAL MONEY. Set a $5 billing alert first. Teardown runs automatically on exit
# (even on failure/Ctrl+C) once the endpoint has been requested.
#
# Required:  aws configure (same region as the console)
#            export SAGEMAKER_ROLE_ARN=arn:aws:iam::<acct>:role/<SageMakerExecutionRole>
# Optional:  INSTALL=1, SKLEARN_FRAMEWORK_VERSION=1.2-1
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

# ---- environment -------------------------------------------------------------
if   [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate
elif [ -f .venv/bin/activate ];     then source .venv/bin/activate
fi
PY="${PYTHON:-$(command -v python || command -v python3)}"
EP=cartvista-churn-ep
step() { printf '\n===== %s =====\n' "$*"; }

if [ "${INSTALL:-0}" = "1" ]; then
  step "Step 1: Install boto3 + sagemaker"
  "$PY" -m pip install -r requirements.txt
fi

: "${SAGEMAKER_ROLE_ARN:?Set SAGEMAKER_ROLE_ARN to your SageMaker execution role ARN}"
aws sts get-caller-identity >/dev/null || { echo "Run 'aws configure' first."; exit 1; }
echo "AWS region: $(aws configure get region)"

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

# ---- Step 2 ------------------------------------------------------------------
step "Step 2: Export the champion (Lab 2) and package/upload for SageMaker"
"$PY" export_champion.py
"$PY" prep_sagemaker.py

# ---- teardown (Section 6) -- armed before deploying ---------------------------
teardown() {
  step "Section 6: MANDATORY teardown"
  aws sagemaker delete-endpoint        --endpoint-name "$EP"        || true
  aws sagemaker delete-endpoint-config --endpoint-config-name "$EP" || true
  aws sagemaker list-endpoints
  date
  echo "Screenshot the empty endpoint list above with its timestamp."
}
trap teardown EXIT

# ---- Step 3 ------------------------------------------------------------------
step "Step 3: Deploy the real-time endpoint (4-8 minutes)"
"$PY" deploy_endpoint.py

# ---- Step 4 ------------------------------------------------------------------
step "Step 4: Invoke it"
"$PY" invoke_endpoint.py
echo "Now open the endpoint's Monitoring tab in the console and screenshot it."
if [ -t 0 ]; then read -r -p "Press Enter when done to tear down the endpoint... " _; fi

# ---- Step 5 ------------------------------------------------------------------
step "Step 5: Create the model package group"
aws sagemaker create-model-package-group \
  --model-package-group-name cartvista-churn \
  || echo "(group probably exists already)"
echo "Register the S3 model as version 1 via: SageMaker Studio -> Models -> Registered models -> Register"
echo "S3 model: $(cat s3_uri.txt)"

step "Lab 5 steps done -- teardown follows"
