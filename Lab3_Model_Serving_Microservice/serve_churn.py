"""CartVista churn scoring microservice (FastAPI + MLflow registry)."""
import time
import mlflow, mlflow.pyfunc, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mlflow import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME, ALIAS = "cartvista-churn", "champion"

app = FastAPI(title="CartVista Churn Service", version="1.0")

# ---- typed request contract: the API *is* this class ----
class Customer(BaseModel):
    tenure_months:    int   = Field(ge=0, le=120)
    monthly_spend:    float = Field(ge=0)
    orders_per_month: int   = Field(ge=0)
    support_tickets:  int   = Field(ge=0)
    uses_discounts:   int   = Field(ge=0, le=1)
    app_sessions:     int   = Field(ge=0)

class BatchRequest(BaseModel):
    customers: list[Customer]

state = {}

@app.on_event("startup")
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    state["model"] = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}@{ALIAS}")
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
    state["version"], state["run_id"] = mv.version, mv.run_id
    print(f"Loaded {MODEL_NAME} v{mv.version} (run {mv.run_id[:8]})")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}

@app.get("/model-info")
def model_info():
    return {"model": MODEL_NAME, "alias": ALIAS,
            "version": state["version"], "run_id": state["run_id"]}

@app.post("/predict")
def predict(customer: Customer):
    t0 = time.perf_counter()
    df = pd.DataFrame([customer.model_dump()])
    pred = int(state["model"].predict(df)[0])
    ms = (time.perf_counter() - t0) * 1000
    return {"churn_prediction": pred,
            "model_version": state["version"],
            "inference_ms": round(ms, 2)}

@app.post("/predict-batch")
def predict_batch(batch: BatchRequest):
    if not batch.customers:
        raise HTTPException(422, "empty batch")
    df = pd.DataFrame([c.model_dump() for c in batch.customers])
    preds = state["model"].predict(df)
    return {"predictions": [int(p) for p in preds],
            "count": len(preds), "model_version": state["version"]}
