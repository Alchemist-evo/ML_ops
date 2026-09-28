"""Serve champion and challenger behind one endpoint, 90/10 split."""
import random, mlflow, mlflow.pyfunc, pandas as pd
from prometheus_client import Counter, make_asgi_app
from fastapi import FastAPI
from pydantic import BaseModel, Field

SPLIT = 0.10          # fraction of traffic the challenger receives

AB_PRED = Counter(
    "ab_predictions_total", "predictions by variant and outcome",
    ["variant", "outcome"])

app = FastAPI(title="CartVista A/B Router")
app.mount("/metrics", make_asgi_app())
state = {}

# ---- same typed request contract as Lab 3 ----
class Customer(BaseModel):
    tenure_months:    int   = Field(ge=0, le=120)
    monthly_spend:    float = Field(ge=0)
    orders_per_month: int   = Field(ge=0)
    support_tickets:  int   = Field(ge=0)
    uses_discounts:   int   = Field(ge=0, le=1)
    app_sessions:     int   = Field(ge=0)

@app.on_event("startup")
def load():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    state["champion"]   = mlflow.pyfunc.load_model(
        "models:/cartvista-churn@champion")
    state["challenger"] = mlflow.pyfunc.load_model(
        "models:/cartvista-churn@challenger")

@app.get("/health")
def health():
    return {"status": "ok", "variants_loaded": sorted(state)}

@app.post("/predict")
def predict(customer: Customer):
    variant = "challenger" if random.random() < SPLIT else "champion"
    df = pd.DataFrame([customer.model_dump()])
    pred = int(state[variant].predict(df)[0])
    AB_PRED.labels(variant=variant, outcome=str(pred)).inc()
    return {"churn_prediction": pred, "variant": variant}
