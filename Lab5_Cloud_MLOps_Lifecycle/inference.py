"""SageMaker sklearn container hooks."""
import joblib, os, pandas as pd

COLS = ["tenure_months", "monthly_spend", "orders_per_month",
        "support_tickets", "uses_discounts", "app_sessions"]

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def predict_fn(data, model):
    df = pd.DataFrame(data, columns=COLS)
    return model.predict(df).tolist()
