"""Container entrypoint: score a CSV of customers with the baked-in model."""
import sys, pandas as pd, mlflow.pyfunc

IN_PATH  = sys.argv[1] if len(sys.argv) > 1 else "/data/input.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "/data/scored.csv"

model = mlflow.pyfunc.load_model("/app/model_export")
df = pd.read_csv(IN_PATH)
features = df.drop(columns=["churn"], errors="ignore")
df["churn_pred"] = model.predict(features)
df.to_csv(OUT_PATH, index=False)
print(f"Scored {len(df)} customers -> {OUT_PATH} "
      f"(predicted churners: {int(df.churn_pred.sum())})")
