"""A downstream consumer: load the champion by alias and score customers."""
import mlflow, pandas as pd
 
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/cartvista-churn@champion")
 
batch = pd.read_csv("churn_data.csv").drop(columns=["churn"]).head(5)
scores = model.predict(batch)
for i, s in enumerate(scores):
    print(f"customer_{i:03d} -> churn prediction: {int(s)}")

