"""Pick the best tracked run and make it the registered champion."""
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
MODEL_NAME = "cartvista-churn"

client = mlflow.MlflowClient()
exp = client.get_experiment_by_name("cartvista-churn")
best = client.search_runs(
    [exp.experiment_id],
    order_by=["metrics.test_auc DESC"],
    max_results=1)[0]

print(f"Best run {best.info.run_id[:8]} auc={best.data.metrics['test_auc']:.4f} "
      f"params={best.data.params}")

mv = mlflow.register_model(
    model_uri=f"runs:/{best.info.run_id}/model",
    name=MODEL_NAME)

client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print(f"Registered {MODEL_NAME} v{mv.version} and set alias @champion")
