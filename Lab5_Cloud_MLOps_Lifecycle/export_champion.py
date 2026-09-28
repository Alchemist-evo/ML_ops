"""Export @champion artifacts to ./model_export for Docker packaging."""
import shutil, pathlib, mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
mv = client.get_model_version_by_alias("cartvista-churn", "champion")
print(f"Exporting cartvista-churn v{mv.version} (run {mv.run_id[:8]})")

dst = pathlib.Path("model_export")
if dst.exists(): shutil.rmtree(dst)
local = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{mv.run_id}/model", dst_path=".")
shutil.move(local, dst)
print(f"Model exported to {dst}/ -- contents: {[p.name for p in dst.iterdir()]}")
