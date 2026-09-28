"""CartVista weekly retrain pipeline: data -> train x2 -> gate -> promote."""
from datetime import datetime
import os, subprocess, json, pathlib
from airflow.decorators import dag, task

PROJECT = pathlib.Path(__file__).resolve().parents[2]   # lab folder (cartvista-mlops)
# a model below this bar must NOT become champion
# (Step B4: set CARTVISTA_AUC_GATE=0.95 -- or edit this line -- to prove the gate works)
AUC_GATE = float(os.environ.get("CARTVISTA_AUC_GATE", 0.72))

def run(cmd):
    r = subprocess.run(cmd, cwd=PROJECT, shell=True,
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    r.check_returncode()
    return r.stdout

@dag(schedule=None, start_date=datetime(2026, 1, 1), catchup=False,
     tags=["cartvista"])
def cartvista_weekly_retrain():

    @task
    def refresh_data():
        run("python generate_data.py")

    @task
    def train_logreg():
        run("python train_tracked.py --model logreg --C 1.0")

    @task
    def train_rf():
        run("python train_tracked.py --model rf --n-estimators 200 --max-depth 8")

    @task
    def validate_best() -> float:
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT}/mlflow.db")
        client = MlflowClient()
        exp = client.get_experiment_by_name("cartvista-churn")
        best = client.search_runs([exp.experiment_id],
                order_by=["metrics.test_auc DESC"], max_results=1)[0]
        auc = best.data.metrics["test_auc"]
        print(f"Best AUC this cycle: {auc:.4f} (gate: {AUC_GATE})")
        if auc < AUC_GATE:
            raise ValueError(f"Quality gate FAILED: {auc:.4f} < {AUC_GATE}")
        return auc

    @task
    def promote():
        run("python register_model.py")

    d = refresh_data()
    gate = validate_best()
    d >> [train_logreg(), train_rf()] >> gate >> promote()

cartvista_weekly_retrain()
