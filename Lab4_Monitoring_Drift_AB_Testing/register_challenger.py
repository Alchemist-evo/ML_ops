"""Step B1: register the most recent run and point the @challenger alias at it."""
import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri('sqlite:///mlflow.db')
c = MlflowClient()
exp = c.get_experiment_by_name('cartvista-churn')
run = c.search_runs([exp.experiment_id],
        order_by=['attributes.start_time DESC'], max_results=1)[0]
mv = mlflow.register_model(f'runs:/{run.info.run_id}/model', 'cartvista-churn')
c.set_registered_model_alias('cartvista-churn', 'challenger', mv.version)
print(f'challenger -> v{mv.version}')
