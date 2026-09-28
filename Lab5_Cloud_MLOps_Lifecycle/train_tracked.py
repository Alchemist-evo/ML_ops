"""CartVista churn training WITH MLflow experiment tracking."""
import argparse, hashlib, pathlib
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

TRACKING_URI = "sqlite:///mlflow.db"     # local file DB -> registry works offline
EXPERIMENT   = "cartvista-churn"

def build_model(kind, args):
    if kind == "logreg":
        return LogisticRegression(C=args.C, max_iter=1000)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=args.n_estimators,
                                      max_depth=args.max_depth, random_state=42)
    raise ValueError(f"unknown model kind: {kind}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg", "rf"], required=True)
    ap.add_argument("--C", type=float, default=1.0)              # logreg
    ap.add_argument("--n-estimators", type=int, default=100)     # rf
    ap.add_argument("--max-depth", type=int, default=6)          # rf
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_csv("churn_data.csv")
    X, y = df.drop(columns=["churn"]), df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    with mlflow.start_run(run_name=f"{args.model}-run"):
        # ---- 1. log WHAT we are about to do ----
        mlflow.log_param("model_kind", args.model)
        if args.model == "logreg":
            mlflow.log_param("C", args.C)
        else:
            mlflow.log_params({"n_estimators": args.n_estimators,
                               "max_depth": args.max_depth})
        data_md5 = hashlib.md5(
            pathlib.Path("churn_data.csv").read_bytes()).hexdigest()[:10]
        mlflow.log_param("data_fingerprint", data_md5)

        # ---- 2. do it ----
        model = build_model(args.model, args)
        model.fit(X_tr, y_tr)

        # ---- 3. log HOW WELL it went ----
        proba = model.predict_proba(X_te)[:, 1]
        preds = model.predict(X_te)
        mlflow.log_metrics({
            "test_auc": roc_auc_score(y_te, proba),
            "test_accuracy": accuracy_score(y_te, preds),
            "test_f1": f1_score(y_te, preds),
        })

        # ---- 4. log THE MODEL ITSELF ----
        mlflow.sklearn.log_model(model, artifact_path="model")

        run = mlflow.active_run().info
        print(f"Logged run {run.run_id[:8]}  auc={roc_auc_score(y_te, proba):.4f}")

if __name__ == "__main__":
    main()
