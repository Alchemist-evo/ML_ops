"""Population Stability Index between training data and live traffic."""
import numpy as np, pandas as pd
from traffic_generator import customer

def psi(expected, actual, bins=10):
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual, cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

train = pd.read_csv("churn_data.csv")
live_normal = pd.DataFrame([customer(False) for _ in range(1000)])
live_drift  = pd.DataFrame([customer(True)  for _ in range(1000)])

for col in ["tenure_months", "support_tickets", "monthly_spend"]:
    print(f"{col:18s} PSI normal={psi(train[col], live_normal[col]):.3f}  "
          f"drift={psi(train[col], live_drift[col]):.3f}")
