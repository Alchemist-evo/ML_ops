"""CartVista Lab 1 -- synthetic customer churn snapshot (fully offline)."""
import numpy as np
import pandas as pd
 
def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    tenure_months    = rng.integers(1, 72, n)          # how long a customer
    monthly_spend    = np.round(rng.uniform(200, 3000, n), 2)
    orders_per_month = rng.poisson(3, n)
    support_tickets  = rng.poisson(1.2, n)
    uses_discounts   = rng.integers(0, 2, n)           # 1 = redeems coupons
    app_sessions     = rng.poisson(12, n)              # sessions last month
 
    # Hidden 'true' churn tendency (students: do NOT use this in features!)
    logit = (-1.2
             - 0.030 * tenure_months
             + 0.0004 * monthly_spend
             - 0.25  * orders_per_month
             + 0.45  * support_tickets
             - 0.40  * uses_discounts
             - 0.05  * app_sessions)
    p_churn = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n) < p_churn).astype(int)
 
    return pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend,
        "orders_per_month": orders_per_month,
        "support_tickets": support_tickets,
        "uses_discounts": uses_discounts,
        "app_sessions": app_sessions,
        "churn": churn,
    })
 
if __name__ == "__main__":
    df = generate()
    df.to_csv("churn_data.csv", index=False)
    print(f"Wrote churn_data.csv | rows={len(df)} | churn rate={df.churn.mean():.2%}")

