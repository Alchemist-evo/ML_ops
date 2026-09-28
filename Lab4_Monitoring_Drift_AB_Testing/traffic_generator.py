"""Send realistic traffic; --drift replays the coupon-campaign population."""
import argparse, random, time, requests

URL = "http://127.0.0.1:8000/predict"

def customer(drift=False):
    if not drift:   # the population the model was trained on
        return {
            "tenure_months": random.randint(1, 72),
            "monthly_spend": round(random.uniform(200, 3000), 2),
            "orders_per_month": max(0, int(random.gauss(3, 1.5))),
            "support_tickets": max(0, int(random.gauss(1.2, 1))),
            "uses_discounts": random.randint(0, 1),
            "app_sessions": max(0, int(random.gauss(12, 4))),
        }
    # coupon-campaign cohort: new, low-engagement, ticket-heavy
    return {
        "tenure_months": random.randint(1, 4),
        "monthly_spend": round(random.uniform(100, 800), 2),
        "orders_per_month": max(0, int(random.gauss(1, 0.8))),
        "support_tickets": max(0, int(random.gauss(4, 1.5))),
        "uses_discounts": 1,
        "app_sessions": max(0, int(random.gauss(4, 2))),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--minutes", type=float, default=3)
    a = ap.parse_args()
    end = time.time() + a.minutes * 60
    n = 0
    while time.time() < end:
        requests.post(URL, json=customer(a.drift), timeout=5)
        n += 1
        time.sleep(0.2)      # ~5 req/s
    print(f"sent {n} requests (drift={a.drift})")
