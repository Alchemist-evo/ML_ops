"""Tiny load test: 300 requests, report the percentiles that matter."""
import time, statistics, requests

URL = "http://127.0.0.1:8000/predict"
PAYLOAD = {"tenure_months": 24, "monthly_spend": 1500.0,
           "orders_per_month": 3, "support_tickets": 1,
           "uses_discounts": 1, "app_sessions": 10}

lat = []
for _ in range(300):
    t0 = time.perf_counter()
    r = requests.post(URL, json=PAYLOAD, timeout=5)
    r.raise_for_status()
    lat.append((time.perf_counter() - t0) * 1000)

lat.sort()
q = lambda p: lat[int(p / 100 * len(lat)) - 1]
print(f"n={len(lat)}  p50={q(50):.1f}ms  p95={q(95):.1f}ms  "
      f"p99={q(99):.1f}ms  max={max(lat):.1f}ms")
