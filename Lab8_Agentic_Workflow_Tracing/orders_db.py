"""A tiny CartVista order store the agent can query."""
from datetime import date

TODAY = date(2026, 7, 20)

ORDERS = {
  "ORD-1001": {"customer": "Meera",  "item": "Alphonso mangoes (2 kg)",
               "category": "perishable",     "status": "delivered",
               "delivered_on": "2026-07-16", "amount": 850},
  "ORD-1002": {"customer": "Rahul",  "item": "Steel water bottle",
               "category": "non-perishable", "status": "delivered",
               "delivered_on": "2026-06-05", "amount": 499},
  "ORD-1003": {"customer": "Meera",  "item": "Organic milk (6 pack)",
               "category": "perishable",     "status": "delivered",
               "delivered_on": "2026-07-05", "amount": 320},
  "ORD-1004": {"customer": "Sana",   "item": "Basmati rice (10 kg)",
               "category": "non-perishable", "status": "out_for_delivery",
               "delivered_on": None,         "amount": 1250},
}

def lookup_order(order_id: str) -> dict:
    o = ORDERS.get(order_id.strip().upper())
    if not o:
        return {"error": f"no order {order_id}"}
    days = None
    if o["delivered_on"]:
        days = (TODAY - date.fromisoformat(o["delivered_on"])).days
    return {**o, "days_since_delivery": days}
