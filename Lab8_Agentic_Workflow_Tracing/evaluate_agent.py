"""Task-success evaluation: 6 scenarios with verifiable outcomes."""
from agent import run_agent, lf

CASES = [
  {"q": "Meera: mangoes in ORD-1001 arrived bruised. Refund possible?",
   "must": ["refund"], "must_not": ["not eligible"]},   # 4d perishable: yes
  {"q": "Rahul: I want to return the bottle from ORD-1002. Can I?",
   "must": ["30"], "must_not": []},        # 45 days > 30: should say no
  {"q": "Meera: refund the milk in ORD-1003 please.",
   "must": ["7"], "must_not": []},         # 15d perishable > 7: no
  {"q": "Sana: cancel ORD-1004 right now.",
   "must": ["out for delivery", "cannot", "not possible"],
   "must_not": []},                        # policy: no cancel in transit
  {"q": "What does CartVista Plus membership cost?",
   "must": ["299"], "must_not": []},       # policy-only: 1 tool suffices
  {"q": "Refund status for order ORD-9999?",
   "must": ["no order", "not found", "doesn't exist"],
   "must_not": []},                        # honest failure on bad id
]

passed = 0; total_steps = 0
for c in CASES:
    out = run_agent(c["q"]); a = (out["answer"] or "").lower()
    ok = (any(m.lower() in a for m in c["must"])
          and not any(m.lower() in a for m in c["must_not"]))
    passed += ok; total_steps += out["steps"]
    print(f"[{'PASS' if ok else 'FAIL'}] steps={out['steps']}  {c['q'][:52]}")
print(f"\ntask success: {passed}/{len(CASES)}   avg steps: {total_steps/len(CASES):.1f}")
lf.flush()
