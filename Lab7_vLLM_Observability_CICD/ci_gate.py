"""Run the golden-set harness; exit non-zero if below thresholds."""
import sys, subprocess, re

THRESHOLDS = {"retrieval": 10, "answers": 8, "faithful": 9}  # of 12

out = subprocess.run([sys.executable, "evaluate_rag.py"],
                     capture_output=True, text=True).stdout
print(out)
m = re.search(r"retrieval=(\d+)/\d+\s+answers=(\d+)/\d+\s+faithful=(\d+)/\d+", out)
scores = dict(zip(THRESHOLDS, map(int, m.groups())))
fails = {k: v for k, v in scores.items() if v < THRESHOLDS[k]}
if fails:
    print(f"QUALITY GATE FAILED: {fails} (thresholds {THRESHOLDS})")
    sys.exit(1)
print("Quality gate passed.")
