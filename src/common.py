"""Shared helpers for the spike. Boring on purpose."""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

SEED = 42
LABELS = ["book", "cancel", "change"]
MODEL = "qwen2.5:3b"
OLLAMA_BASE = "http://localhost:11434"

for d in (RUNS, RESULTS, ARTIFACTS):
    d.mkdir(exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def macro_f1(golds, preds):
    from sklearn.metrics import f1_score
    return f1_score(golds, preds, labels=LABELS, average="macro", zero_division=0)


class CallLogger:
    """Appends one JSONL row per LM call to runs/<name>.jsonl."""

    def __init__(self, name, fresh=True):
        self.path = RUNS / f"{name}.jsonl"
        if fresh and self.path.exists():
            self.path.unlink()
        self.count = 0

    def log(self, **row):
        self.count += 1
        row["ts"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_result(stage, condition, f1=None, validity=None, calls=None, wallclock_s=None):
    """One row per (stage, condition) in results/results.csv; replaces existing row."""
    import csv
    path = RESULTS / "results.csv"
    fields = ["stage", "condition", "f1", "validity", "calls", "wallclock_s"]
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f)
                    if not (r["stage"] == stage and r["condition"] == condition)]
    rows.append({
        "stage": stage, "condition": condition,
        "f1": "" if f1 is None else f"{f1:.4f}",
        "validity": "" if validity is None else f"{validity:.4f}",
        "calls": "" if calls is None else calls,
        "wallclock_s": "" if wallclock_s is None else f"{wallclock_s:.1f}",
    })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
