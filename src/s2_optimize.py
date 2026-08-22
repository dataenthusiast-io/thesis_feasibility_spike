"""S2 — DSPy + MIPROv2 end-to-end against macro F1 (confirms A2).

Baseline dspy.Predict program vs MIPROv2(auto="light")-optimized program,
same model (qwen2.5:3b via Ollama), same test set.

Also importable: build_lm/IntentSignature/make_examples/evaluate are reused
by s4_recover.py.
"""
import json
import time
from typing import Literal

import dspy

from common import (ARTIFACTS, DATA, MODEL, OLLAMA_BASE, RESULTS, RUNS,
                    SEED, append_result, load_jsonl, macro_f1)

MAX_CALLS = 5000  # spec hard cap per stage


def build_lm():
    return dspy.LM(f"ollama_chat/{MODEL}", api_base=OLLAMA_BASE,
                   api_key="", temperature=0, max_tokens=200, cache=False)


class IntentSignature(dspy.Signature):
    """Classify the airline customer's utterance into exactly one intent:
    book (wants to book a new flight), cancel (wants to cancel an existing
    reservation), or change (wants to change an existing reservation)."""

    text: str = dspy.InputField(desc="one customer utterance")
    intent: Literal["book", "cancel", "change"] = dspy.OutputField()


def metric(example, pred, trace=None):
    """Per-example exact match on the intent label (MIPRO's objective).
    Macro F1 is computed separately for reporting."""
    try:
        return str(pred.intent).strip().lower() == example.intent
    except Exception:
        return False


def make_examples(path):
    return [dspy.Example(text=r["text"], intent=r["intent"]).with_inputs("text")
            for r in load_jsonl(path)]


def evaluate(program, path):
    rows = load_jsonl(path)
    golds, preds = [], []
    for r in rows:
        try:
            out = program(text=r["text"])
            label = str(out.intent).strip().lower()
        except Exception:
            label = "__invalid__"
        golds.append(r["intent"])
        preds.append(label if label in ("book", "cancel", "change") else "__invalid__")
    return macro_f1(golds, preds), golds, preds


def dump_history(lm, name):
    rows = []
    for h in lm.history:
        rows.append({
            "prompt": str(h.get("messages") or h.get("prompt"))[:4000],
            "response": str(h.get("outputs"))[:4000],
            "kwargs": {k: v for k, v in (h.get("kwargs") or {}).items()
                       if isinstance(v, (int, float, str, bool))},
        })
    with open(RUNS / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def run_mipro(trainset, lm, run_name):
    """Run MIPROv2 auto=light; returns (optimized_program, wallclock, n_calls)."""
    optimizer = dspy.MIPROv2(metric=metric, auto="light", seed=SEED,
                             num_threads=4, max_errors=100)
    program = dspy.Predict(IntentSignature)
    calls_before = len(lm.history)
    t0 = time.time()
    optimized = optimizer.compile(program, trainset=trainset,
                                  requires_permission_to_run=False)
    wall = time.time() - t0
    n_calls = len(lm.history) - calls_before
    dump_history(lm, run_name)
    if n_calls > MAX_CALLS:
        print(f"WARNING: optimization consumed {n_calls} calls (> {MAX_CALLS} cap)")
    return optimized, wall, n_calls


def main():
    lm = build_lm()
    dspy.configure(lm=lm)
    trainset = make_examples(DATA / "dev.jsonl")

    baseline = dspy.Predict(IntentSignature)
    t0 = time.time()
    f1_base, _, preds_base = evaluate(baseline, DATA / "test.jsonl")
    wall_base = time.time() - t0
    calls_base = len(lm.history)
    validity_base = sum(p != "__invalid__" for p in preds_base) / len(preds_base)
    print(f"baseline (unoptimized Predict): macro F1 = {f1_base:.4f}, "
          f"validity = {validity_base:.4f}, {wall_base:.0f}s, {calls_base} calls")
    append_result("s2", "baseline_unoptimized", f1=f1_base, validity=validity_base,
                  calls=calls_base, wallclock_s=wall_base)

    print("running MIPROv2 auto=light ...")
    try:
        optimized, wall_opt, calls_opt = run_mipro(trainset, lm, "s2_mipro_history")
        crashed = False
    except Exception as e:
        print(f"MIPROv2 CRASHED: {type(e).__name__}: {e}")
        dump_history(lm, "s2_mipro_history")
        crashed = True
        raise

    optimized.save(ARTIFACTS / "optimized_program.json")
    print(f"optimization done: {wall_opt:.0f}s, {calls_opt} LM calls; "
          f"program saved to artifacts/optimized_program.json")

    calls_before = len(lm.history)
    t0 = time.time()
    f1_opt, _, preds_opt = evaluate(optimized, DATA / "test.jsonl")
    wall_eval = time.time() - t0
    validity_opt = sum(p != "__invalid__" for p in preds_opt) / len(preds_opt)
    print(f"optimized: macro F1 = {f1_opt:.4f}, validity = {validity_opt:.4f}")
    append_result("s2", "mipro_optimization", calls=calls_opt, wallclock_s=wall_opt)
    append_result("s2", "optimized", f1=f1_opt, validity=validity_opt,
                  calls=len(lm.history) - calls_before, wallclock_s=wall_eval)

    stats = {
        "completed_without_crash": not crashed,
        "baseline_macro_f1": round(f1_base, 4),
        "optimized_macro_f1": round(f1_opt, 4),
        "delta": round(f1_opt - f1_base, 4),
        "optimization_wallclock_s": round(wall_opt, 1),
        "optimization_lm_calls": calls_opt,
    }
    with open(RESULTS / "s2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
