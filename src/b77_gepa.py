"""Spike 3: GEPA probe on the two title-critical Banking77 cells.

Purpose: ground the thesis's optimizer-instrument decision (proposal Step 4/8
uses MIPROv2 while L4 cites GEPA, ICLR 2026, as outperforming it). Cells:
  1. Clean lift: GEPA on clean dev, evaluated on held-out clean test.
     Comparators: DSPy baseline 0.5035, MIPROv2 E1 0.5307 (+2.7).
  2. Recovery: GEPA on synthetically shifted dev, evaluated on shifted test.
     Comparators: frozen E1 0.3292, MIPROv2 re-opt 0.3390, manual 0.3696.

Reflection LM: llama3.1:8b (local availability; the thesis allocates a
DeepSeek-family proposer per proposal 3.6, so these numbers are design
evidence, not result cells). Task model unchanged: qwen2.5:3b.
"""
import json
import time

import dspy

from common import ARTIFACTS, DATA, OLLAMA_BASE, RESULTS, SEED, append_result
from b77_classify import get_labels
from b77_optimize import build_lm, build_signature, evaluate, make_examples
from s2_optimize import dump_history

REFLECTION_MODEL = "llama3.1:latest"


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA metric: score plus textual feedback for the reflection step."""
    try:
        ok = str(pred.intent).strip().lower() == gold.intent
    except Exception:
        ok = False
    feedback = ("Correct." if ok
                else f"Incorrect. The correct intent is '{gold.intent}'.")
    return dspy.Prediction(score=1.0 if ok else 0.0, feedback=feedback)


def run_gepa(trainset, sig, lm, run_name):
    reflection_lm = dspy.LM(f"ollama_chat/{REFLECTION_MODEL}",
                            api_base=OLLAMA_BASE, api_key="",
                            temperature=1.0, max_tokens=4000, cache=False)
    # Budget matched to the MIPROv2 E1 comparator on task-model calls
    # (E1 consumed 685; GEPA auto="light" would budget 996 rollouts, which
    # would confound an algorithm comparison with a bigger budget).
    optimizer = dspy.GEPA(metric=gepa_metric, max_metric_calls=700,
                          reflection_lm=reflection_lm, seed=SEED,
                          num_threads=4)
    calls_before = len(lm.history)
    t0 = time.time()
    optimized = optimizer.compile(dspy.Predict(sig), trainset=trainset)
    wall = time.time() - t0
    n_calls = len(lm.history) - calls_before
    dump_history(lm, run_name)
    return optimized, wall, n_calls


def main():
    labels = get_labels()
    lm = build_lm()
    dspy.configure(lm=lm)
    sig = build_signature(labels)
    stats = {}

    # Cell 1: clean lift
    print("GEPA cell 1: clean dev -> clean test")
    opt_clean, wall1, calls1 = run_gepa(make_examples(DATA / "b77_dev.jsonl"),
                                        sig, lm, "b77_gepa_clean_history")
    opt_clean.save(ARTIFACTS / "b77_gepa_program_clean.json")
    f1_clean, val_clean = evaluate(opt_clean, DATA / "b77_test.jsonl", labels)
    print(f"  GEPA clean: macro F1 = {f1_clean:.4f} (MIPRO E1: 0.5307, baseline: 0.5035)")
    append_result("b77_gepa", "optimized_clean", f1=f1_clean, validity=val_clean,
                  calls=calls1, wallclock_s=wall1)
    stats["gepa_clean_macro_f1"] = round(f1_clean, 4)
    stats["gepa_clean_opt_calls"] = calls1
    stats["gepa_clean_opt_wallclock_s"] = round(wall1, 1)

    # Cell 2: recovery on shifted data
    print("GEPA cell 2: shifted dev -> shifted test")
    opt_synth, wall2, calls2 = run_gepa(make_examples(DATA / "b77_dev_synth.jsonl"),
                                        sig, lm, "b77_gepa_synth_history")
    opt_synth.save(ARTIFACTS / "b77_gepa_program_synth.json")
    f1_synth, val_synth = evaluate(opt_synth, DATA / "b77_test_synth.jsonl", labels)
    print(f"  GEPA recovery: macro F1 = {f1_synth:.4f} "
          f"(frozen E1: 0.3292, MIPRO reopt: 0.3390, manual: 0.3696)")
    append_result("b77_gepa", "reoptimized_on_synth", f1=f1_synth,
                  validity=val_synth, calls=calls2, wallclock_s=wall2)
    stats["gepa_recovery_macro_f1"] = round(f1_synth, 4)
    stats["gepa_recovery_opt_calls"] = calls2
    stats["gepa_recovery_opt_wallclock_s"] = round(wall2, 1)

    with open(RESULTS / "b77_gepa_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
