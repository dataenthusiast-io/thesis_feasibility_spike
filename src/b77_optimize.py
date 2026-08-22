"""Spike 2, Q-B: does MIPROv2 produce a measurable lift on clean Banking77?

Baseline dspy.Predict vs MIPROv2(auto=light)-optimized, qwen2.5:3b, evaluated
on the held-out Banking77 test slice. This is the title-critical test.
"""
import json
import time
from typing import Literal

import dspy

from common import (ARTIFACTS, DATA, OLLAMA_BASE, RESULTS, RUNS, SEED,
                    append_result, load_jsonl)
from b77_classify import get_labels
from s2_optimize import dump_history

MODEL = "qwen2.5:3b"


def build_lm():
    return dspy.LM(f"ollama_chat/{MODEL}", api_base=OLLAMA_BASE, api_key="",
                   temperature=0, max_tokens=300, cache=False)


def build_signature(labels):
    intent_type = Literal[tuple(labels)]

    class B77Signature(dspy.Signature):
        """Classify the banking customer's message into exactly one of the 77
        support intents."""
        text: str = dspy.InputField(desc="one customer message")
        intent: intent_type = dspy.OutputField()

    return B77Signature


def make_examples(path):
    return [dspy.Example(text=r["text"], intent=r["intent"]).with_inputs("text")
            for r in load_jsonl(path)]


def metric(example, pred, trace=None):
    try:
        return str(pred.intent).strip().lower() == example.intent
    except Exception:
        return False


def evaluate(program, path, labels):
    rows = load_jsonl(path)
    golds, preds = [], []
    for r in rows:
        try:
            out = program(text=r["text"])
            label = str(out.intent).strip().lower()
        except Exception:
            label = "__invalid__"
        golds.append(r["intent"])
        preds.append(label if label in labels else "__invalid__")
    from sklearn.metrics import f1_score
    f1 = f1_score(golds, preds, labels=labels, average="macro", zero_division=0)
    validity = sum(p != "__invalid__" for p in preds) / len(preds)
    return f1, validity


def run_mipro(trainset, program, lm, run_name, prompt_model=None, zero_shot=False):
    kwargs = dict(metric=metric, auto="light", seed=SEED,
                  num_threads=4, max_errors=200)
    if prompt_model:
        kwargs["prompt_model"] = dspy.LM(f"ollama_chat/{prompt_model}",
                                         api_base=OLLAMA_BASE, api_key="",
                                         temperature=0.7, max_tokens=1000,
                                         cache=False)
    if zero_shot:
        # 77 classes: a handful of demos cannot cover the label space and
        # empirically biases the model toward demoed classes; search
        # instructions only.
        kwargs["max_bootstrapped_demos"] = 0
        kwargs["max_labeled_demos"] = 0
    optimizer = dspy.MIPROv2(**kwargs)
    calls_before = len(lm.history)
    t0 = time.time()
    optimized = optimizer.compile(program, trainset=trainset,
                                  requires_permission_to_run=False)
    wall = time.time() - t0
    n_calls = len(lm.history) - calls_before
    dump_history(lm, run_name)
    return optimized, wall, n_calls


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for artifacts/results rows")
    ap.add_argument("--prompt-model", default=None,
                    help="separate proposer model for MIPRO instruction candidates")
    ap.add_argument("--zero-shot", action="store_true",
                    help="instruction-only search, no few-shot demos")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    labels = get_labels()
    lm = build_lm()
    dspy.configure(lm=lm)
    sig = build_signature(labels)
    trainset = make_examples(DATA / "b77_dev.jsonl")

    baseline = dspy.Predict(sig)
    t0 = time.time()
    f1_base, val_base = evaluate(baseline, DATA / "b77_test.jsonl", labels)
    print(f"baseline Predict on b77 test: macro F1 = {f1_base:.4f}, validity = {val_base:.4f}")
    append_result("b77_dspy", f"baseline_unoptimized{tag}", f1=f1_base, validity=val_base,
                  calls=len(lm.history), wallclock_s=time.time() - t0)

    print(f"running MIPROv2 auto=light on clean b77 dev (tag={args.tag or 'default'}) ...")
    optimized, wall_opt, calls_opt = run_mipro(trainset, dspy.Predict(sig), lm,
                                               f"b77_mipro_history{tag}",
                                               prompt_model=args.prompt_model,
                                               zero_shot=args.zero_shot)
    optimized.save(ARTIFACTS / f"b77_optimized_program{tag}.json")
    append_result("b77_dspy", f"mipro_optimization{tag}", calls=calls_opt, wallclock_s=wall_opt)

    calls_before = len(lm.history)
    t0 = time.time()
    f1_opt, val_opt = evaluate(optimized, DATA / "b77_test.jsonl", labels)
    print(f"optimized on b77 test: macro F1 = {f1_opt:.4f}, validity = {val_opt:.4f}")
    append_result("b77_dspy", f"optimized{tag}", f1=f1_opt, validity=val_opt,
                  calls=len(lm.history) - calls_before, wallclock_s=time.time() - t0)

    stats = {
        "baseline_macro_f1": round(f1_base, 4),
        "baseline_validity": round(val_base, 4),
        "optimized_macro_f1": round(f1_opt, 4),
        "optimized_validity": round(val_opt, 4),
        "lift": round(f1_opt - f1_base, 4),
        "optimization_wallclock_s": round(wall_opt, 1),
        "optimization_lm_calls": calls_opt,
    }
    with open(RESULTS / f"b77_dspy_stats{tag}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
