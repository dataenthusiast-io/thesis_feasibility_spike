"""Spike 2, Q-D/Q-E: Banking77 under synthetic shift + recovery.

Q-D: manual prompt and frozen-optimized program on b77_test_synth (degradation).
Q-E: re-run MIPROv2 on the synthetically shifted dev set, evaluate on
b77_test_synth (recovery). Requires b77_optimize.py and synth_perturb.py ran.
"""
import json
import time

import dspy

from common import ARTIFACTS, DATA, RESULTS, append_result
from b77_classify import get_labels, classify_file
from b77_optimize import build_lm, build_signature, evaluate, make_examples, run_mipro


def main():
    labels = get_labels()

    # Q-D part 1: manual prompt on synthetic shift (degradation vs clean_3b)
    classify_file("qwen2.5:3b", DATA / "b77_test_synth.jsonl",
                  "b77_synth_3b", "synth_3b")

    lm = build_lm()
    dspy.configure(lm=lm)
    sig = build_signature(labels)

    # Q-D part 2: frozen clean-optimized program (E1: instruction-only search,
    # 8B proposer) on synthetic shift
    frozen = dspy.Predict(sig)
    frozen.load(ARTIFACTS / "b77_optimized_program_e1.json")
    t0 = time.time()
    f1_frozen, val_frozen = evaluate(frozen, DATA / "b77_test_synth.jsonl", labels)
    print(f"frozen optimized on b77_test_synth: macro F1 = {f1_frozen:.4f}")
    append_result("b77_recover", "frozen_opt_on_synth", f1=f1_frozen,
                  validity=val_frozen, calls=len(lm.history),
                  wallclock_s=time.time() - t0)

    # Q-E: re-optimize on synthetically shifted dev, evaluate on shifted test
    print("re-optimizing on b77_dev_synth (MIPROv2 auto=light) ...")
    trainset = make_examples(DATA / "b77_dev_synth.jsonl")
    reopt, wall_opt, calls_opt = run_mipro(trainset, dspy.Predict(sig), lm,
                                           "b77_reopt_history",
                                           prompt_model="llama3.1:latest",
                                           zero_shot=True)
    reopt.save(ARTIFACTS / "b77_reoptimized_program_synth.json")
    append_result("b77_recover", "reoptimization", calls=calls_opt,
                  wallclock_s=wall_opt)

    calls_before = len(lm.history)
    t0 = time.time()
    f1_reopt, val_reopt = evaluate(reopt, DATA / "b77_test_synth.jsonl", labels)
    print(f"re-optimized on b77_test_synth: macro F1 = {f1_reopt:.4f}")
    append_result("b77_recover", "reoptimized_on_synth", f1=f1_reopt,
                  validity=val_reopt, calls=len(lm.history) - calls_before,
                  wallclock_s=time.time() - t0)

    stats = {
        "frozen_optimized_on_synth": round(f1_frozen, 4),
        "reoptimized_on_synth": round(f1_reopt, 4),
        "recovery_delta": round(f1_reopt - f1_frozen, 4),
        "reoptimization_wallclock_s": round(wall_opt, 1),
        "reoptimization_lm_calls": calls_opt,
    }
    with open(RESULTS / "b77_recover_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
