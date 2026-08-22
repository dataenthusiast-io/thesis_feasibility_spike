"""S4 (stretch) — re-optimization recovery on noisy data (confirms A4).

Perturb dev at N2 with a DIFFERENT seed (SEED+1) than the test perturbation,
re-run MIPROv2 auto=light on that noisy dev set, then compare on test_n2:
frozen S2-optimized program vs re-optimized program.
"""
import json
import time

import dspy

from common import (ARTIFACTS, DATA, RESULTS, SEED, append_result,
                    load_jsonl, macro_f1)
from s2_optimize import (IntentSignature, build_lm, evaluate, make_examples,
                         run_mipro)
from s3_perturb import perturb
from common import write_jsonl


def main():
    lm = build_lm()
    dspy.configure(lm=lm)

    # noisy dev set, different seed than test perturbation
    dev_rows = load_jsonl(DATA / "dev.jsonl")
    noisy_dev = [{"text": perturb(r["text"], "n2", seed=SEED + 1, idx=i),
                  "intent": r["intent"]} for i, r in enumerate(dev_rows)]
    write_jsonl(DATA / "dev_n2.jsonl", noisy_dev)
    trainset = make_examples(DATA / "dev_n2.jsonl")

    # frozen S2 program on test_n2
    frozen = dspy.Predict(IntentSignature)
    frozen.load(ARTIFACTS / "optimized_program.json")
    f1_frozen, _, preds_f = evaluate(frozen, DATA / "test_n2.jsonl")
    validity_f = sum(p != "__invalid__" for p in preds_f) / len(preds_f)
    print(f"frozen S2-optimized on test_n2: macro F1 = {f1_frozen:.4f}")
    append_result("s4", "frozen_s2_on_n2", f1=f1_frozen, validity=validity_f,
                  calls=len(preds_f))

    print("re-optimizing on noisy dev (MIPROv2 auto=light) ...")
    reopt, wall_opt, calls_opt = run_mipro(trainset, lm, "s4_mipro_history")
    reopt.save(ARTIFACTS / "reoptimized_program_n2.json")
    append_result("s4", "reoptimization", calls=calls_opt, wallclock_s=wall_opt)

    t0 = time.time()
    f1_reopt, _, preds_r = evaluate(reopt, DATA / "test_n2.jsonl")
    validity_r = sum(p != "__invalid__" for p in preds_r) / len(preds_r)
    print(f"re-optimized on test_n2: macro F1 = {f1_reopt:.4f}")
    append_result("s4", "reoptimized_on_n2", f1=f1_reopt, validity=validity_r,
                  calls=len(preds_r), wallclock_s=time.time() - t0)

    stats = {
        "frozen_s2_on_test_n2": round(f1_frozen, 4),
        "reoptimized_on_test_n2": round(f1_reopt, 4),
        "recovery_delta": round(f1_reopt - f1_frozen, 4),
        "reoptimization_wallclock_s": round(wall_opt, 1),
        "reoptimization_lm_calls": calls_opt,
    }
    with open(RESULTS / "s4_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
