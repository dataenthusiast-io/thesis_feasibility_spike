"""Spike 2: manual-prompt classifier for Banking77 (77 labels).

Same design as s1_classify but with the 77-label closed set listed in the
system prompt. Parameterized by model so the 1B/3B headroom question (Q-A)
runs through identical code.
"""
import json
import re
import time

import ollama

from common import DATA, RESULTS, CallLogger, append_result, load_jsonl

client = ollama.Client()


def get_labels():
    return (DATA / "b77_labels.txt").read_text().split()


def build_system_prompt(labels):
    return (
        "You are an intent classifier for a banking customer-support system. "
        "Classify the customer's message into exactly one intent from this "
        "list:\n" + "\n".join(f"- {l}" for l in labels) +
        '\n\nAnswer with JSON only, exactly in this form: {"intent": "<label>"}'
    )


def parse_intent(raw, labels):
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    intent = obj.get("intent")
    if isinstance(intent, str) and intent.strip().lower() in labels:
        return intent.strip().lower()
    return None


def classify_file(model, path, run_name, condition):
    labels = get_labels()
    system_prompt = build_system_prompt(labels)
    rows = load_jsonl(path)
    logger = CallLogger(run_name)
    t0 = time.time()
    preds, valid, calls = [], 0, 0
    for i, r in enumerate(rows):
        label = None
        for attempt in (1, 2):
            resp = client.chat(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": r["text"]}],
                options={"temperature": 0, "seed": 42},
            )
            calls += 1
            raw = resp["message"]["content"]
            label = parse_intent(raw, labels)
            logger.log(attempt=attempt, prompt=r["text"], response=raw[:2000],
                       parsed=label,
                       prompt_tokens=resp.get("prompt_eval_count"),
                       completion_tokens=resp.get("eval_count"))
            if label is not None:
                break
        preds.append(label if label else "__invalid__")
        valid += label is not None
        if (i + 1) % 50 == 0:
            print(f"  {condition}: {i + 1}/{len(rows)}", flush=True)
    wall = time.time() - t0
    golds = [r["intent"] for r in rows]
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(golds, preds, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(golds, preds)
    stats = {
        "condition": condition, "model": model, "n": len(rows),
        "macro_f1": round(f1, 4), "accuracy": round(acc, 4),
        "validity_after_retry": round(valid / len(rows), 4),
        "lm_calls": calls, "wallclock_s": round(wall, 1),
    }
    append_result("b77_manual", condition, f1=f1, validity=valid / len(rows),
                  calls=calls, wallclock_s=wall)
    print(json.dumps(stats, indent=2))
    return stats


def main():
    all_stats = [
        classify_file("qwen2.5:3b", DATA / "b77_test.jsonl", "b77_clean_3b", "clean_3b"),
        classify_file("llama3.2:1b", DATA / "b77_test.jsonl", "b77_clean_1b", "clean_1b"),
    ]
    with open(RESULTS / "b77_manual_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    main()
