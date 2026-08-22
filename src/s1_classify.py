"""S1 — local SLM structured classification over test (confirms A1).

Also importable: classify_file() is reused by s3_perturb.py to score
perturbed test sets with the identical manual-prompt classifier.
"""
import json
import re
import time

import ollama

from common import (LABELS, MODEL, RESULTS, CallLogger, append_result,
                    load_jsonl, macro_f1, DATA)

SYSTEM_PROMPT = (
    "You are an intent classifier for an airline customer-service system. "
    "Classify the customer's utterance into exactly one intent: "
    '"book" (wants to book a new flight), '
    '"cancel" (wants to cancel an existing reservation), or '
    '"change" (wants to change an existing reservation). '
    'Answer with JSON only, exactly in this form: {"intent": "<label>"}'
)

client = ollama.Client()


def parse_intent(raw):
    """Return a legal label or None. Tolerates code fences / surrounding text."""
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    intent = obj.get("intent")
    if isinstance(intent, str) and intent.strip().lower() in LABELS:
        return intent.strip().lower()
    return None


def classify_one(text, logger, use_format_json=False):
    """Returns (label_or_None, first_try_valid, n_calls)."""
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": text}],
        options={"temperature": 0, "seed": 42},
    )
    if use_format_json:
        kwargs["format"] = "json"
    calls = 0
    for attempt in (1, 2):  # one retry on invalid JSON
        resp = client.chat(**kwargs)
        calls += 1
        raw = resp["message"]["content"]
        label = parse_intent(raw)
        logger.log(stage="s1", attempt=attempt, prompt=text, response=raw,
                   parsed=label,
                   prompt_tokens=resp.get("prompt_eval_count"),
                   completion_tokens=resp.get("eval_count"))
        if label is not None:
            return label, attempt == 1, calls
    return None, False, calls


def classify_file(path, run_name, condition, use_format_json=False):
    rows = load_jsonl(path)
    logger = CallLogger(run_name)
    t0 = time.time()
    preds, first_valid, valid_after_retry, total_calls = [], 0, 0, 0
    for i, r in enumerate(rows):
        label, first_ok, calls = classify_one(r["text"], logger, use_format_json)
        preds.append(label if label else "__invalid__")
        first_valid += first_ok
        valid_after_retry += label is not None
        total_calls += calls
        if (i + 1) % 50 == 0:
            print(f"  {condition}: {i + 1}/{len(rows)}", flush=True)
    wall = time.time() - t0
    golds = [r["intent"] for r in rows]
    f1 = macro_f1(golds, preds)
    validity = valid_after_retry / len(rows)
    stats = {
        "condition": condition,
        "n": len(rows),
        "macro_f1": round(f1, 4),
        "validity_first_try": round(first_valid / len(rows), 4),
        "validity_after_retry": round(validity, 4),
        "lm_calls": total_calls,
        "wallclock_s": round(wall, 1),
        "s_per_100": round(wall / len(rows) * 100, 1),
    }
    append_result("s1", condition, f1=f1, validity=validity,
                  calls=total_calls, wallclock_s=wall)
    return stats, golds, preds


def main():
    stats, golds, preds = classify_file(DATA / "test.jsonl", "s1_test", "clean")
    print(json.dumps(stats, indent=2))
    from sklearn.metrics import classification_report
    print(classification_report(golds, preds, labels=LABELS, zero_division=0))
    with open(RESULTS / "s1_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    if stats["validity_after_retry"] < 0.95:
        print("validity < 95% — rerunning with format=json (constrained decoding)")
        stats2, golds2, preds2 = classify_file(
            DATA / "test.jsonl", "s1_test_formatjson", "clean_formatjson",
            use_format_json=True)
        print(json.dumps(stats2, indent=2))
        with open(RESULTS / "s1_stats_formatjson.json", "w") as f:
            json.dump(stats2, f, indent=2)


if __name__ == "__main__":
    main()
