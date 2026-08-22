"""Spike 2: synthetic form-shift via LLM paraphrase (the prof's angle).

Instead of corrupting characters, rewrite each utterance as realistic drifted
input: casual chat register, indirect phrasing (describe the situation instead
of naming the action), light texting habits. Meaning and intent label must stay
inferable by a human agent.

Generator: llama3.1:8b (different model family than the qwen classifiers, to
reduce generator-classifier coupling). Temperature 0, seed 42: deterministic.

Outputs:
  data/test_synth.jsonl        AirDialogue test subsample (150, 50/class), shifted
  data/b77_test_synth.jsonl    Banking77 test (308), shifted
  data/b77_dev_synth.jsonl     Banking77 dev (154), shifted (for re-optimization)
"""
import json
import random

import ollama

from common import DATA, RESULTS, SEED, CallLogger, load_jsonl, write_jsonl

GEN_MODEL = "llama3.1:latest"

REWRITE_PROMPT = (
    "You rewrite customer messages to simulate how real customers write in "
    "casual chat support. Rewrite the message: informal, conversational, "
    "lowercase and contractions are fine, mild texting habits allowed. Where "
    "natural, be indirect: describe the situation or what happened instead of "
    "naming the requested action with its obvious keyword. The rewrite MUST "
    "express the same underlying request or issue, so a human support agent "
    "would still understand what the customer wants. Do not add new factual "
    "details. Do not answer the message. Output ONLY the rewritten message, "
    "nothing else."
)

client = ollama.Client()


def synthesize(text, logger):
    resp = client.chat(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": REWRITE_PROMPT},
                  {"role": "user", "content": text}],
        options={"temperature": 0, "seed": SEED},
    )
    out = resp["message"]["content"].strip().strip('"')
    # guard against runaway/chatty outputs only; colloquial indirect rewrites
    # of short queries are legitimately several times longer than the input
    ok = bool(out) and len(out) <= max(6 * len(text), 600)
    logger.log(prompt=text, response=out, ok=ok,
               prompt_tokens=resp.get("prompt_eval_count"),
               completion_tokens=resp.get("eval_count"))
    return out if ok else text, ok


def shift_file(src, dst, logger, limit=None, stratify_labels=None):
    rows = load_jsonl(src)
    if limit and len(rows) > limit:
        rng = random.Random(SEED)
        by_class = {}
        for r in rows:
            by_class.setdefault(r["intent"], []).append(r)
        per_class = limit // len(by_class)
        rows = [r for label in sorted(by_class)
                for r in rng.sample(by_class[label], per_class)]
        rng.shuffle(rows)
    out, fails = [], 0
    for i, r in enumerate(rows):
        text, ok = synthesize(r["text"], logger)
        fails += not ok
        out.append({"text": text, "intent": r["intent"]})
        if (i + 1) % 50 == 0:
            print(f"  {dst.name}: {i + 1}/{len(rows)}", flush=True)
    write_jsonl(dst, out)
    print(f"wrote {dst.name} ({len(out)}, {fails} generation fallbacks)")
    return rows, out


def main():
    logger = CallLogger("synth_generation")
    jobs = [
        (DATA / "test.jsonl", DATA / "test_synth.jsonl", 150),
        (DATA / "b77_test.jsonl", DATA / "b77_test_synth.jsonl", None),
        (DATA / "b77_dev.jsonl", DATA / "b77_dev_synth.jsonl", None),
    ]
    sanity = {}
    for src, dst, limit in jobs:
        orig, shifted = shift_file(src, dst, logger, limit=limit)
        rng = random.Random(SEED)
        idx = rng.sample(range(len(orig)), 10)
        pairs = [{"intent": orig[i]["intent"], "clean": orig[i]["text"],
                  "synthetic": shifted[i]["text"]} for i in idx]
        sanity[dst.name] = pairs
        print(f"\n=== sanity pairs {dst.name} ===")
        for p in pairs[:10]:
            print(f"[{p['intent']}] CLEAN: {p['clean']}")
            print(f"{' ' * len(p['intent'])}  SYNTH: {p['synthetic']}\n")
    with open(RESULTS / "synth_sanity_pairs.json", "w") as f:
        json.dump(sanity, f, indent=2, ensure_ascii=False)
    print(f"total generation calls: {logger.count}")


if __name__ == "__main__":
    main()
