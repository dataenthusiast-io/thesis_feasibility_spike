"""Build dev/test slices for the spike.

Source: local HuggingFace Arrow cache of google/air_dialogue (train split),
already on disk under data/raw/ from the Transfer-Projekt. The Transfer-Projekt's
processed files (data/transfer_projekt/) are NOT reused: they contain full multi-turn
dialogues, not the single-utterance schema this spike fixes, and only 300 rows.

Derivation rule (spike spec fallback rule, applied to the local cache, with one
amendment discovered during the S1 dry-run):
- keep records with correct_sample == True and intent goal in {book, cancel, change}
- text = concatenation of the customer's first turns ("customer: " prefix stripped),
  accumulating turns until the text reaches MIN_WORDS words, up to MAX_TURNS turns
- skip dialogues with fewer than two customer turns or empty text
- stratified sample, seed 42: dev = 150 (50/class), test = 300 (100/class), disjoint

AMENDMENT RATIONALE: the spec's verbatim rule ("first two customer turns") produced
~12% of texts that carry no intent at all (turn 1 = greeting, turn 2 = self-
introduction, e.g. "Hi. I am Margaret Miller."; the intent only appears in turn 3).
Such texts are not human-classifiable and violate the fixed task definition
(single intent-bearing utterance). The amendment is mechanical and label-agnostic:
keep appending customer turns (max 3) until the text has >= 12 words. Evidence:
runs/s1_test_specrule_v0.jsonl / results/s1_stats_specrule_v0.json (the v0 run:
validity 88%, every failure a greeting-only text answered '{"intent": "none"}').
"""
import random

from datasets import Dataset

from common import DATA, LABELS, SEED, write_jsonl

ARROW = (DATA / "raw" / "google___air_dialogue" / "air_dialogue_data" / "0.0.0"
         / "dbdbe7bcef8d344bc3c68a05600f3d95917d6898" / "air_dialogue-train.arrow")

DEV_PER_CLASS = 50
TEST_PER_CLASS = 100
MIN_WORDS = 12
MAX_TURNS = 3


def derive_text(dialogue_turns):
    customer_turns = [t[len("customer:"):].strip()
                      for t in dialogue_turns if t.startswith("customer:")]
    if len(customer_turns) < 2:
        return None
    used, words = [], 0
    for turn in customer_turns[:MAX_TURNS]:
        used.append(turn)
        words += len(turn.split())
        if len(used) >= 2 and words >= MIN_WORDS:
            break
    text = " ".join(used).strip()
    return text or None


def main():
    ds = Dataset.from_file(str(ARROW))
    print(f"loaded {len(ds)} raw records")

    by_class = {label: [] for label in LABELS}
    rng = random.Random(SEED)
    # Reservoir-free approach: full pass, then sample. Full pass over 321k rows
    # is a one-time cost; keep it simple.
    needed = DEV_PER_CLASS + TEST_PER_CLASS
    for i, ex in enumerate(ds):
        if not bool(ex.get("correct_sample")):
            continue
        goal = (ex.get("intent") or {}).get("goal")
        if goal not in LABELS:
            continue
        text = derive_text(ex["dialogue"])
        if text is None:
            continue
        by_class[goal].append({"text": text, "intent": goal})
        # Early exit once each class has a generous pool to sample from.
        if all(len(v) >= needed * 20 for v in by_class.values()):
            print(f"pool filled after scanning {i + 1} records")
            break

    dev, test = [], []
    for label in LABELS:
        pool = by_class[label]
        print(f"class {label}: pool of {len(pool)}")
        picked = rng.sample(pool, needed)
        dev.extend(picked[:DEV_PER_CLASS])
        test.extend(picked[DEV_PER_CLASS:])

    rng.shuffle(dev)
    rng.shuffle(test)
    write_jsonl(DATA / "dev.jsonl", dev)
    write_jsonl(DATA / "test.jsonl", test)
    print(f"wrote data/dev.jsonl ({len(dev)}) and data/test.jsonl ({len(test)})")
    for name, rows in (("dev", dev), ("test", test)):
        counts = {l: sum(1 for r in rows if r["intent"] == l) for l in LABELS}
        print(f"  {name} class counts: {counts}")
    print("sample rows:")
    for r in test[:3]:
        print(" ", r)


if __name__ == "__main__":
    main()
