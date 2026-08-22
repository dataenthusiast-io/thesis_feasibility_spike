"""Spike 2: build Banking77 dev/test slices.

Source: official PolyAI CSVs (data/raw/banking77/{train,test}.csv).
dev = 2/class = 154 (from official train split, optimizer material)
test = 4/class = 308 (from official test split, held out)
Stratified, seed 42. Schema: {text, intent} with intent = the 77 label strings.
"""
import random

import pandas as pd

from common import DATA, SEED, write_jsonl

DEV_PER_CLASS = 2
TEST_PER_CLASS = 4


def sample_split(csv_path, per_class, rng):
    df = pd.read_csv(csv_path)
    rows = []
    for label, grp in df.groupby("category"):
        texts = sorted(grp["text"].tolist())  # sort for determinism
        picked = rng.sample(texts, per_class)
        rows.extend({"text": t, "intent": label} for t in picked)
    rng.shuffle(rows)
    return rows


def main():
    rng = random.Random(SEED)
    dev = sample_split(DATA / "raw" / "banking77" / "train.csv", DEV_PER_CLASS, rng)
    test = sample_split(DATA / "raw" / "banking77" / "test.csv", TEST_PER_CLASS, rng)
    labels = sorted({r["intent"] for r in dev})
    assert len(labels) == 77, len(labels)
    write_jsonl(DATA / "b77_dev.jsonl", dev)
    write_jsonl(DATA / "b77_test.jsonl", test)
    with open(DATA / "b77_labels.txt", "w") as f:
        f.write("\n".join(labels))
    print(f"wrote b77_dev.jsonl ({len(dev)}), b77_test.jsonl ({len(test)}), {len(labels)} labels")
    for r in test[:3]:
        print(" ", r)


if __name__ == "__main__":
    main()
