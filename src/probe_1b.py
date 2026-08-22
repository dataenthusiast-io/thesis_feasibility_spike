"""Tier probe: llama3.2:1b on AirDialogue clean and N3 test sets.

Ran post-spike-1 to answer whether the clean-data ceiling holds across the
SLM range (it does: 0.9746 clean) and how the 1B tier degrades under shift
(0.6722 at N3, validity 88.7%). Produces runs/probe_clean_1b.jsonl and
runs/probe_n3_1b.jsonl; results cited in the central report.
"""
import json

import s1_classify
from common import DATA

s1_classify.MODEL = "llama3.2:1b"


def main():
    for path, cond in [(DATA / "test.jsonl", "clean_1b"),
                       (DATA / "test_n3.jsonl", "n3_1b")]:
        stats, _, _ = s1_classify.classify_file(path, f"probe_{cond}", cond)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
