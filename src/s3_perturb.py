"""S3 — noise pipeline v0 (confirms A3).

perturb(text, severity) composes three perturbation types whose intensity
scales with severity N1 < N2 < N3:
  1. character-level typos (keyboard-adjacent substitute / swap / drop / duplicate)
  2. ASR-style errors (lowercase, strip punctuation, homophone substitutions,
     occasional word drops)
  3. truncation (cut the final 10-30% of the utterance)

Deterministic: each (seed, severity, index) gets its own RNG, so output is
independent of processing order.

Run: writes data/test_n{1,2,3}.jsonl, prints 10 sanity-check pairs per severity,
then scores the unchanged S1 classifier on clean vs N1/N2/N3.
"""
import json
import random
import string

from common import DATA, RESULTS, SEED, load_jsonl, write_jsonl

# intensity parameters per severity
PARAMS = {
    "n1": dict(typo_p=0.02, asr_p=0.5, homophone_p=0.10, drop_p=0.02,
               trunc_p=0.25, trunc_frac=(0.10, 0.15)),
    "n2": dict(typo_p=0.05, asr_p=0.8, homophone_p=0.25, drop_p=0.05,
               trunc_p=0.60, trunc_frac=(0.10, 0.20)),
    "n3": dict(typo_p=0.09, asr_p=1.0, homophone_p=0.40, drop_p=0.10,
               trunc_p=1.00, trunc_frac=(0.20, 0.30)),
}

KEYBOARD_ADJACENT = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

HOMOPHONES = {
    "to": "two", "too": "to", "two": "too", "for": "four", "four": "for",
    "flight": "fright", "fare": "fair", "plane": "plain", "week": "weak",
    "meet": "meat", "no": "know", "new": "knew", "one": "won", "wait": "weight",
    "buy": "by", "by": "buy", "hi": "high", "be": "bee", "so": "sew",
    "would": "wood", "there": "their", "your": "you're", "here": "hear",
}


def _char_typos(text, rng, p):
    out = []
    for ch in text:
        if ch.isalpha() and rng.random() < p:
            op = rng.choice(["sub", "drop", "dup", "swap"])
            low = ch.lower()
            if op == "sub" and low in KEYBOARD_ADJACENT:
                out.append(rng.choice(KEYBOARD_ADJACENT[low]))
            elif op == "drop":
                pass
            elif op == "dup":
                out.append(ch + ch)
            else:  # swap with previous char
                if out:
                    prev = out.pop()
                    out.extend([ch, prev])
                else:
                    out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _asr(text, rng, homophone_p, drop_p):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = []
    for w in text.split():
        if rng.random() < drop_p:
            continue
        if w in HOMOPHONES and rng.random() < homophone_p:
            w = HOMOPHONES[w]
        words.append(w)
    return " ".join(words)


def _truncate(text, rng, frac_range):
    frac = rng.uniform(*frac_range)
    keep = max(1, int(len(text) * (1 - frac)))
    return text[:keep].rstrip()


def perturb(text, severity, seed=SEED, idx=0):
    p = PARAMS[severity]
    rng = random.Random(f"{seed}:{severity}:{idx}")
    if rng.random() < p["asr_p"]:
        text = _asr(text, rng, p["homophone_p"], p["drop_p"])
    text = _char_typos(text, rng, p["typo_p"])
    if rng.random() < p["trunc_p"]:
        text = _truncate(text, rng, p["trunc_frac"])
    return text


def build_sets(src_path=None, seed=SEED, prefix="test"):
    src_path = src_path or DATA / "test.jsonl"
    rows = load_jsonl(src_path)
    paths = {}
    for sev in ("n1", "n2", "n3"):
        out = [{"text": perturb(r["text"], sev, seed=seed, idx=i),
                "intent": r["intent"]} for i, r in enumerate(rows)]
        path = DATA / f"{prefix}_{sev}.jsonl"
        write_jsonl(path, out)
        paths[sev] = path
    return rows, paths


def main():
    rows, paths = build_sets()
    print("wrote", ", ".join(str(p) for p in paths.values()))

    # manual sanity check: 10 random pairs per severity
    rng = random.Random(SEED)
    sample_idx = rng.sample(range(len(rows)), 10)
    sanity = {}
    for sev in ("n1", "n2", "n3"):
        perturbed = load_jsonl(paths[sev])
        pairs = [{"intent": rows[i]["intent"], "clean": rows[i]["text"],
                  "perturbed": perturbed[i]["text"]} for i in sample_idx]
        sanity[sev] = pairs
        print(f"\n=== {sev.upper()} sanity pairs ===")
        for pr in pairs:
            print(f"[{pr['intent']}] CLEAN: {pr['clean']}")
            print(f"          PERT : {pr['perturbed']}\n")
    with open(RESULTS / "s3_sanity_pairs.json", "w") as f:
        json.dump(sanity, f, indent=2, ensure_ascii=False)

    # score the unchanged S1 classifier on each severity
    from s1_classify import classify_file
    all_stats = []
    for sev in ("n1", "n2", "n3"):
        stats, _, _ = classify_file(paths[sev], f"s3_test_{sev}", sev)
        print(json.dumps(stats, indent=2))
        all_stats.append(stats)
    with open(RESULTS / "s3_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    main()
