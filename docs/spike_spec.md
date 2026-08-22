# Spike Spec: Thesis Feasibility MVP

## Context (read first)

This spike de-risks a master thesis experiment. The thesis investigates whether automatic prompt optimization (DSPy/MIPROv2) lets small local language models substitute for frontier API models in conversational intent classification, and whether a closed re-optimization loop preserves quality when the input data degrades (distribution shift). The planned thesis design is a factorial grid (model size x prompt regime x clean/noisy data); before committing to it, four technical assumptions must be confirmed cheaply. A prior project already achieved macro F1 = 0.993 on this task with a frontier API model and a hand-written prompt, so near-ceiling scores on clean data are expected and unremarkable; the spike is about whether the machinery works, not about maximizing scores.

**Task definition (fixed, do not improvise):** single-utterance intent classification. Input: one customer utterance (string). Output: exactly one label from the closed set `["book", "cancel", "change"]` (these exact lowercase strings; they define the JSON schema, the DSPy signature, and the metric). Data schema for all files: JSONL with fields `text` (str) and `intent` (str, one of the three labels).

**Purpose.** De-risk the master thesis experiment before literature phase and Exposé. This spike confirms three load-bearing assumptions with minimal implementation. It produces throwaway-quality code but keep-quality evidence: a short report with numbers and a go/no-go per assumption.

**The three assumptions to confirm:**
- **A1:** A small local LM (via Ollama) can classify conversational utterances into intents with valid structured JSON output at acceptable reliability.
- **A2:** DSPy with MIPROv2 runs end-to-end against a macro F1 metric on this task and completes an optimization within a small, fixed budget.
- **A3:** Parameterized noise injection measurably degrades classification quality (creating the headroom the thesis design depends on).
- **A4 (stretch, only if A1-A3 pass quickly):** Re-optimizing on noise-perturbed dev data recovers a measurable share of lost quality.

**Non-goals.** No full grid, no frontier API model, no multiple SLMs, no cost model, no latency benchmarking, no statistical testing, no polished code. Anything not needed for A1-A4 is out of scope.

## Environment

- Python 3.11+, uv or venv
- Dependencies: `dspy` (latest stable), `ollama` Python client, `scikit-learn` (metrics), `pandas`
- Ollama installed locally with model `qwen2.5:3b` (instruct-tuned by default; primary). Fallback if hardware-constrained: `llama3.2:1b`. Pull at setup, verify with a smoke call.
- DSPy connects to Ollama via its OpenAI-compatible endpoint (`http://localhost:11434/v1`) or DSPy's native Ollama client, whichever the installed DSPy version supports; verify with a one-call smoke test before S2. Set temperature 0 for classification calls.
- Reproducibility: fixed random seed (42) everywhere; log all LM calls (prompt, response, token counts if available) to `runs/` as JSONL.

## Data

- Source: AirDialogue-derived intent classification set from the Transfer-Projekt (user provides JSONL/CSV matching the schema above; ask for it before falling back). Fallback: load AirDialogue from Hugging Face (`google/air_dialogue`); each dialogue's ground-truth intent is in its metadata (`customer_intention` / goal field: book, change, cancel). Derivation rule for the fallback: take the concatenation of the customer's first two turns as `text`, the metadata goal as `intent`, skip dialogues with missing or other goals, and document the rule in `prepare_data.py`.
- Slice sizes (stratified by class, seed 42):
  - `dev`: 150 examples (optimizer training/validation material)
  - `test`: 300 examples (held out, never touched by optimization)
- Persist slices to `data/dev.jsonl` and `data/test.jsonl` so all spike stages use identical data.

## Stage S1 — Local SLM structured classification (confirms A1)

**Build:** a minimal classifier: system/user prompt asking the model to classify `text` into exactly one of the 3 intents and answer as JSON `{"intent": "<label>"}`. One retry on invalid JSON. Run over `test`.

**Measure and report:**
- JSON validity rate (share of responses parseable and containing a legal label, counting retries separately)
- Macro F1 on `test`
- Wall-clock time per 100 examples

**Acceptance (go):** validity ≥ 95% after retry; macro F1 meaningfully above chance (> 0.6 expected given the Transfer-Projekt's 0.993 with a frontier model; record whatever results, the number itself is a finding). If validity < 95%: try constrained decoding via Ollama's `format: json` option before declaring no-go.

## Stage S2 — DSPy + MIPROv2 end-to-end (confirms A2)

**Build:**
- DSPy signature: `text -> intent` (with class labels in the signature docstring/description)
- Module: `dspy.Predict` or `dspy.ChainOfThought` (start with Predict; simpler)
- LM: the same Ollama model via DSPy's Ollama/OpenAI-compatible client
- Metric: exact-match on intent label (per-example), macro F1 computed separately for reporting
- Optimizer: `MIPROv2` with the smallest sensible budget (`auto="light"`), optimizing on `dev` (use an internal split if MIPROv2 requires train/val)

**Measure and report:**
- Optimization completes without crash: yes/no, wall-clock time, number of LM calls consumed
- Macro F1 on `test`: unoptimized DSPy program vs. optimized program (same model, same data)
- The optimized prompt(s): dump to `artifacts/optimized_program.json` for inspection

**Acceptance (go):** the run completes within the budget and produces an inspectable optimized program. Improvement over baseline is expected but NOT required for go; a flat result on 3 easy classes is consistent with the ceiling effect and itself informative. No-go only if DSPy+Ollama integration is structurally broken (crashes, cannot target metric, runaway call counts).

## Stage S3 — Noise pipeline v0 (confirms A3)

**Build:** a perturbation module `perturb(text, severity)` with severity in {N1, N2, N3}, composing three perturbation types (probabilities scale with severity):
1. Character-level typos: swap/drop/duplicate characters (keyboard-adjacent substitutions where easy)
2. ASR-style errors: lowercase everything, strip punctuation, homophone-ish substitutions from a small hardcoded map (e.g. "to/two", "flight/fright"-style confusions), occasional word drops
3. Truncation: cut the final 10-30% of the utterance

Deterministic under seed. Apply to `test` producing `test_n1/2/3.jsonl`. Manual sanity check: print 10 random pairs per severity; perturbed text must remain human-classifiable (label-preserving). Note any examples where the label becomes ambiguous.

**Measure and report:** macro F1 of the S1 classifier (unchanged, manual prompt) on clean vs. N1 vs. N2 vs. N3. 

**Acceptance (go):** monotonic or near-monotonic degradation with severity, and a total clean-to-N3 drop of at least ~5 F1 points. If the model is robust and barely degrades, escalate severity parameters once; if still flat, that is a real design risk to flag (the thesis would need harsher shift types or a harder corpus), which is exactly what the spike exists to discover.

## Stage S4 — Recovery (stretch, confirms A4)

Only if S1-S3 pass with time left: perturb `dev` at N2 with a different seed than test, re-run MIPROv2 (`auto="light"`) on the perturbed dev set, evaluate the re-optimized program on `test_n2`. Report: F1 recovered vs. the frozen optimized program from S2 on `test_n2`, plus re-optimization call count. Any positive delta is a strong signal; no delta is a documented risk, not a spike failure.

## Deliverables

```
spike/
  data/            # dev/test slices + perturbed variants
  src/
    prepare_data.py
    s1_classify.py
    s2_optimize.py
    s3_perturb.py
    s4_recover.py   # stretch
  artifacts/       # optimized programs, prompts
  runs/            # JSONL call logs
  results/
    results.csv    # one row per (stage, condition): f1, validity, calls, wallclock
    SPIKE_REPORT.md
```

**SPIKE_REPORT.md must contain:** per assumption A1-A4 a go/no-go with the supporting numbers, the clean vs. N1-N3 degradation table, 5 example perturbed utterances per severity, total LM calls and wall-clock per stage, and a short "risks discovered" section (anything that should change the thesis design).

## Constraints for implementation

- Keep every stage runnable independently (`python src/s1_classify.py`) and idempotent.
- Hard cap: no stage may exceed ~2h wall-clock on consumer hardware or ~5,000 LM calls; abort and report if exceeded (that itself is a feasibility finding).
- No external APIs, no paid calls; everything local.
- Prefer boring code over abstractions; this is a spike.
