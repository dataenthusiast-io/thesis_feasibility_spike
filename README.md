# Thesis Feasibility Spikes: Self-Improving LM Pipelines under Distribution Shift

Feasibility spikes for the master thesis *Self-Improving Language Model Pipelines for Conversational Intent Classification under Distribution Shift* (FOM). Two spike runs, executed 2026-08-21, validate the technical assumptions of the planned factorial experiment before the literature phase and Expose.

**Start here: [results/SPIKE_REPORT_CENTRAL.md](results/SPIKE_REPORT_CENTRAL.md)** contains the consolidated verdicts, all metrics, the decision history, and the resulting design decisions.

## What was tested

| Run | Question | Outcome |
|---|---|---|
| 1 (AirDialogue) | A1: local SLM + reliable structured JSON output | GO (F1 0.997, 100% validity) |
| 1 | A2: DSPy + MIPROv2 end to end within budget | GO |
| 1 | A3: parametric noise degrades quality measurably | GO (monotonic, -9.8 pts at N3) |
| 1 | A4: re-optimization recovers after shift | No recovery at N2 corruption (diagnosed: no headroom) |
| 2 (Banking77) | Headroom for the thesis grid | Yes (0.52 at 3B vs 0.997 on AirDialogue) |
| 2 | Measurable DSPy/MIPROv2 lift | Yes (+2.7 F1 pts held-out; +0.0 with naive optimizer config) |
| 2 | LLM-synthetic form shift as drift instrument | Yes (-11 to -20 pts, realistic; label-drift caveat) |
| 2 | Recovery under form shift | First positive delta (+1.0 pts, single seed) |

Decisions taken on this basis: the thesis corpus is switched to Banking77, LLM-synthetic form shift is the primary drift instrument (with a label-validation protocol), and optimizer configuration (proposer model, demo policy) is an explicit experimental variable.

## Repository structure

```
README.md                    this file
docs/
  spike_spec.md              spec for run 1 (assumptions A1 to A4, stages S1 to S4)
  thesis_proposal_draft.md   current thesis proposal draft
  thesis_context_dump.md     decision log / full context for the thesis design
  transfer_projekt_implementation.md   prior project (Transfer-Projekt) reference
src/
  common.py                  shared helpers (paths, labels, metric, call logging)
  prepare_data.py            AirDialogue slices (incl. documented derivation-rule amendment)
  s1_classify.py             run 1, S1: manual-prompt classifier, clean eval
  s2_optimize.py             run 1, S2: DSPy + MIPROv2 on clean data
  s3_perturb.py              run 1, S3: parametric corruption (N1 to N3) + eval
  s4_recover.py              run 1, S4: re-optimization after corruption
  probe_1b.py                run 1 addendum: llama3.2:1b tier probe (clean + N3)
  prepare_banking77.py       run 2: Banking77 slices
  b77_classify.py            run 2: manual-prompt classifier (77 labels)
  b77_optimize.py            run 2: MIPROv2 incl. E1 config (--prompt-model, --zero-shot)
  b77_recover.py             run 2: synthetic-shift eval + re-optimization
  synth_perturb.py           run 2: LLM-synthetic form-shift generation
data/
  dev.jsonl, test.jsonl      AirDialogue slices (150 / 300, seed 42)
  test_n{1,2,3}.jsonl        corrupted AirDialogue test sets
  test_synth.jsonl           synthetically shifted AirDialogue test subsample (150)
  dev_n2.jsonl               corrupted dev set (run 1, S4)
  b77_dev.jsonl, b77_test.jsonl        Banking77 slices (154 / 308, seed 42)
  b77_{dev,test}_synth.jsonl           synthetically shifted Banking77 sets
  b77_labels.txt             the 77 Banking77 intent labels
  raw/banking77/             official PolyAI CSVs (downloaded 2026-08-21)
  raw/google___air_dialogue/ local HF Arrow cache of AirDialogue (387 MB, untracked)
  transfer_projekt/          legacy data of the prior project (not used by the spikes)
artifacts/                   optimized DSPy programs (baseline and re-optimized, both runs)
runs/                        JSONL logs of every LM call, per stage (incl. superseded v0 runs)
results/
  SPIKE_REPORT_CENTRAL.md    consolidated report (start here)
  SPIKE_REPORT.md            run 1 stage-level detail (superseded)
  SPIKE2_REPORT.md           run 2 stage-level detail (superseded)
  results.csv                one row per (stage, condition): f1, validity, calls, wall-clock
  *_stats*.json              per-stage metrics, incl. intermediate/v0 evidence
  *sanity_pairs.json         perturbation sanity-check pairs (label-preservation evidence)
```

## Reproduction

Everything runs locally; no API keys, no paid calls.

```bash
# environment (system Python 3.14 is unsupported by DSPy; use 3.12)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python dspy ollama scikit-learn pandas datasets optuna

# models
ollama serve &
ollama pull qwen2.5:3b      # task model
ollama pull llama3.2:1b     # tier probe
ollama pull llama3.1:8b     # shift generator + MIPRO proposer (run 2)

# run 1 (AirDialogue; requires the raw Arrow cache under data/raw/)
.venv/bin/python src/prepare_data.py
.venv/bin/python src/s1_classify.py
.venv/bin/python src/s2_optimize.py
.venv/bin/python src/s3_perturb.py
.venv/bin/python src/s4_recover.py
.venv/bin/python src/probe_1b.py

# run 2 (Banking77; prepare script expects data/raw/banking77/{train,test}.csv)
.venv/bin/python src/prepare_banking77.py
.venv/bin/python src/b77_classify.py
.venv/bin/python src/b77_optimize.py                                        # naive config (+0.0)
.venv/bin/python src/b77_optimize.py --tag e1 --prompt-model llama3.1:latest --zero-shot
.venv/bin/python src/synth_perturb.py
.venv/bin/python src/b77_recover.py
```

Determinism: seed 42 and temperature 0 everywhere; perturbation is deterministic per (seed, severity, index). LM outputs are deterministic per machine via Ollama seeding; exact numbers may vary marginally across hardware.

## Data provenance

- **AirDialogue** (Wei et al. 2018, Google): loaded from a local HuggingFace Arrow cache of `google/air_dialogue` (train split, 321,459 records). The cache (387 MB) is not tracked; obtain via HuggingFace if reproducing from scratch. Note: `datasets` 5.x no longer loads script-based datasets; load the Arrow file directly (see `src/prepare_data.py`).
- **Banking77** (Casanueva et al. 2020, PolyAI): official CSVs from the PolyAI `task-specific-datasets` GitHub repository (CC-BY-4.0; verify upstream license before thesis submission).
- **Transfer-Projekt data** (`data/transfer_projekt/`): 300-dialogue sample and ground truth from the prior project; kept for provenance, not consumed by any spike code.

## Notes for thesis reuse

- The spikes are throwaway-quality code with keep-quality evidence: every claim in the reports traces to a call log in `runs/` and a row in `results/results.csv`.
- The data derivation amendment (intent-free texts under the naive turn-slicing rule) is documented in `src/prepare_data.py` and evidenced by `runs/s1_test_specrule_v0.jsonl`.
- Superseded artifacts are deliberately retained (v0 stats, naive-optimizer artifacts, per-run reports); they document negative and intermediate results referenced by the central report.
