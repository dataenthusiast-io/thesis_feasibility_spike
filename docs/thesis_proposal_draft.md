# Master Thesis Draft Proposal

**Working title.** Self-Improving Language Model Pipelines for Conversational Intent Classification under Distribution Shift

**Basis.** This thesis builds on the Transfer-Projekt "Large Language Models for the Analysis of Unstructured Conversational Data," which designed and validated a zero-shot LLM pipeline for intent classification of airline customer dialogues (macro F1 = 0.993, schema failure rate 0.0%) and identified architecture economics, prompt robustness, and scalability as open questions. Up to 30% of the Transfer-Projekt is formally reusable and covers the literature foundation, the pipeline architecture, the cost model, and the AirDialogue baseline.

**Synopsis.** Conversational data is growing rapidly, and enterprises must convert it into insights under constrained resources: budget, latency, governance capacity, and engineering attention. Production adds a further constraint: data distributions shift over time, and static systems degrade silently. This thesis investigates, in a single factorial experiment with comparative analysis, (a) whether automatic prompt optimization allows small, locally deployable language models to substitute for frontier models in intent classification, and (b) whether a closed optimization loop preserves classification quality under distribution shift where static configurations degrade. The design proceeds in three stages: baseline measurement on clean data, systematic degradation through controlled perturbation, and closed-loop re-optimization on the shifted data.

## 1 Problem and Motivation

**(1) Conversational data is growing rapidly.** Chat, voice, and messaging interfaces make an increasing share of enterprise interaction data dialogic; industry estimates place unstructured data at 70 to 90% of enterprise data, growing substantially faster than structured data.

**(2) Enterprises must analyze it.** At enterprise volumes, manual analysis is infeasible; unanalyzed conversational data represents unrealized insight into customer intent, friction, and demand.

**(3) LLM pipelines make the analysis feasible.** The Transfer-Projekt demonstrated reliable conversion of conversations into analyzable structure.

**(4) But enterprise resources are constrained in every dimension.** API inference cost scales linearly with volume regardless of falling unit prices; latency budgets constrain synchronous use; EU AI Act obligations and data residency requirements bind governance capacity; and engineering attention is consumed by manual prompt maintenance, which peer-reviewed work identifies as a genuine reliability problem rather than an inconvenience.

**(5) Production data does not stand still.** Noise (ASR errors, typos, colloquialisms) and distribution shift silently degrade static systems, while conventional remediation through relabeling, retraining, or manual prompt repair is slow and expensive.

**(6) A promising resolution: self-improving pipelines on right-sized models.** Small, locally deployable models address cost, latency, and residency constraints; automatic prompt optimization (APO) systematizes prompt engineering and, operated as a closed loop, may adapt the system to shifted data without new labels or human intervention. Whether this holds, and at what cost, is an open empirical question and the subject of this thesis.

## 2 Research Aim

Quantify, in a factorial comparative experiment, (a) whether automatic prompt optimization substitutes for model scale in conversational intent classification and (b) whether a closed optimization loop preserves quality under distribution shift where static configurations degrade.

**Guiding question:** How do model scale, prompt regime, and data condition interact in determining the quality-cost frontier of conversational intent classification pipelines?

## 3 Experimental Design

### 3.1 Factors and conditions

**Factor 1 — Model:** TF-IDF + SVM (classical static baseline, trained rather than prompted) | local SLM in three size tiers (~1B, ~3B, ~7-8B) | frontier API model (upper anchor).

**Factor 2 — Prompt regime** (language model conditions only): manual-static (expert-written prompt, frozen) | optimized-static (optimized via APO on clean data, then frozen) | optimized-adaptive (re-optimized by the closed loop after the shift).

**Data condition:** clean (original corpus) | shifted (systematic perturbation at three severity levels N1 to N3).

| Model | manual-static | optimized-static | optimized-adaptive |
|---|---|---|---|
| TF-IDF + SVM | n/a (trained baseline; cannot adapt without new labels) | n/a | n/a |
| SLM ~1B | clean + shifted | clean + shifted | shifted |
| SLM ~3B | clean + shifted | clean + shifted | shifted |
| SLM ~7-8B | clean + shifted | clean + shifted | shifted |
| Frontier API | clean + shifted | clean + shifted | shifted |

The adaptive regime exists only under shift by construction; on clean data it coincides with optimized-static.

### 3.2 Procedure

**Step 1 — Harness.** The Transfer-Projekt pipeline (segmentation, classification, structured output, evaluation) is extended into the experimental harness with a unified model interface, so API models and local models (served via Ollama or vLLM) run through identical code paths, and all runs are logged with token counts, wall-clock latency, and configuration hashes for reproducibility.

**Step 2 — Data preparation.** The corpus is split into an optimization/development set (used by the APO optimizer and for prompt iteration) and a held-out test set (never seen during any optimization), stratified by intent class.

**Step 3 — Manual prompts.** Expert prompts are written per model family under a documented protocol (fixed iteration budget, fixed development data), then frozen. This operationalizes "manual-static" as realistic practitioner effort rather than a strawman.

**Step 4 — Prompt optimization.** Each language model condition is compiled as a program in an APO framework (DSPy with MIPROv2 as primary optimizer) against macro F1 on the development set, under a fixed and logged optimization budget (number of candidate prompts, number of evaluation calls). The resulting prompts are frozen as "optimized-static." Optimization cost is recorded as a first-class measurement.

**Step 5 — Clean evaluation.** All grid cells are evaluated once on the held-out clean test set, establishing the baseline frontier and, simultaneously, the definitive task-specific model comparison for the SLM tiers.

**Step 6 — Shift generation.** Distribution shift is simulated by a parameterized perturbation pipeline applied to the evaluation data: character- and word-level noise (typos, ASR-style phonetic confusions, truncation) and meaning-preserving colloquial rephrasing, at severity levels N1 to N3, with manual spot-checks for realism and label validity. Because perturbations preserve the intent label, ground truth remains valid without any relabeling; this is what makes label-free adaptation measurable in Step 8. An identical protocol applied to a second public corpus is an option for external validity.

**Step 7 — Static evaluation under shift.** All manual-static and optimized-static cells are re-evaluated on the shifted test sets without any modification, yielding degradation curves per model size and prompt regime.

**Step 8 — Adaptive re-optimization.** For the optimized-adaptive regime, the optimizer is re-run against a shifted slice of the development set (original labels, perturbed inputs), then evaluated on the shifted test set. Measured outcomes: recovered quality relative to the static regimes and the cost of re-optimization. The re-optimization trigger and data-access protocol are fixed in advance.

**Step 9 — Analysis.** Descriptive quality-cost frontier plots per data condition; per-cell-pair significance testing on the held-out set (e.g., bootstrap or McNemar); and the two central interaction analyses: optimization gain by model scale (clean) and prompt regime by data condition (under shift). The total cost of ownership model from the Transfer-Projekt is extended with self-hosting and optimization cost components.

### 3.3 Hypotheses (draft)

- **H1:** Quality gained per additional cost unit decreases markedly above the small-model tier.
- **H2:** The quality lift from automatic prompt optimization is inversely related to model scale.
- **H3:** Under distribution shift, the optimized-adaptive regime retains significantly more macro F1 than both static regimes, at a measurable re-optimization cost; the classical baseline quantifies the degradation of non-adaptable systems.

All outcomes are informative: recent work indicates that LLM-driven optimization loops can be unstable, so quantified limits of the adaptive regime constitute a finding in their own right.

### 3.4 Model selection

Selection criteria for the SLM tiers: open weights; quantized operation on commodity GPU (≤16 GB VRAM); three distinct size tiers; documented instruction-following and structured-output capability; workable license. Candidates: Llama 3.2 1B, Gemma 2 2B (tiny); Qwen 2.5 3B, Phi-4-mini (small); Qwen 2.5 7B, Llama 3.1 8B (mid). Pre-selection uses published instruction-following scores plus a pilot run on a validation slice; the clean-data evaluation itself constitutes the definitive task-specific comparison, mitigating benchmark contamination concerns.

### 3.5 Implementation scope

**Reused from the Transfer-Projekt:** pipeline core (segmentation, classification, structured output), evaluation metrics and reporting, linear API cost model, prepared AirDialogue corpus.

**To be built:** unified model interface covering API and local inference (Ollama or vLLM); training of the classical baseline (TF-IDF + SVM on the labeled development split; low effort, stated for completeness); the perturbation pipeline including a validation protocol for label preservation and realism; optimization budget logging; extension of the cost model with self-hosting and optimization components.

**Deliberately out of scope:** fine-tuning of the SLMs. This is a design decision rather than an omission: the research question concerns prompt-level adaptation as a substitute for scale, and weight-level tuning would confound the comparison, expand scope, and reintroduce exactly the labeling and retraining cost that the closed-loop regime is designed to avoid. Also out of scope: proprietary production data and multi-turn dialogue state tracking.

## 4 Literature Requirements

Rather than a fixed reference list, the structured literature review will be driven by the claims that require academic foundation. Each anchor below states what the literature must establish and for which part of the thesis.

| # | Anchor claim / topic | What the literature must provide | Serves |
|---|---|---|---|
| L1 | Growth and dominance of unstructured/conversational data | Primary-source quantification (or correction) of the 70-90% and growth-rate estimates; peer-reviewed or authoritative institutional sources replacing vendor figures | Motivation (1) |
| L2 | Paradigm evolution of NLP-based text classification | Canonical architecture and paradigm literature (transformer, pre-training, in-context learning) | Foundations; reused from Transfer-Projekt |
| L3 | Prompt sensitivity as a reliability problem | Peer-reviewed evidence quantifying performance variance under prompt/formatting changes | Motivation (4), justification of APO |
| L4 | Automatic prompt optimization: methods and state of the art | The APO method family (search-based, textual-gradient, Bayesian/instruction-plus-demo optimization), incl. the frameworks used (DSPy, MIPROv2) and their evaluation practice | Method Step 4, related work |
| L5 | Distribution shift and robustness of NLP systems | Definitions and taxonomy of shift; established noise/perturbation protocols and ASR-robustness work to ground the shift simulation | Method Step 6, related work |
| L6 | Closed-loop / self-improving LLM systems | Current state of continuous or feedback-driven optimization, incl. documented instability of optimization loops | Motivation (6), H3 framing, discussion |
| L7 | Intent classification benchmarks and their difficulty | Benchmark provenance, class granularity, known ceilings and baselines for the chosen corpora | Corpus choice, Step 5 |
| L8 | Small language models and local deployment | Technical reports and comparative evidence for the candidate SLMs; quantization and serving considerations | Model selection 3.4 |
| L9 | Inference economics and cost-aware model selection | Peer-reviewed or authoritative analyses of inference cost structures, cost-performance trade-offs, and cascading/selection approaches | TCO model, discussion |
| L10 | Governance constraints on architecture choice | EU AI Act quality-management/monitoring obligations and data residency implications for API vs. on-prem deployment | Motivation (4), discussion |
| L11 | Comparative baselines closest to this design | Prior work combining cost, model choice, and intent classification, to state the delta of this thesis precisely | Related work |

## Appendix: Preliminary References Identified So Far

**Canonical foundations.** Vaswani et al. 2017 (NeurIPS); Devlin et al. 2019 (NAACL); Brown et al. 2020 (NeurIPS); Hinton et al. 2015.

**Automatic prompt optimization.** Khattab et al. 2023, DSPy (ICLR 2024); Opsahl-Ong et al. 2024, MIPROv2 (EMNLP 2024); Pryzant et al. 2023 (EMNLP 2023); Zhou et al. 2022, APE (ICLR 2023); Yang et al. 2023, OPRO (ICLR 2024); Yuksekgonul et al. 2024, TextGrad (journal venue to verify); Agrawal et al. 2025, GEPA (preprint).

**Prompt sensitivity and robustness.** Sclar et al. 2024 (ICLR); Li et al. 2023, robust prompt optimization under distribution shifts (EMNLP).

**Benchmarks.** Wei et al. 2018, AirDialogue (EMNLP); Larson et al. 2019, CLINC150 (EMNLP); Casanueva et al. 2020, Banking77 (ACL workshop).

**Cost and annotation economics.** Chen et al. 2023, FrugalGPT (venue to verify); Loukas et al. 2023 (venue to verify); Hsieh et al. 2023 (ACL Findings); Gilardi et al. 2023 (PNAS).

**Systems tradition.** Kephart & Chess 2003, autonomic computing (IEEE Computer).

**Industry estimates (grey literature, motivation only).** Gartner/IDC data-growth estimates; Epoch AI inference price analyses.
