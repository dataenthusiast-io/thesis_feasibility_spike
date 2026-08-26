# Central Spike Report: Thesis Feasibility Validation

**Thesis (working title):** Self-Improving Language Model Pipelines for Conversational Intent Classification under Distribution Shift

**Date:** 2026-08-21 · **Scope:** two spike runs, executed same day, fully local
**Run 1:** AirDialogue (spec assumptions A1 to A4) · **Run 2:** corpus and shift-instrument decision (Banking77, synthetic form shift)

---

## In brief (plain-language interpretation)

Three questions drove the spike; each has a short answer before the numbers.

**Does the machinery work?** Yes. A 3B local model classifies reliably with valid structured output, and DSPy/MIPROv2 optimizes end to end on consumer hardware in minutes. No technical blockers anywhere.

**Is there anything left to optimize?** Not on AirDialogue: every model tier scores near perfect on clean data, so neither optimization gains nor model-size differences are measurable there. On Banking77 (77 fine-grained intents) the same models score 0.52 (3B) and 0.27 (1B): real headroom, real scale differences. This settled the corpus decision.

**Does self-optimization work?** In two parts. *On clean data:* yes, demonstrated. The optimizer found a prompt that beats the hand-written one by 2.7 F1 points on held-out data (though only after configuration fixes; the out-of-the-box run gained exactly zero). *Under drift:* the central story of the spike, told by one table. Realistic drift was simulated by rewriting every test message into casual, indirect chat language; then all three prompt regimes were scored on it:

| Prompt | clean data | drifted data |
|---|---|---|
| hand-written, frozen | 0.518 | 0.370 |
| optimized on clean data, frozen | **0.531** (best) | **0.329** (worst) |
| re-optimized on drifted data (the loop) | n/a | 0.339 |

Read across the rows: the optimized prompt wins on clean data but falls hardest under drift, ending *below* the hand-written prompt it had beaten. Optimizing on one distribution creates brittleness against the next one, which is precisely the production problem the thesis loop targets. The loop was then retested on the drifted data: re-optimization recovered ground for the first time (+1.0 pt vs the frozen prompt), but the gain is within optimizer noise and still trails the hand-written prompt. **Conclusion: the spike proves all preconditions of the thesis (headroom, realistic drift, demonstrated brittleness, a recovery mechanism pointing the right way); whether the loop delivers significant recovery under proper budgets and repeated seeds is exactly the open question the thesis answers.**

---

## 1 Executive summary

The spike validates the thesis machinery end to end and settles two design questions with metrics.

| # | Question | Verdict | Key evidence |
|---|---|---|---|
| A1 | Can a small local LM classify intents with reliable structured JSON output? | **GO** | 100% validity, macro F1 0.997 (qwen2.5:3b, AirDialogue clean) |
| A2 | Does DSPy + MIPROv2 run end to end against macro F1 within budget? | **GO** | Completes in 148 to 372 s, 665 to 708 LM calls per run, no crashes |
| A3 | Does parameterized noise measurably degrade quality? | **GO** | Monotonic 0.997 to 0.899 (N1 to N3), plus synthetic shift -11 pts |
| A4 | Does re-optimization recover quality after shift? | **Directional** | -1.0 pts under corruption (no headroom), +1.0 pts under form shift |
| D1 | Does AirDialogue have headroom for the thesis grid? | **NO** | 0.997 (3B) and 0.975 (1B) clean: saturated across the SLM range |
| D2 | Does Banking77 restore headroom? | **YES** | 0.52 (3B) / 0.27 (1B) clean; 25 pts scale separation |
| D3 | Is DSPy lift measurable? (title-critical) | **YES, on Banking77** | +2.7 F1 pts held-out at minimal budget; +0.0 with naive configuration |
| D4 | Is LLM-synthetic shift a viable drift instrument? | **YES, with protocol** | -11 pts (AirDialogue), -15 to -20 pts (Banking77); label-drift caveat |

**Decisions taken on this basis:** the thesis corpus is switched to Banking77; LLM-synthetic form shift is adopted as the primary drift instrument, subject to a label-validation protocol; optimizer configuration (proposer model, demo policy) is treated as an explicit experimental variable. Section 2 documents how these decisions emerged; sections 4 and 5 hold the per-run evidence; section 6 states the resulting design decisions in full; section 7 logs the assumptions that still require verification.

### 1.1 The decision grid

The spike process converged on the insight that corpus choice and shift instrument are two orthogonal design axes: the corpus determines *headroom* (room for optimization and scale effects), the shift instrument determines *realism and recoverability* (whether lost quality is prompt-shaped and can be re-optimized back). Measured cells, macro F1 of the 3B manual-prompt classifier:

| Shift instrument \\ Corpus | AirDialogue (3 classes) | Banking77 (77 classes) |
|---|---|---|
| clean (no shift) | 0.997 (saturated) | 0.518 (headroom) |
| parametric corruption (N3) | 0.899 (unrealistic at effective severity) | not run (instrument deprioritized, see Iteration 3) |
| synthetic form shift | 0.884 (realistic, works) | **0.370 (thesis arena)** |

Overlaid on this grid, the two title-critical mechanisms:

| Mechanism | AirDialogue | Banking77 |
|---|---|---|
| APO lift on clean data (MIPROv2, held-out test) | +0.000 (nothing to gain) | +0.027 (E1 config; +0.000 naive) |
| Optimized-static under synthetic shift | not run | 0.329 (degrades more than manual 0.370: brittleness) |
| Recovery by re-optimization after shift | -0.010 (corruption, N2) | **+0.010 (form shift)** |

The bottom-right region (Banking77 under synthetic form shift, with optimizer configuration controlled) is where every effect the thesis needs is simultaneously measurable: headroom, APO lift, scale separation, brittleness of static optimization, and directional recovery.

---

## 2 Decision history: how one spike became two

The spike was planned as a single run against four assumptions. Seven decision points, each triggered by an unexpected result or a challenge to the design, turned it into two runs and a corpus decision. This section preserves that reasoning; the numbers are detailed in sections 4 and 5.

**Iteration 1: data amendment (Run 1, S1 dry run).** The first S1 execution under the spec-verbatim derivation rule ("customer's first two turns") returned validity 88% and F1 0.933. Inspection of the call logs showed every single failure was a greeting-plus-self-introduction text with no intent content ("Hi. I am Margaret Miller."); the model answered `{"intent": "none"}`, correctly. Constrained decoding changed nothing, isolating a data problem rather than a formatting problem. Decision: mechanical, label-agnostic amendment (accumulate customer turns, max 3, until 12 words); archive the v0 evidence. Effect: validity 88% to 100%, F1 0.933 to 0.997. Lasting insight: derivation errors masquerade as model unreliability and must be controlled in the thesis pipeline.

**Iteration 2: ceiling diagnosis and tier probe.** S2 produced a flat 0.0 optimization delta at F1 0.997, and S4 produced negative recovery (-0.98 pts). The question arose whether the ceiling was specific to the 3B model or a property of the corpus. A probe with llama3.2:1b answered it: 0.975 clean. Conclusion: clean AirDialogue is saturated across the entire planned SLM range; H1 and H2 have no measurable variance there. This confirmed, with numbers, the ceiling concern already named in the Transfer-Projekt limitation section.

**Iteration 3: realism challenge to the corruption instrument.** A review of the perturbed examples challenged the corruption pipeline: only near-destructive severity (N3: dense typos, truncation into the intent verb) moved the metric, while realistic severities (N1/N2, plausible typo and ASR profiles) cost at most 3.2 pts. Diagnosis: on a keyword-redundant 3-class task, only information destruction hurts, and destroyed information is unrecoverable by prompt-level adaptation, which is why S4 had nothing to fix. Conclusion: on this corpus the corruption instrument measures noise robustness, not adaptable drift, and cannot power H3. Candidate remedy: a harder corpus (Banking77, shortlisted earlier in the thesis process for exactly this property).

**Iteration 4: the supervisor's synthetic-data angle reframes the choice.** Before committing to a corpus switch, the supervisor's proposal (LLM-rewritten realistic variants instead of corrupted text) was assessed. Analysis: synthetic form shift addresses realism and recoverability (style and register are learnable from demonstrations), but not headroom; a saturated task stays saturated under meaning-preserving rewrites. This reframed corpus and shift instrument as orthogonal axes (section 1.1) and defined Run 2 as a 2x2 across both, plus the title-critical DSPy test and a recovery cell, so the instrument-versus-corpus question could rest on measured cells rather than positions.

**Iteration 5: prediction correction on synthetic shift.** The working prediction was that meaning-preserving rewrites cannot hurt a 3-class keyword task. Measured: -11.2 pts on AirDialogue, more than maximal parametric corruption (-9.8), at near-perfect format validity. Realistic indirection ("i was supposed to go on a hike this weekend but it's been cancelled and now i'm free - can i modify the dates") removes the keyword anchors the classifier leans on. Effect: the synthetic-shift instrument is validated even on the easy corpus; the corpus switch remains necessary for clean-data headroom, but the instrument decision was settled here.

**Iteration 6: zero-lift diagnosis and optimizer reconfiguration (E1).** The first MIPROv2 run on Banking77 returned a lift of exactly +0.000: the optimizer explored instruction and demo candidates, all scored below the unmodified baseline on dev (44 to 48 vs 54), and correctly returned the baseline. Diagnosis: (a) few-shot demos structurally cannot cover 77 classes and bias the model toward the demoed classes; (b) by default the 3B task model proposes its own instruction candidates, and proposes poorly. Reconfiguration (E1): llama3.1:8b as dedicated `prompt_model`, instruction-only search (no demos). Result: +2.7 pts on held-out test at the smallest budget. Effect: rescued the title-critical claim and promoted optimizer configuration from plumbing to an explicit experimental variable.

**Iteration 7: generation guard bug and label-drift discovery.** The first synthetic generation silently kept 30% of originals because an over-strict output-length guard rejected legitimate rewrites (a colloquial, indirect rewrite of a five-word banking query is naturally several times longer). Fixed and fully regenerated (0 fallbacks). The sanity pairs of the regenerated sets then surfaced genuine label drift in a minority of rewrites (a cancel rewritten as "change or even cancel"; a top-up question turned into a missing-transfer complaint). Effect: clean shifted sets, plus the requirement, now part of the design, that synthetic shift carries a label-validation protocol and generator/classifier family separation.

**Iteration 8: inversion proposal and goal reframing.** After the corpus decision, an inversion of the ceiling problem was considered: keep AirDialogue and redefine the goal as *retaining* near-ceiling quality under degrading data rather than improving further. Assessment against the measured cells: the retention idea strengthens the thesis narrative (production systems are judged on not degrading, matching H3 and the supervisor's original two-step design), but it does not rescue AirDialogue, because the prompt-regime factor collapses there (MIPRO found nothing on clean data, so manual-static and optimized-static are the same artifact), H2 loses all variance, and the design would rest solely on recovery deltas that have so far been within optimizer noise. Decision: the retention framing is adopted as the primary formulation of the experimental goal (quality retention relative to a clean anchor), applied on Banking77; the corpus decision stands. One decisive cell for the inversion (re-optimization on synthetically shifted AirDialogue) remains unmeasured and is logged in section 7.

**Iteration 9: optimizer instrument probe (GEPA, 2026-08-26).** After the proposal fixed MIPROv2 as optimizer of record while citing GEPA (ICLR 2026) as outperforming it, the instrument choice was grounded empirically: GEPA was run on the two title-critical Banking77 cells under maximum comparability (same task model, same data and seed, the same llama3.1:8b in the reflection role that MIPROv2 E1 used as proposer, and the budget capped at 700 metric calls to approximate E1's 685 task-model calls; realized task-model calls were 861 and 850 per cell, roughly a quarter more than E1, inflated by parse-retry calls, which is recorded as a residual comparability caveat in GEPA's favor; the thesis's Phi-4 proposer allocation applies to future runs only and is deliberately not used here). Results: clean lift 0.5514 (GEPA) vs 0.5307 (MIPROv2) vs 0.5035 (baseline); recovery on shifted test 0.3504 (GEPA) vs 0.3390 (MIPROv2) vs 0.3292 (frozen), manual prompt 0.3696. Reading: GEPA outperforms MIPROv2 on both cells at matched budget (single seed each), yet the qualitative ordering is unchanged: recovery under both optimizers remains below the manual prompt at small budgets. The brittleness and partial-recovery pattern is therefore not optimizer-specific, which strengthens the H2/H3 evidence base and pre-answers the instrument objection. Decision: MIPROv2 remains optimizer of record (reference-method rationale unchanged); the Step 8 GEPA robustness check is upgraded from optional to planned for the H3-critical cells; the record-vs-robustness allocation is revisited only if the thesis pilot reproduces a GEPA advantage beyond seed variance. *Addendum (same day):* review flagged that the brittleness claim for GEPA was initially asserted without measuring the frozen-clean-program-under-shift cell; the cell was then measured (0.3480). It confirms the pattern with data (drop of 20.3 pts, near-identical to MIPROv2's 20.2, ending below the manual prompt) and corrects the recovery reading: GEPA's within-optimizer recovery is +0.2 pts against its own frozen program, smaller than MIPROv2's +1.0; the earlier +2.1 figure compared GEPA's adapted program against MIPROv2's frozen one, a cross-optimizer number. Both within-optimizer recovery deltas lie inside seed noise.

**Net effect of the iterations:** Run 1 alone would have concluded "machinery works, but no headroom, unrealistic shift, no recovery." The iterations converted that into: a validated instrument (synthetic form shift), a validated arena (Banking77), a measurable title claim (+2.7 pts APO lift), an observed brittleness effect motivating H3, and a positive recovery signal, with every negative intermediate result retained as evidence.

---

## 3 Setup

- **Environment:** Python 3.12 (uv venv; system Python 3.14 is unsupported by DSPy), dspy 3.3.0 + optuna, scikit-learn, pandas, Ollama. Temperature 0 and seed 42 everywhere. All runs local, zero API cost.
- **Models:** qwen2.5:3b (primary task model), llama3.2:1b (tier probe), llama3.1:8b (synthetic-shift generator and MIPRO instruction proposer; deliberately a different family than the classifiers to limit generator-classifier coupling).
- **Data, Run 1:** AirDialogue from the locally cached train split (321,459 rows). Slices seed 42, stratified: dev 150 (50/class), test 300 (100/class), labels {book, cancel, change}.
- **Data, Run 2:** Banking77 (official PolyAI CSVs, 77 intents). Slices seed 42, stratified: dev 154 (2/class, from official train), test 308 (4/class, from official test).
- **Reproducibility:** every stage is an independent idempotent script under `src/`; all LM calls logged to `runs/` as JSONL; per-condition metrics in `results/results.csv`; optimized programs in `artifacts/`.

---

## 4 Run 1: AirDialogue (assumptions A1 to A4)

### 4.1 Data derivation finding (before any model ran)

The planned derivation rule ("customer's first two turns") produced ~12% intent-free texts (greeting plus self-introduction, e.g. "Hi. I am Margaret Miller."; the intent appears in turn 3). A first S1 run under that rule scored 88% validity / F1 0.933, and every failure was the model correctly answering `{"intent": "none"}` to an unclassifiable text; constrained decoding changed nothing, confirming a data problem, not a formatting problem. Amended rule (mechanical, label-agnostic): accumulate customer turns (max 3) until the text reaches 12 words. Lesson for the thesis pipeline: naive dialogue slicing injects label noise that masquerades as model unreliability. Evidence: `runs/s1_test_specrule_v0.jsonl`.

### 4.2 A1: structured classification (GO)

qwen2.5:3b, manual prompt, JSON output, one retry allowed, 300 clean test examples: **macro F1 0.9967, validity 100% first try**, 17.5 s per 100 examples. Matches the Transfer-Projekt frontier-model result (0.993) with a 3B local model, confirming the anticipated ceiling.

### 4.3 A2: DSPy + MIPROv2 end to end (GO)

MIPROv2 `auto="light"` on dev completed in 148 s / 673 calls; optimized program inspectable (`artifacts/optimized_program.json`). Test F1 unchanged at 0.9967 (expected at ceiling: dev score was already 100%). Setup findings: MIPROv2 requires the optional `optuna` package; `requires_permission_to_run` is deprecated in dspy 3.3.

### 4.4 A3: parametric corruption degrades quality (GO)

Three composed perturbation types (keyboard typos, ASR-style errors, truncation), deterministic under seed:

| Condition | macro F1 (3B) | validity |
|---|---|---|
| clean | 0.9967 | 100% |
| N1 | 0.9900 | 100% |
| N2 | 0.9650 | 99.7% |
| N3 | 0.8991 | 96.7% |

Monotonic, total drop 9.8 pts. Realism caveat: only near-destructive severity (N3) moved the metric; N1/N2, which resemble realistic typo/ASR profiles, cost at most 3 pts. On a saturated 3-class task, only information destruction hurts, which motivated Run 2.

### 4.5 A4: re-optimization after corruption (no recovery, diagnosed)

Frozen clean-optimized program on corrupted test (N2): 0.9701. Re-optimized on corrupted dev: 0.9603 (**delta -0.98 pts**, 683 calls, 136 s). Diagnosis: at N2 the frozen program had almost nothing left to recover (max 2.7 pts, below optimizer run-to-run variance of about ±1 pt), and character corruption destroys information no prompt can restore. The recovery mechanism was tested in a cell with nothing prompt-shaped to fix.

### 4.6 Tier probe: the ceiling holds across the SLM range

| Model | clean F1 | N3 F1 | N3 validity |
|---|---|---|---|
| qwen2.5:3b | 0.9967 | 0.8991 | 96.7% |
| llama3.2:1b | 0.9746 | 0.6722 | 88.7% |

Even the 1B tier is within 2.2 pts of perfect on clean data. Conclusion: clean AirDialogue offers no variance for H1/H2 anywhere in the planned model range; under shift, scale separation is large (30 pts at N3 for 1B).

---

## 5 Run 2: Banking77 and synthetic shift (decisions D1 to D4)

### 5.1 Headroom (D2)

Manual prompt (77 labels listed, JSON output, one retry), clean test:

| Model | macro F1 | accuracy | validity |
|---|---|---|---|
| qwen2.5:3b | 0.5175 | 0.536 | 93.8% |
| llama3.2:1b | 0.2670 | 0.273 | 66.2% |

~48 pts of headroom at the 3B tier, ~25 pts of scale separation, and structured-output reliability becomes a live variable at 1B. Consistent with published difficulty (zero-shot LLMs around low-70s accuracy, fine-tuned ~94).

### 5.2 Measurable DSPy lift (D3, title-critical)

MIPROv2 `auto="light"` on clean dev, `dspy.Predict`, held-out clean test:

| Configuration | test macro F1 | lift |
|---|---|---|
| DSPy baseline (unoptimized) | 0.5035 | - |
| MIPRO default (self-proposed instructions, few-shot demos) | 0.5035 | +0.000 |
| MIPRO E1 (llama3.1:8b proposer, instruction-only, no demos) | **0.5307** | **+0.027** |

The zero under the default configuration is a finding: few-shot demos structurally cannot cover 77 classes (demo candidates scored worse on dev, biasing toward demoed classes), and a 3B model proposing its own instructions proposes poorly. With a stronger proposer and zero-shot instruction search, the lift is +2.7 pts at the smallest budget, single seed. Larger budgets and repeated seeds are expected to widen this; quantifying that is thesis work. Optimizer configuration therefore enters the design as an explicit variable.

### 5.3 Synthetic form shift as drift instrument (D4)

Every utterance rewritten by llama3.1:8b into casual, indirect chat register (deterministic, temperature 0). Degradation, 3B:

| Corpus / prompt | clean | synthetic shift | drop |
|---|---|---|---|
| AirDialogue, manual | 0.9967 | 0.8844 | -11.2 pts |
| Banking77, manual | 0.5175 | 0.3696 | -14.8 pts |
| Banking77, optimized-static (E1 frozen) | 0.5307 | 0.3292 | -20.2 pts |

Notable: synthetic shift degrades AirDialogue *more* than maximal character corruption (0.884 vs 0.899) while remaining realistic and format-clean (98.7% validity). Realistic indirection removes the keywords the classifier leans on.

**Brittleness finding (H3-relevant):** the clean-optimized prompt loses more under shift than the manual prompt (-20.2 vs -14.8 pts, ending below it: 0.329 vs 0.370). Optimized-static overfits the clean distribution. This is the failure mode the adaptive regime targets, observed in own data.

**Label-preservation caveat:** synthetic rewriting puts an LLM inside the measurement instrument. Observed failure modes: a cancel rewritten as "change or even cancel" (label blur), a top-up question rewritten as a missing-transfer complaint (label flip). Measured degradation therefore conflates model failure with label drift. Required protocol for the thesis: generator/classifier family separation (practiced here) plus label validation on a sample (human spot-check or independent judge). Parametric corruption preserves labels by construction and can remain a controlled secondary severity axis.

### 5.4 Optimizer instrument probe: GEPA vs MIPROv2 at matched budget (added 2026-08-26)

The title-critical cells rerun with GEPA under maximum comparability (details and decision in Iteration 9), including the frozen-program-under-shift cell added after review:

| Program (all single seed) | clean test | shifted test |
|---|---|---|
| manual prompt, frozen | 0.5175 | 0.3696 |
| MIPROv2 E1 optimized on clean, frozen | 0.5307 | 0.3292 |
| GEPA optimized on clean, frozen | **0.5514** | 0.3480 |
| MIPROv2 re-optimized on shifted dev | n/a | 0.3390 |
| GEPA re-optimized on shifted dev | n/a | **0.3504** |

Reading: (a) GEPA wins the clean lift at matched budget (+4.8 vs +2.7 over baseline). (b) The brittleness pattern replicates across optimizers as a measured result: GEPA's frozen program drops 20.3 pts under shift (MIPROv2: 20.2; manual: 14.8) and ends below the manual prompt. (c) Within-optimizer recovery deltas are +1.0 (MIPROv2) and +0.2 (GEPA), both inside seed noise; GEPA's adapted program is the best shifted-data program measured (0.3504) but still trails the manual prompt. Artifacts: `artifacts/b77_gepa_program_clean.json`, `artifacts/b77_gepa_program_synth.json`; logs: `runs/b77_gepa_*_history.jsonl`; stats: `results/b77_gepa_stats.json`.

### 5.5 Recovery under form shift (A4 revisited)

Re-running MIPRO (E1 configuration) on the synthetically shifted dev set, evaluated on the shifted test set: frozen 0.3292 to re-optimized 0.3390, **recovery +0.98 pts** (665 calls, 368 s). First positive delta (contrast -0.98 under corruption), directionally supporting the adaptive regime, but single-seed and within optimizer noise; the re-optimized program still trails the manual prompt under shift (0.339 vs 0.370). Claiming H3 requires larger optimization budgets, repeated seeds, and larger test sets.

---

## 6 Design decisions for the thesis

1. **Corpus: Banking77 replaces AirDialogue.** AirDialogue is saturated across the SLM range on clean data (D1); Banking77 restores headroom, scale separation, measurable APO lift, and realistic shift sensitivity (D2 to D4). The AirDialogue saturation result itself remains reusable: it validates the Transfer-Projekt limitation section and motivates the switch in the thesis narrative, preserving continuity.
2. **Drift instrument: synthetic form shift is primary**, with a pre-registered label-validation protocol and generator/classifier family separation; parametric corruption remains available as an optional, fully controlled secondary axis. Severity is parameterized via rewrite instructions (degree of indirection, register).
3. **Optimizer configuration is an experimental dimension:** proposer model strength and demo policy interact with class count; the naive default produced zero lift where a corrected configuration produced +2.7 pts.
4. **Recovery experiments target cells with prompt-shaped headroom:** form shift (recoverable in principle) rather than information-destroying corruption; small models and high severities, where validity failures (prompt-addressable) are part of the loss.
5. **Statistical power:** test sets of 1000+ examples per cell and 3 to 5 optimizer seeds per condition (local calls are free). Deltas of ±1 pt at n≈300 are inside noise; spike-observed optimizer variance was about ±1 pt.
6. **Data pipeline:** utterance derivation includes an intent-bearing-content criterion (Run 1 lost 12 validity points to a naive slicing rule); the classification unit (single utterance vs full dialogue) trades redundancy against shift sensitivity and is fixed deliberately per corpus (short units maximize experimental signal and match early-intent-detection practice).
7. **Tooling constraints (resolved, to be documented in the Exposé):** DSPy requires Python 3.13 or lower; MIPROv2 needs the optional optuna dependency; Literal-typed outputs surface off-label answers as parse errors and the metric handles them.
8. **Goal framing: quality retention under drift is the primary objective formulation.** The experiment is framed as maintaining a clean-data quality anchor while input quality degrades, rather than maximizing absolute scores; hypotheses on optimization lift and scale substitution operate within that frame (Iteration 8).

## 7 Logged assumptions requiring verification

The spike results rest on the following assumptions, ordered by potential impact. Items 1, 2 and 4 could qualitatively change a conclusion and must be verified before the design is finalized; the remaining items bound the precision of the reported numbers.

1. **Label preservation of synthetic rewrites is assumed, not established.** Only 10 pairs per set were spot-checked (by the executing model, not a human); known drift cases are documented (a cancel rewritten as "change or even cancel", a top-up question turned into a missing-transfer complaint). Measured degradation therefore conflates model failure with label drift in unknown proportions. *Verification: human review of 30 to 50 random pairs per corpus and an estimated label-drift rate; this doubles as the pilot of the thesis validation protocol. Evidence: `results/synth_sanity_pairs.json`.*

2. **Shift severity is authored, not calibrated.** The rewrite instruction explicitly demands informality and indirection ("describe what happened instead of naming the action"), so the measured drops (-11 to -20 pts) are downstream of that specific wording and were not calibrated against any real drifted corpus. *Verification: review the prompt in `src/synth_perturb.py` and endorse or revise the realism claim; plan a severity ladder (degree of indirection, register) for the thesis.*

3. **The generator violated its own "no new factual details" rule in some rewrites** (invented scenarios such as "i ordered something last week"). The shift is therefore partly topical rather than purely stylistic; topical drift is the primary label-flip mechanism on fine-grained intents and feeds assumption 1.

4. **The recovery test used matched shift distributions.** Dev and test sets were shifted by the identical generator and prompt, so re-optimization adapted to exactly the distribution it was tested on. The +1.0 pt recovery is accordingly a best-case signal. *Verification: the thesis design tests recovery against a held-out shift variant or a second generator.*

5. **Every optimizer delta is a single run at the smallest budget** (MIPROv2: +2.7 clean, +1.0 within-optimizer recovery, -1.0 under corruption; GEPA at matched 700-call budget: +4.8 clean, +0.2 within-optimizer recovery; one seed each); observed run-to-run variance is about ±1 pt. The clean-data lifts are measured on held-out data and are therefore not dev overfitting, but each magnitude is one draw, including the GEPA-over-MIPROv2 clean-lift gap of about 2 pts. All deltas carry this qualifier until rerun with 3 to 5 seeds and larger budgets.

6. **Generator/proposer coupling in the Run 2 recovery cell:** llama3.1:8b both generated the shifted data and proposed instruction candidates during re-optimization. *Verification: the thesis separates generator, proposer, and task model into three distinct families, or explicitly discusses the coupling.*

7. **Method scope is narrow:** only `dspy.Predict` was tested (no ChainOfThought), and the optimizer maximizes exact-match accuracy on a class-balanced dev set, which approximates but is not identical to the reported macro F1.

8. **One design-relevant cell is unmeasured:** re-optimization on synthetically shifted AirDialogue (the decisive test for the inversion idea of Iteration 8) was not run; the corpus decision rests on the structural arguments (regime collapse, dead H2) plus the measured recovery pattern, not on that cell. *Verification if the inversion is revisited: one MIPRO run on a shifted AirDialogue dev set, roughly 20 minutes with the existing harness.*

9. **Practical bounds:** the AirDialogue synthetic evaluation used a stratified 150-example subsample (not the full 300); the Banking77 dev set holds only 2 examples per class, so MIPRO's internal 100-example validation set covers only part of the label space per trial; determinism is per machine (Ollama seeding), so exact numbers may vary marginally on other hardware.

## 8 Cost and reproducibility

Total across both runs: roughly 10,000 LM calls, about 2 hours wall-clock, all local, zero API cost. Hard caps (2 h or 5,000 calls per stage) were never approached by any single stage. Every stage rerunnable via `python src/<stage>.py`; complete call logs in `runs/`, per-condition metrics in `results/results.csv`, optimized programs in `artifacts/`, sanity-check pairs in `results/s3_sanity_pairs.json` and `results/synth_sanity_pairs.json`.

Per-run detail reports (superseded by this document, retained for full stage-level numbers): `SPIKE_REPORT.md` (Run 1), `SPIKE2_REPORT.md` (Run 2).
