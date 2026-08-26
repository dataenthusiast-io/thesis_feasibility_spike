# Deep Review: Issues and Inconsistencies (2026-08-25)

Scope: full read of `thesis_context_dump.md`, `thesis_proposal_draft.md`, `SPIKE_REPORT_CENTRAL.md` (including section 7), `thesis_literature_base.md`, `thesis_project_plan.md`. PDF sources not consulted, per instruction. Issues are numbered R1 to R24 for reference, grouped by severity. Each item states where it lives, what the problem is, why it matters, and a suggested fix. Nothing has been changed in your files.

## What holds up (checked, no action needed)

- All spike numbers quoted in the proposal and context dump match the spike report exactly (0.9967/0.975 AirDialogue clean, 0.5175/0.267 Banking77 clean, 0.884/0.370 shifted, +0.027 lift, 0.329 vs 0.370 brittleness, +0.98/-0.98 recovery deltas).
- The citation-scoping notes (Pan & Yang definitions only; Shumailov risk family only; two Huang papers kept distinct) are correctly and consistently applied in both proposal and literature base.
- Spike section 7 items 1, 4 and 5 are genuinely addressed in the proposal (judge protocol, held-out shift variant in Step 8, 3 to 5 seeds).
- The decision history is internally consistent; Iteration 8 reasoning correctly explains why the retention framing does not rescue AirDialogue.

---

## A. Critical: framing and measurement (decide before the supervisor conversation)

### R1. The retention ratio has a denominator problem that can invert conclusions
**Where:** Proposal Section 2 (primary measure), 3.5 (H2, H3).
**What:** Retention is defined per cell as shifted F1 divided by that cell's own clean anchor. Because optimized-static has a *higher* clean anchor than manual-static (0.531 vs 0.518), it is penalized twice: higher denominator and lower numerator. A configuration could in principle beat manual-static in absolute shifted F1 and still lose on retention ratio. The ratio and the absolute shifted score can order regimes differently, and H2/H3 currently do not say which ordering they claim.
**Why it matters:** This is exactly the directional-precision issue you flagged as load-bearing. An examiner can ask: "is the thesis about *retaining relative to your own peak* or about *who is best when the data drifts*?" Those are different practical questions and the practitioner cares about the second.
**Fix:** Keep the ratio as primary (it matches the retention framing) but (a) state explicitly that absolute shifted F1 is always co-reported and that any ratio/absolute rank inversion is analyzed, and (b) phrase H2 and H3 unambiguously in terms of one measure each. Alternative worth considering: define retention against a *common per-model anchor* (best clean score for that model) so all regimes share a denominator.

### R2. The adaptive regime has no clean anchor of its own, so its retention ratio is undefined as written
**Where:** Proposal 3.1 (grid: adaptive exists only under shift) vs Section 2 ("ratio of shifted-data macro F1 to the clean-data anchor of the same cell").
**What:** The adaptive cells have no clean-data measurement. Presumably the optimized-static clean anchor of the same model is borrowed, but this is nowhere stated, and the choice is not neutral: borrowing the *optimized* anchor (higher) disadvantages the adaptive regime against manual-static in ratio terms. In spike numbers: manual retention 0.715, adaptive 0.639 against the optimized anchor but 0.655 against the manual anchor.
**Fix:** One sentence in Section 2 fixing the anchor convention for adaptive cells, with rationale. This interacts directly with R1.

### R3. H3 is currently half-contradicted by the only measured evidence, and the proposal does not say so
**Where:** Proposal 3.5 H3; spike 5.4.
**What:** H3 claims adaptive retains more than *both* static regimes. The spike's single measured recovery cell supports adaptive > optimized-static (+1.0 pt, within noise) but shows adaptive *below* manual-static (0.339 vs 0.370). The proposal cites the brittleness evidence for H2 but stays silent on the fact that the same table points against half of H3.
**Why it matters:** Concealing an adverse directional signal in your own pilot data is examiner bait; surfacing it is a strength ("all outcomes are informative" is already your stance).
**Fix:** Add one honest sentence to H3: the spike's minimal-budget single-seed cell showed recovery over the frozen prompt but not over the manual prompt; whether proper budgets and seeds close that gap is precisely the open question. This also sharpens why bigger budgets are part of the design, not a luxury.

### R4. "Without new labels" is true only by construction of the instrument, and the stronger "label-free" wording overclaims
**Where:** Guiding question and Step 8 ("original labels, shifted inputs"); context dump H3 wording ("label-free, human-free").
**What:** MIPROv2 optimizes against a labeled dev set. The adaptive loop therefore *requires labeled data from the shifted distribution*. The design gets this for free only because the synthetic form shift preserves labels by construction, so the original labels remain valid. Real production drift offers no such guarantee: under topic drift or label shift, the old dev labels are wrong and the loop as designed cannot run. "No *new* labels" (proposal) is accurate; "label-free" (context dump, and implicitly the H3 narrative) is not.
**Why it matters:** This is the single biggest external-validity boundary of the thesis and it is currently implicit. It also silently narrows the title: the loop is validated for *form/register shift specifically*, not distribution shift in general.
**Fix:** (a) Purge "label-free" everywhere in favor of "without new labels"; (b) add an explicit assumption statement in 3.3 or 3.7: the loop presupposes meaning-preserving shift under which existing dev labels stay valid, and the conclusions are scoped to that shift class; (c) carry this into the limitations chapter plan.

### R5. The closed loop has no trigger, and the design never says that detecting drift is out of scope
**Where:** Proposal Section 2 ("without human intervention"), 3.7 out-of-scope list; L6 (Casimiro et al. carries exactly this trigger logic).
**What:** In the experiment, re-optimization is triggered exogenously by the experimenter. A production closed loop needs to *know* the distribution shifted (drift detection) before it re-optimizes. The thesis measures "does re-optimization work when triggered," not "can the pipeline autonomously self-improve." The out-of-scope list (fine-tuning, production data, multi-turn) does not mention trigger/detection.
**Why it matters:** "Self-improving... without human intervention" invites the question in the defense. You already cite the right source for the answer (Casimiro et al., when-to-adapt).
**Fix:** Add "drift detection / adaptation triggering" to 3.7 out-of-scope with one sentence of rationale, citing Casimiro et al. as where that question lives.

### R6. "Optimized against macro F1" is likely technically infeasible as stated
**Where:** Proposal Step 4 ("optimized with MIPROv2 against macro F1"); spike section 7 item 7.
**What:** MIPROv2 needs a per-example metric. Macro F1 is a set-level metric that does not decompose per example; the spike actually optimized exact-match accuracy on a class-balanced dev set as a proxy, and its section 7 explicitly logs this as a scope caveat. The proposal quietly upgrades the proxy to the real thing.
**Why it matters:** A methods chapter that claims to optimize a metric the optimizer cannot see is attackable; the proxy gap (balanced-accuracy proxy vs macro F1 target) is also a legitimate small threat to validity worth one sentence.
**Fix:** Reword Step 4: optimization target is per-example accuracy on a class-balanced dev set as a proxy for macro F1 (or a batch-level metric if you decide to implement one); state the proxy relationship openly.

### R7. AirDialogue's role in the actual experiment is undefined, with large effort implications
**Where:** Proposal 3.2 ("remains the low-complexity control anchor") and 3.1 (the grid has no corpus dimension).
**What:** Nothing specifies *which cells* run on AirDialogue. Full grid times two corpora roughly doubles the experiment; reusing only spike numbers means the control anchor rests on throwaway-quality code the plan says will be rebuilt. Between those extremes, nothing is fixed.
**Fix:** Decide and write down the AirDialogue slice explicitly, for example: manual-static only, clean plus one shift severity, 1B/3B tiers, rerun on the rebuilt harness. Cheap, preserves the continuity narrative, avoids the doubling.

### R8. The classical baseline has no measurable cells in the grid as drawn
**Where:** Proposal 3.1 grid (TF-IDF+SVM row: n/a in all three regime columns) vs H1 ("all static configurations") and Factor 1.
**What:** As displayed, the SVM is in the design but is never evaluated: every cell in its row is n/a. Presumably intended: trained once on clean data, evaluated clean and shifted, as a fourth regime-like column or a footnote. H1 implicitly includes it; the grid excludes it.
**Fix:** Give the baseline its own column ("trained-static: clean + shifted") or an explicit note under the table. Also decide whether H1 formally covers it.

## B. Major: method validity gaps (fix in the Exposé, before implementation)

### R9. Role-to-family allocation is unresolved and the current candidate list makes it tight
**Where:** Proposal 3.3 (generator/task/judge separation), 3.6 (family separation "across generator, task, proposer, and judge roles"); spike section 7 item 6; context dump open items.
**What:** The spike used llama3.1:8b as *both* shift generator and MIPRO proposer, and Llama 3.1 8B is simultaneously a mid-tier *task* candidate in 3.6. The constraint as stated needs four disjoint roles, plus a fifth family (or at least a second generator/instruction) for the held-out shift variant in Step 8, on top of task families Llama, Qwen, Phi, Gemma and a frontier API anchor. The allocation is satisfiable (Mistral, DeepSeek, OpenAI, Anthropic, Google remain) but nobody has actually assigned it, and some assignments have side effects (a weak proposer sabotages the E1 lesson; a local generator limits rewrite quality).
**Fix:** Add a small role-to-family table to 3.6: task families, generator family, proposer family, judge family, held-out-variant generator family. Resolve the Llama collision explicitly (either drop Llama 3.1 8B as task candidate or move generator/proposer elsewhere).

### R10. The severity ladder is authored, not calibrated, and has no manipulation check
**Where:** Proposal 3.3 ("parameterized severity ladder"); spike section 7 item 2.
**What:** Only one severity of form shift was ever measured. Nothing establishes that the planned ladder is monotone (severity 2 actually shifts more than severity 1) or how "severity" is quantified beyond the prompt wording. Corruption had a natural parameter; register distance does not.
**Fix:** Specify a manipulation check in Step 6: an independent quantitative signal per severity level (for example embedding distance to the original, judge-scored register/indirection rating, or lexical-overlap statistics), reported alongside the F1 curves. Without it, "graded severity" is an assertion.

### R11. The judge protocol checks label preservation but not content preservation
**Where:** Proposal 3.3 label-validity protocol; spike section 7 item 3.
**What:** The spike documented that the generator invents factual details ("i ordered something last week"), making the shift partly topical rather than purely stylistic. A rewrite can add invented content, stay label-consistent, and pass the judge, yet the instrument then no longer measures *form* shift. Spike item 3 is logged as directly feeding the label-drift mechanism, and the current protocol only catches its downstream symptom.
**Fix:** Extend the judge rubric to two criteria: label preserved AND no material content added/removed. Report both flag rates. This closes the last unaddressed high-impact item from spike section 7.

### R12. Exclusion/regeneration creates a selection effect that correlates with severity
**Where:** Proposal 3.3 (flagged pairs regenerated or excluded, exclusion rate reported).
**What:** Higher severities will produce more label-drift flags. Excluding flagged pairs systematically removes the *hardest* rewrites, so measured degradation at high severity is biased downward, and comparisons across severities compare differently-filtered populations. Regeneration until the judge passes has a milder version of the same effect. Per-class exclusion imbalance additionally distorts macro F1's class weighting.
**Fix:** Fix the policy in advance: prefer regeneration (bounded attempts) over exclusion; report exclusion per severity AND per class; keep test sets identity-matched across severities where possible (same original utterance appears at every severity) so comparisons are paired.

### R13. Dev set sizing for optimization is unspecified, though the spike flagged it
**Where:** Proposal Step 2 (test sizing only); spike section 7 item 9 (dev 2 per class; MIPRO's internal validation covered only part of the label space per trial).
**What:** The power spec covers test sets. The optimizer's dev set drives everything in Steps 4 and 8, and the spike's 154-example dev is explicitly logged as a precision bound. Banking77's train split permits far more.
**Fix:** Specify dev size and stratification per corpus in Step 2 (for example 5 to 10 per class for Banking77) and note the MIPRO internal-validation coverage consideration.

### R14. Seed-count power is not connected to the expected effect size
**Where:** Proposal Step 4 / power spec (3 to 5 seeds, observed variance about ±1 pt); spike recovery +1.0 pt.
**What:** If the true recovery effect stays near 1 point, 3 to 5 seeds against ±1 pt run-to-run variance is marginal for a significance claim on the seed dimension. The implicit bet is that larger budgets widen the effect (spike section 7 item 5), but the design never states the minimum detectable effect it is powered for.
**Fix:** Either state the expected effect size the budget increase should produce and note detectability, or commit to a conditional rule (if pilot deltas stay near noise, increase seeds; local calls are free, so seeds are cheap for the SLM cells, only the frontier cells pay).

### R15. The adaptive-on-clean cell is missing: adaptation cost in the other direction
**Where:** Proposal 3.1 grid (adaptive: shifted only).
**What:** Nothing measures what the shift-adapted prompt does on *clean* data. If drift reverts (or traffic is mixed), a loop that gains 2 points on shifted data but loses 8 on clean data is a bad production system. One extra evaluation per adaptive cell (no new optimization) answers it.
**Fix:** Add clean-test evaluation of the re-optimized programs to Step 8. Near-zero cost, closes an obvious defense question about the retention narrative.

### R16. Re-optimization initialization policy is undefined (loop vs restart)
**Where:** Proposal Step 8; "closed loop" framing throughout.
**What:** Does the adaptive regime warm-start from the optimized-static program or run MIPROv2 fresh on shifted dev? The spike restarted fresh. "Closed loop" and "self-improving" connote incremental adaptation of an existing artifact; a fresh restart is re-compilation. Both are defensible, but they are different mechanisms with different cost profiles and the text never chooses.
**Fix:** One sentence in Step 8 fixing the policy (and, if fresh restart, a brief note that this is the DSPy-native notion of re-compilation, defusing the terminology question).

### R17. Title scope vs instrument scope
**Where:** Title ("under Distribution Shift") vs 3.3 (one shift type: synthetic form/register shift, covariate shift with labels preserved).
**What:** The thesis operationalizes exactly one member of the shift taxonomy it cites (no label shift, no concept drift, no topic drift; the corruption axis is optional). The title's general claim is fine for a title, but the limitations chapter must scope conclusions to meaning-preserving covariate shift, and the WILDS/OOD-survey citations in L5 give the taxonomy language to do so.
**Fix:** No title change needed. Add the scoping sentence to 3.3 and park it for the limitations chapter. Interacts with R4.

## C. Statistical phrasing

### R18. H2 as phrased is a non-inferiority claim the stats plan cannot test
**Where:** Proposal 3.5 H2 ("retains *no more* quality than manual-static"); L14.
**What:** "No more than" is a null-style/equivalence-direction claim. The cited apparatus (bootstrap, Wilcoxon, Friedman) tests for *differences*; failing to find optimized > manual is not evidence of "no more than," and confirming H2 by absence of significance is the classic underpowered-null trap. The spike evidence actually suggests the stronger directional claim: optimized-static retains *less*.
**Fix:** Either make H2 directional ("optimized-static retains less quality under shift than manual-static"), which the spike supports and standard tests can assess, or keep the non-inferiority form and add equivalence testing (TOST) with a pre-registered margin to the Step 9 plan plus a supporting citation in L14. The directional version is cleaner and bolder.

### R19. Frontier API cells: seeds and determinism are ill-defined
**Where:** Proposal 3.1 (frontier anchor), Step 4 (seeded determinism).
**What:** Temperature 0 does not guarantee determinism for hosted API models, and "optimizer seed" controls only the local search, not the API model's sampling. The reproducibility claims in Step 1 are written as if all models were local.
**Fix:** One sentence acknowledging that frontier cells are reproducible in procedure but not bit-exact, with n repeats reported instead.

## D. Literature base housekeeping

### R20. The "Related-work chapter spine" section in the literature base is empty
**Where:** `thesis_literature_base.md`, heading present with no content; the proposal (Section 4) explicitly points to the base for the spine, and the spine text currently lives only in the proposal's closing paragraph.
**Fix:** Copy the spine (L3→L4→L6, L5→L7, L8→L9, L11, L12→L13, L1/L2/L10) into the base, since the base is declared the source of truth.

### R21. TextGrad appears in the proposal's L4 row but not in the canonical base
**Where:** Proposal Section 4 L4 ("TextGrad, Nature venue to verify: would be a journal anchor in the APO core") vs literature base L4 table (no TextGrad entry at all).
**What:** The condensed index contains an item the canonical file lacks, inverting the declared hierarchy. Note the stakes: if TextGrad's Nature venue is confirmed, it becomes the journal anchor L4 currently lacks (the base itself flags "APO SOTA is essentially unpublished in journals," which TextGrad would partially defuse).
**Fix:** Add TextGrad to the base L4 table with its verify flag.

### R22. Stale verification item: "Zhou et al., Neurocomputing volume/pages"
**Where:** Literature base, "Remaining verification needed," listed under L2.
**What:** No Zhou et al. Neurocomputing item exists anywhere in the base. The only Zhou et al. is APE (ICLR 2023) in L4. This looks like a leftover from a superseded version and pollutes an otherwise precise verification list. Also note the base's verification list and the proposal's differ slightly: the proposal lists TextGrad and NCAA; the base lists the Zhou ghost and NCAA but not TextGrad (consequence of R21).
**Fix:** Delete the Zhou line or restore whatever source it referred to; reconcile the two verification lists (one list, in the base, referenced by the proposal).

### R23. Gu et al. 2025 (*Patterns*) should join the verification list
**Where:** Literature base L12, marked "Confirmed."
**What:** This survey circulated as an arXiv preprint (2411.15594); a *Patterns* journal publication is plausible but is exactly the kind of venue claim the review has been verifying elsewhere, and it is currently the lead citation of L12. Marking it "Confirmed" without a locator (volume/article number) is inconsistent with the rigor applied to Bayer et al. (article number flagged) one section later.
**Fix:** Add volume/article number or move it to the verification list until confirmed.

## E. Cross-document synchronization

### R24. The project plan contradicts the current design in at least four places
**Where:** `thesis_project_plan.md` (known-stale per the context dump, but the specific deltas were never listed).
**What:**
- Phase 0 asks the supervisor to confirm "AirDialogue + controlled perturbation, second corpus optional": superseded (Banking77 primary, synthetic form shift primary, corruption optional secondary).
- Phase 2 works "the L1-L11 requirements table": now L1-L16, and the exit criterion should include the methodology concepts.
- Phase 4 build order omits the shift generator with judge-validation protocol, now the largest new build item per proposal 3.7.
- Phase 7's defense-asset list includes "why AirDialogue+noise over CLINC150": the actual decision to defend is "why Banking77 over AirDialogue (and CLINC150), why synthetic form shift over corruption, why LLM-as-judge over manual review."
Additionally, Phase 0's rule "literature largely complete BEFORE formal registration" is worth keeping visible: per the context dump, the supervisor conversation on the v4 design has not happened yet, so Phase 0 is formally still open while Phases 1 and 2 have largely run. Not a problem, but the plan's gating logic no longer describes reality.
**Fix:** Fifteen-minute resync pass over the plan once the supervisor conversation lands (doing it before risks a second resync).

---

## Suggested order of attack

1. Decide R1/R2/R18 together (they are one decision: what exactly H2/H3 claim, on which measure, against which anchor). This changes proposal text in Sections 2 and 3.5 only.
2. Fold in the one-sentence fixes: R3 (H3 honesty), R4 (label wording + assumption), R5 (trigger out-of-scope), R6 (metric proxy), R16 (init policy), R19 (API determinism).
3. Resolve the two scope decisions that affect effort: R7 (AirDialogue slice) and R9 (family allocation table). Both are supervisor-conversation material.
4. Method-protocol additions for the Exposé: R10 (manipulation check), R11 (content-preservation criterion), R12 (exclusion policy), R13 (dev sizing), R15 (adaptive-on-clean cell).
5. Literature housekeeping in one pass: R20 to R23.
6. R24 after the supervisor conversation.

---

## Resolution addendum (2026-08-25, decisions taken and applied)

All decisions below were discussed and confirmed by Nicolas before application; edits applied to `thesis_proposal_draft.md` and `thesis_literature_base.md` the same day. For the defense decisions log.

- **R1/R2 (measure):** Own-anchor retention ratio stays primary. Adaptive cells borrow the optimized-static clean anchor of the same model (conservative: holds the loop to the higher anchor). Absolute shifted F1 co-reported for every cell; ratio/absolute rank inversions analyzed explicitly. Alternatives considered: common per-model anchor, absolute-primary. Reason: preserves framing C while defusing the denominator objection.
- **R18 (H2 form):** H2 rephrased as a regime-by-data-condition interaction: the clean-data optimization lift does not survive shift (difference-in-differences, testable with the existing bootstrap plan). Alternatives considered: bold ranking claim ("retains less," no safety net), non-inferiority + TOST (machinery-heavy). Reason: directional and falsifiable, but confirmed even if optimized-static merely equals manual under shift; the ranking outcome is reported as effect size. Note: H2 concerns the *frozen* optimized prompt; the claim that continuous optimization wins under drift lives in H3 and is untouched.
- **R7 (AirDialogue):** Minimal control slice: manual-static only, two SLM tiers, clean plus one severity, rerun on the rebuilt harness. Alternatives: spike-numbers-only (rests on throwaway code), fuller grid (doubles effort for a corpus with no H2/H3 variance).
- **R9 (roles/families, revised once during discussion):** Task ladder Llama 3.2 1B / Qwen 2.5 3B / Qwen 2.5 7B / OpenAI frontier; generator Mistral; proposer DeepSeek (pilot-verified per the E1 lesson); judge Anthropic; held-out variant second instruction or Gemma-family generator. The first recommendation (Llama as generator, 1B dropped from tasks) was reversed after Nicolas pushed back: generator continuity is worth little because all shifted data is regenerated and the severity ladder recalibrated (R10) anyway, while dropping the 1B tier would kill the structured-output-validity finding. Spike findings remain valid either way as design evidence, not result cells.
- **Defaults applied without objection:** R16 fresh re-compilation (framework-native), R12 regeneration-over-exclusion with per-severity/per-class reporting and identity-matched severity sets, R14 conditional seed-increase rule, R6 accuracy-proxy wording for the optimization target, R19 API non-determinism note, plus R3, R4, R5, R8, R10, R11, R13, R15, R17 as recommended in the review above.
- **R20 to R23:** applied to the literature base (spine filled, TextGrad added with verify flag, stale Zhou item removed, Gu et al. locator moved to verification).
- **R24:** deferred until after the supervisor conversation, as planned. (Superseded: resynced 2026-08-25 at Nicolas's request; one light touch-up may follow the supervisor conversation.)

---

## Second addendum (2026-08-26): external draft review, items 1-9

Nicolas brought a nine-item review of the current draft; all items assessed, discussed, and applied same day. For the decisions log:

- **Item 1 (DeepSeek lineage trap), accepted with one correction:** locally runnable DeepSeek-R1 models are distills of Llama 3.1 and Qwen 2.5 bases (the task families), silently breaking family separation; proposer reassigned to Phi-4 14B (chosen over Gemma-2 9B to keep Gemma free for the Step 8 held-out variant); Phi-4-mini leaves the alternates. Correction to the item's reasoning: DeepSeek R1 proper is open-weights and Western-hosted, so the governance-collision argument was overstated; the disqualifiers are lineage plus the impracticality of a 600B MoE for a proposer role.
- **Item 2 (benchmark contamination), accepted with one added nuance:** limitation paragraph added to 3.2, per-model canary probe added to Step 5, citations added to L7 (Sainz 2023, Golchin & Surdeanu 2024, Magar & Schwartz 2022). Nuance added on direction: conservative for H3, but *flattering for H1* (part of measured degradation may be lost memorization advantage), so H1 effect sizes carry the caveat; regime comparisons on shifted data are largely immune (model-level effect).
- **Item 3 (judge vs severity ladder), accepted:** judge criteria reworded to label inferability plus content fidelity (no invented facts); deliberate omission of action naming licensed as a severity property, preventing regeneration pressure from compressing the ladder.
- **Item 4 (MIPROv2 vs GEPA), accepted as rationale + conditional check:** optimizer-of-record rationale in Step 4; optional GEPA robustness check on H3-critical cells, contingent on the Phase 5 budget review.
- **Item 5 (validity decomposition), accepted:** Step 9 co-reports per-cell validity and decomposes retention into validity times conditional accuracy; noted that this strengthens the H3 narrative (format failures are prompt-addressable).
- **Item 6 (tiny tier), decided:** Qwen 2.5 1.5B becomes tiny primary (fully family-controlled ladder); Llama 3.2 1B kept as a single cross-family robustness cell. Alternatives considered: keep Llama primary (family confound remains), Qwen-only (loses cross-family signal).
- **Items 7-9, accepted as one-sentence fixes:** exclusion propagation across severities; frontier model-version pinning and projected call budget; pre-registration wording for manual prompt freezing.
- Plan, briefing, and literature base updated accordingly. Nothing challenged the spine: retention framing, Banking77, synthetic instrument, and the grid all hold.
