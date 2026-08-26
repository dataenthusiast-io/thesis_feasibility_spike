# Master Thesis: Full Context Dump (v2)

Purpose of this file: complete context for a fresh AI session. Covers where the thesis idea came from, how it evolved, every major decision with rationale, current state, and open items. Supersedes the earlier v1 dump.

## Who / setting

Nicolas, part-time Master's student at FOM University (Germany) alongside full-time work as Principal Data & AI consultant (currently embedded at an airline group as AI initiative lead, running among other things a production conversational-analysis pipeline; the thesis deliberately uses NO company data or artifacts). Thesis builds on a completed Transfer-Projekt (project work); up to 30% of it is formally reusable in the thesis.

## The foundation: the Transfer-Projekt

Title: "Large Language Models for the Analysis of Unstructured Conversational Data." Built and validated a zero-shot LLM pipeline (four phases: segmentation, classification, structured output, evaluation) classifying airline customer dialogues (AirDialogue dataset, 3 intent classes: book / cancel / change). Results: macro F1 = 0.993, schema failure rate 0.0%, projected cost ~$0.75 per 10k interactions (linear API cost model). Key limitation identified: ceiling effect. Explicitly named open questions: architecture economics, prompt robustness, scalability.

## Evolution of the thesis idea (chronological, with decisions)

### Part 1: finding the topic (pre-spike)

1. **Start: vague hybrid hypothesis** ("everything becomes conversational, ML+LLM hybrid is the future, best of both worlds"). Rejected as unfalsifiable slogan.

2. **Three hybrid patterns examined.** A: confidence-based cascade. B: LLM-as-teacher distillation. C: ensemble/arbitration (rejected). B initially favored.

3. **Dataset ceiling problem recognized early** (AirDialogue too easy). Fine-grained alternatives scouted: Banking77, CLINC150, HWU64, MASSIVE, ATIS (rejected, saturated like AirDialogue). Loukas et al. 2023 identified as closest related work on Banking77.

4. **Option D exploration** (interview-sellability driven): D1 model right-sizing across the spectrum (favored), D2 OOS detection, D3 synthetic data (rejected), D4 pure decision framework (rejected).

5. **Prompt optimization idea added** (user's own production experience with eval loops). Fusion insight: optimization might substitute for model scale.

6. **Course correction:** right-sizing = spine, prompt optimization = dimension of the grid (not the other way around).

7. **User's narrative arc fixed as the intro chain** (validated by research): conversational data exploding -> enterprises must analyze -> AI can -> but resources limited in every dimension -> best-fit architecture is the real problem. Falling-price objection pre-handled.

8. **Supervisor feedback (pivotal):** wants quantitative "which SLM and why," an explicit grid sketch, Step 1 = original dataset with 2-3+ SLMs (expect ~0.993), Step 2 = inject noise into original data OR noise a second dataset. Moved design toward controlled degradation of AirDialogue rather than switching to a harder dataset outright.

9. **Self-healing idea (user) + deflation (assistant).** Researched terminology: "self-improving pipelines" is DSPy's own subtitle (anchor term used in the title); "distribution shift" is the correct academic term (Li et al. 2023, EMNLP); "self-healing" is autonomic-computing vocabulary (Kephart & Chess 2003, kept as one discussion reference); "recursive/recurrent" rejected for the title (RNN collision / technically wrong + AI-safety baggage). Self-healing deflated from a separate phase/identity into ONE FACTOR LEVEL of a factorial experiment (the "optimized-adaptive" prompt regime). Loop-instability caveat noted early.

**State at end of Part 1:** proposal draft v3, factorial design (model x prompt regime x data condition), title "Self-Improving Language Model Pipelines for Conversational Intent Classification under Distribution Shift," AirDialogue + noise as planned corpus/instrument, quantitative-experiment-plus-comparative-analysis framing (not "building a self-healing system").

### Part 2: the feasibility spike (pivotal, changed the design substantially)

10. **Spike spec written** for Claude Code: confirms four assumptions (A1 local SLM structured output, A2 DSPy/MIPROv2 end-to-end, A3 noise creates measurable degradation, A4 stretch: re-optimization recovers). Fixed labels [book, cancel, change], Qwen 2.5 3B via Ollama, hard caps, SPIKE_REPORT.md deliverable. Context block, exact task definition, and Ollama/DSPy plumbing added after a self-review pass to make it Claude-Code-ready cold.

11. **Spike executed, two runs, one day, fully local (~10,000 LM calls, ~2h, zero API cost).** Results, condensed:
    - **A1, A2: GO.** 3B model hits F1 0.997 on clean AirDialogue, 100% JSON validity; DSPy/MIPROv2 runs end-to-end within budget.
    - **Data derivation bug found and fixed:** naive "first two turns" slicing produced 12% intent-free texts scored as model failures; fixed by an intent-bearing-content rule. Lesson generalized into the method (Step 2 of the procedure).
    - **AirDialogue confirmed saturated across the SLM range** (0.997 at 3B, 0.975 at 1B clean): H1/H2 have no variance there. This matched the supervisor's own prediction.
    - **Parametric noise instrument mostly failed:** only near-destructive corruption moved the metric; realistic severities cost ~3pts; the lost information was unrecoverable by prompts (A4 corruption cell: -0.98pts, no recovery).
    - **The supervisor's own synthetic-rewrite idea was tested as an alternative instrument and worked much better:** LLM-rewritten casual/indirect rephrasing degraded quality MORE than max corruption (-11 to -20 pts) while staying realistic and format-clean, and the loss was prompt-shaped (recoverable in principle).
    - **Banking77 tested as an alternative corpus:** restored headroom (0.518 clean at 3B vs 0.997 on AirDialogue), 25pt scale separation, and made DSPy optimization measurable (naive config: +0.000; corrected config with a stronger proposer model and no few-shot demos: +2.7pts). This promoted "optimizer configuration" to an explicit experimental variable.
    - **Key new finding (brittleness):** optimized-static prompts degraded MORE under shift than manual prompts and ended up below them (0.329 vs 0.370 on shifted Banking77). Optimized-static overfits the clean distribution. This became the empirical basis for H2/H3.
    - **Recovery signal:** +1.0pt under form shift (vs -1.0 under corruption), single seed, within optimizer noise, tested on a matched shift distribution (best case, flagged as a limitation to fix in the real design).
    - **Label-drift risk surfaced:** synthetic rewriting occasionally blurred or flipped intents; this became a first-class method requirement (validation protocol), not a footnote.
    - Full findings in `SPIKE_REPORT_CENTRAL.md` (uploaded by user, read in full).

12. **Post-spike decision round.** Two big questions resolved together:
    - **Corpus: switched to Banking77.** User explicitly fine with this ("no problem w vertical, keep the thesis vertical agnostic, we just pick this bc it's best fit"); both corpora's spike numbers are kept in the thesis as data-driven selection evidence (turns a preference into evidence, also lets AirDialogue survive as the low-complexity control anchor tying back to the Transfer-Projekt).
    - **Label validation: LLM-as-judge instead of manual review**, per user's explicit request. Design: frontier model from a third model family (distinct from generator and task models) judges 100% of rewritten pairs; flagged pairs excluded/regenerated; small human-checked anchor set verifies the judge itself once. Grounded in Zheng et al. 2023 (NeurIPS) and later a full L12 literature cluster.
    - **Goal framing precision (user's explicit ask, the most important structural decision of this phase):** three candidate framings laid out - (A) optimization thesis ("APO beats manual"), (B) robustness comparison (purely descriptive), (C) retention thesis (does closed-loop re-optimization retain the clean-data quality anchor under shift, where static configurations degrade). **Framing C adopted.** Retention ratio (shifted F1 / clean anchor per cell) became the primary measure; clean-data optimization lift and cost frontier demoted to secondary/supporting. Hypotheses reordered: H1 degradation (all static configs degrade), H2 brittleness (optimized-static is no better than manual-static under shift), H3 retention through adaptation (core: optimized-adaptive retains significantly more, at reportable cost, label-free and human-free). Title required no change: "self-improving" (mechanism) + "under distribution shift" (condition) already implied retention as the connecting objective; only the aim statement needed to say it explicitly.

13. **Proposal fully rewritten (v4) around framing C** with all spike evidence woven in: data-driven corpus-selection table (3.2), shift-instrument section with the corruption-vs-form-shift finding (3.3), LLM-as-judge protocol as a first-class method element, optimizer configuration as a controlled variable, reordered hypotheses (3.5), matched-distribution caveat addressed via a held-out shift variant in the procedure (Step 8), power spec (1000+ examples/cell, 3-5 optimizer seeds, ±1pt variance noted).

### Part 3: literature phase (in progress)

14. **User ran an external literature review process** (two passes: conference/top-tier + journal-prioritized) and produced a distilled literature base covering L1-L13, pasted for review.

15. **Assistant reviewed it critically** against the proposal's actual claims. Verdict: strong journal-first strategy, honest tiering, good chapter-mapping. Four fixes applied: (a) restored Li et al. 2023 EMNLP ("robust prompt optimization against distribution shifts") into L5 as the closest prior work on the thesis's exact question, which had been dropped; (b) added Huang et al. 2024 ICLR ("LLMs cannot self-correct reasoning yet") to L6 as peer-reviewed backing for the loop-instability/"all outcomes are informative" framing; (c) resolved Weld et al. (L7) to ACM Computing Surveys (Q1), closing a venue-TBC flag; (d) flagged GEPA's ICLR 2026 acceptance and TextGrad's journal venue as still needing verification. Two citation-scoping cautions issued and preserved as standing notes: Pan & Yang (L5) anchors *definitions* only, not LLM-era operational claims (WILDS and NLP-OOD surveys carry the operational framing); Shumailov et al. (*Nature*, L13) anchors the *synthetic-data-degeneration risk family*, not direct evidence for the thesis's evaluation-data case (it concerns training-data collapse). Both kept, not dropped, but scoped precisely, since misapplied they'd be examiner bait but scoped correctly they're two of the strongest journal citations in the thesis.

16. **User asked whether ~50 sources is "enough for a whole thesis."** Answer: no, and not meant to be, this is the anchor skeleton (1-5 sources per structural claim); a finished FOM thesis lands at 80-120 references, filled in claim-by-claim during writing. Three genuine gaps identified at that point: statistics/significance-testing methodology (nothing cited despite promising bootstrap tests and seed variance), experimental-design methodology (no citation for "why a factorial experiment"), and macro F1 under class imbalance (undefined metric choice). Offered as L14-L16.

17. **User ran a second literature pass specifically for these methodology gaps** (M1 stats/seed-variance, M2 experimental design, M3 macro F1), pasted for review alongside a re-paste of the L1-L13 base. Assistant reviewed: strong, well-scoped picks (Dror et al. 2018 ACL + Berg-Kirkpatrick et al. 2012 for significance testing; Wohlin et al., Springer monograph, for experimental design, explicitly the most examiner-expected citation for that gap; Sokolova & Lapalme 2009, Q1 IPM, for macro F1). Accepted as-is.

18. **Structural decision: keep literature as a separate file, not merged into the proposal.** Rationale: the proposal is a discussion/Exposé-track document meant to stay readable in one sitting (grew from an intended 2-pager but should not become a bibliography); the literature base is a working document that will keep growing through the entire writing phase past 50 references. Two-layer structure implemented: `thesis_literature_base.md` (canonical, full annotated bibliography, L1-L13 + M1-M3 merged from both user passes, includes the citation-scoping notes, the related-work chapter spine, and the outstanding-verification list) and `thesis_proposal_draft.md` (condensed index table, L1-L16 including the three new M-derived rows, one line pointing to the full file as source of truth).

## Final design (current state, v4 + literature)

**Title:** Self-Improving Language Model Pipelines for Conversational Intent Classification under Distribution Shift

**Aim (framing C, retention):** quantify the extent to which a closed prompt-optimization loop retains classification quality under distribution shift, compared to static configurations, across model scales, including re-optimization cost. Primary measure: retention ratio (shifted F1 / clean anchor per cell) plus recovery delta of adaptive over frozen on identical shifted data. Guiding question: "To what extent can closed-loop prompt re-optimization retain the classification quality of language model pipelines under distribution shift, without human intervention or new labels?"

**Factorial design.** Factor 1 model: TF-IDF+SVM (classical static baseline, trained not prompted) | SLM ~1B | ~3B | ~7-8B (local, Ollama/vLLM) | frontier API (anchor). Factor 2 prompt regime (LM conditions only): manual-static | optimized-static (APO on clean data, frozen) | optimized-adaptive (re-optimized on shifted dev data). Data condition: clean | shifted (LLM-synthetic form shift, graded severity). Optimizer configuration (proposer model, demo policy) is a controlled, documented variable, not left implicit.

**Corpus: Banking77 primary** (77 fine-grained, semantically overlapping intents; structural fit chosen over vertical concerns, both corpora's spike measurements serve as the documented, data-driven selection evidence). **AirDialogue: control/continuity anchor** (low-complexity, connects to Transfer-Projekt, demonstrates the ceiling effect empirically).

**Shift instrument: LLM-synthetic form shift** (casual/indirect register rewriting, graded severity), validated by the spike as realistic and prompt-shaped-recoverable, unlike parametric corruption (demoted to optional secondary axis, preserves labels by construction). **Label-validity protocol (first-class method element):** generator/task/judge model-family separation; full-coverage LLM-as-judge validation by a frontier model from a third family; small human-checked anchor set verifying the judge itself once.

**Hypotheses:** H1 shift degrades all static configurations. H2 optimized-static retains no more than manual-static under shift (brittleness; directional spike evidence already exists: 0.329 vs 0.370). H3 (core) optimized-adaptive retains significantly more than both static regimes at reportable cost, label-free, human-free. Secondary: clean-data APO lift positive and inversely related to scale; quality-per-cost falls above the small-model tier.

**Explicitly out of scope (design decisions, rationale ready):** SLM fine-tuning (would confound prompt-level vs weight-level adaptation, reintroduces the labeling/retraining cost the loop avoids); company/production data; multi-turn dialogue state.

**Power spec:** 1000+ test examples per cell where the corpus permits, 3-5 optimizer seeds per condition (spike-observed run-to-run variance ~±1pt). Recovery tested against a held-out shift variant, not only the matched distribution the spike used (spike limitation, fixed in the real design).

**SLM candidates:** Llama 3.2 1B / Gemma 2 2B (tiny), Qwen 2.5 3B / Phi-4-mini (small), Qwen 2.5 7B / Llama 3.1 8B (mid). Two tiers already have spike-measured Banking77 numbers (3B: 0.518, 1B: 0.267 clean).

## Literature (current state)

Two-file structure: `thesis_proposal_draft.md` carries a condensed L1-L16 index; `thesis_literature_base.md` is the canonical, full annotated bibliography (source of truth going forward). Topics: L1 data growth, L2 NLP paradigm evolution, L3 prompt sensitivity, L4 APO methods, L5 distribution shift/robustness (incl. Li et al. 2023 as closest prior work), L6 closed-loop/self-improving systems (incl. Huang et al. 2024 instability), L7 intent benchmarks, L8 SLMs/quantization, L9 inference economics (FrugalGPT confirmed TMLR), L10 EU AI Act/GDPR governance, L11 closest baseline (Loukas et al. 2023), L12 LLM-as-judge, L13 synthetic data/label-drift risk, L14 significance testing, L15 experimental design methodology, L16 macro F1 metric. Journal-first strategy throughout; conference-dominance stated openly where genuine (L3, L4-SOTA, L11, M1/L14). Citation-scoping notes fixed for Pan & Yang (definitions only) and Shumailov (risk family only, not direct evidence). Outstanding verification: Hassani et al. Q-rank, GEPA ICLR 2026 acceptance/figures, TextGrad journal venue, NCAA review authorship, IJLIT backup item, Bayer et al. article number.

## Artifacts produced so far

1. `thesis_proposal_draft.md` — current send-ready draft (v4, framing C, post-spike, post-literature): title, basis/continuity, synopsis, 6-step motivation chain, research aim + guiding question + primary measure, factorial design (3.1 factors/grid, 3.2 data-driven corpus selection, 3.3 shift instrument + label-validity protocol, 3.4 nine-step procedure, 3.5 hypotheses, 3.6 model selection, 3.7 implementation scope incl. out-of-scope rationale), condensed L1-L16 literature index pointing to the full base.
2. `thesis_literature_base.md` — canonical working bibliography, L1-L13 + M1-M3(as L14-16), venue tiers, per-item placement notes, related-work chapter spine, citation-scoping notes, outstanding verification list.
3. `spike_spec.md` — feasibility spike spec given to Claude Code (executed successfully; superseded as a live task, kept as documentation of the method).
4. `SPIKE_REPORT_CENTRAL.md` — the executed spike's own output (user-provided, not assistant-authored); contains the full decision history (9 iterations), per-run evidence tables, and a logged list of assumptions still requiring verification before the real experiment (label-preservation rate, shift-severity calibration, generator topical drift, matched-distribution best-case recovery, single-seed optimizer variance, generator/proposer coupling, narrow method scope, one unmeasured cell, sample-size caveats). This file should be treated as primary evidence, not a summary; several of its section-7 caveats are directly load-bearing for the thesis's methods chapter (e.g., the recovery-against-held-out-shift-variant requirement is already reflected in proposal Step 8, but the rest of section 7 should be reread when finalizing 3.4).

## Style/working preferences relevant for a new session

- No em dashes or en dashes ever in outputs.
- Wants genuine sparring/pushback, not agreement; but watch for over-pivots, confirm hierarchy changes explicitly before rewriting ("X stays the spine, Y becomes a dimension/factor") rather than silently swapping which idea leads.
- Academic register for prof-facing docs; no internal commentary or meta-narration in deliverables sent externally.
- Values: interview-sellable framing ("IP enterprises care about regardless of vertical"), cost-conscious/skeptic-of-hype positioning, forward-looking systems angle, but corrects hype language back to precise academic terms when pushed (self-healing -> self-improving/adaptive; wants exact directional clarity between "better than manual," "clean vs shifted," and "retain quality," and considers this precision load-bearing from title to conclusion).
- Wants a decisions log maintained for the eventual defense (this dump plus the spike report's decision history section serve that purpose).
- Comfortable directing significant technical/empirical work (the spike) to be done by Claude Code from a written spec, then reviewing results before applying them to the thesis design; expects the same "report before applying" discipline from the assistant when a document's implications are non-trivial.

## Open items

- Supervisor discussion of the current (framing C, Banking77, LLM-as-judge) proposal not yet held; extract from it: confirmation of the corpus switch and framing C, FOM Exposé formalities/registration timing, and reaction to the vertical-agnostic argument for Banking77.
- Real (non-spike) implementation not yet started: spike code exists but was explicitly scoped as throwaway-quality; Phase 4 (implementation) rebuilds properly per the project plan.
- Verify flagged literature items (GEPA, TextGrad, Hassani Q-rank, NCAA authorship, IJLIT item, Bayer article number).
- Re-review `SPIKE_REPORT_CENTRAL.md` section 7 (logged assumptions) when finalizing the methods chapter; several items there are not yet fully addressed in the proposal (e.g., item 6, generator/proposer coupling in the recovery cell; item 3, generator topical drift feeding label drift).
- Literature phase continues (target 80-120 references by Exposé); related-work chapter can be drafted once L1-L16 reading is substantially complete.
- A separate `thesis_project_plan.md` (phases 0-7, no timing) exists from before the spike and remains directionally valid but has not been re-synced with spike/literature learnings; worth a light revisit in a future session.
