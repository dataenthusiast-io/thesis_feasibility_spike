# Master Thesis: Full Context Dump

Purpose of this file: complete context for a fresh AI session. Covers where the thesis idea came from, how it evolved, every major decision with rationale, current state, and open items.

## Who / setting

Nicolas, part-time Master's student at FOM University (Germany) alongside full-time work as Principal Data & AI consultant (currently embedded at an airline group as AI initiative lead, running among other things a production conversational-analysis pipeline; the thesis deliberately uses NO company data or artifacts). Thesis builds on a completed Transfer-Projekt (project work); up to 30% of it is formally reusable in the thesis.

## The foundation: the Transfer-Projekt

Title: "Large Language Models for the Analysis of Unstructured Conversational Data." Built and validated a zero-shot LLM pipeline (four phases: segmentation, classification, structured output, evaluation) classifying airline customer dialogues (AirDialogue dataset, 3 intent classes: book / cancel / change). Results: macro F1 = 0.993, schema failure rate 0.0%, projected cost ~$0.75 per 10k interactions (linear API cost model). Key limitation identified: ceiling effect (3 easy classes, everything saturates). Explicitly named open questions: architecture economics, prompt robustness, scalability.

## Evolution of the thesis idea (chronological, with decisions)

1. **Start: vague hybrid hypothesis.** Initial idea: "everything becomes conversational, ML+LLM hybrid is the future, best of both worlds." Rejected as unfalsifiable slogan; also "everything becomes conversational" demoted to motivation, not research claim.

2. **Three hybrid patterns examined.** A: confidence-based cascade (cheap model first, defer low-confidence to LLM; FrugalGPT lineage). B: LLM-as-teacher distillation (LLM labels corpus, classical model trains on synthetic labels). C: ensemble/arbitration (rejected: doubles cost, weak research object). B was initially recommended.

3. **Dataset problem recognized.** AirDialogue is saturated; hybrid/optimization value cannot be shown at a 0.993 ceiling. Fine-grained benchmarks examined: Banking77 (77 classes, semantically overlapping, empirically HARD: zero-shot LLMs ~low 70s, fine-tuned ~94), CLINC150 (150 classes + out-of-scope label, few-shot baselines ~86), HWU64, MASSIVE (multilingual incl. German), ATIS (rejected: old, saturated like AirDialogue). Interim choice: CLINC150 primary. Relevant related work found: Loukas et al. 2023 (LLM cost-performance on Banking77) = closest prior work, delta must be stated.

4. **Option D exploration (interview-sellability driven).** D1: model right-sizing across the spectrum (classical ML, SLMs, frontier API): quality-per-cost frontier, on-prem threshold. D2: OOS detection + taxonomy evolution. D3: synthetic data (rejected: risk). D4: pure decision framework (rejected: thin). D1 became the favored frame.

5. **Prompt optimization idea added (user's own production experience with eval loops).** Manual prompting is fuzzy/unmaintainable; DSPy/MIPROv2 etc. systematize it. Key literature: prompt brittleness is peer-reviewed fact (Sclar et al. 2024 ICLR: formatting alone shifts accuracy up to 76 points). Fusion insight: optimization might SUBSTITUTE for model scale (optimized small model ≈ frontier model?) — merges cost story (D1) with mechanism story.

6. **Course correction.** A draft over-pivoted to prompt optimization as the thesis; corrected to: right-sizing = spine, prompt optimization = dimension of the experimental grid.

7. **User's broader narrative fixed as the intro arc** (validated by research): conversational data exploding (70-90% unstructured figures are grey literature, cite as ranges; CSA 2026 survey diverges) → enterprises must analyze → AI can → but resources limited in EVERY dimension (cost volume-linear even as unit prices fall 9x-900x/yr per Epoch AI; latency; EU AI Act/residency; engineering attention on prompt maintenance) → best-fit architecture is the real problem. Falling-price objection is pre-handled: reframe from "LLMs expensive" to "economics volume-linear and multi-dimensional."

8. **Supervisor feedback (pivotal):** wants (a) quantitative justification "which SLM and why", (b) explicit experiment grid sketch, (c) Step 1 = original dataset (2-3+ SLMs), expect ~0.993, (d) Step 2 = inject noise into original data OR noise a second dataset. This moved the design from "harder dataset" to "controlled degradation of AirDialogue" — methodologically nicer (parametric noise control, labels preserved for free).

9. **Self-healing idea (user) + deflation (assistant).** User: make it forward-looking, recursive/self-improving/self-healing pipeline that survives production drift. Research showed: "self-improving pipelines" is DSPy's own subtitle (the anchored term); "distribution shift" is the correct academic term (Li et al. 2023 EMNLP); "self-healing" is autonomic-computing vocabulary (Kephart & Chess 2003, keep as one discussion reference); "recursive/recurrent" rejected for the title (recurrent = RNN collision; recursive = technically wrong + AI-safety baggage). Critically: self-healing was deflated from a thesis identity / separate phase into ONE FACTOR LEVEL of a factorial experiment. Also noted: recent work shows LLM optimization loops can be unstable → quantified limits of the adaptive regime are a finding, not a failure.

## Final design (current state)

**Title:** Self-Improving Language Model Pipelines for Conversational Intent Classification under Distribution Shift

**Factorial comparative experiment.** Factor 1 model: TF-IDF+SVM (classical static baseline, trained not prompted) | SLM ~1B | ~3B | ~7-8B (local, Ollama/vLLM) | frontier API (anchor). Factor 2 prompt regime (LM conditions only): manual-static | optimized-static (APO on clean data, frozen) | optimized-adaptive (re-optimized after shift). Data condition: clean AirDialogue | shifted (parameterized perturbation N1-N3: typos, ASR-style errors, truncation, colloquial rephrasing; label-preserving by construction → adaptation is label-free).

**Two carrying interactions:** (1) optimization gain x model scale on clean data (does optimization substitute for scale?); (2) prompt regime x data condition under shift (does the closed loop beat static engineering?). Hypotheses: H1 diminishing quality-per-cost above small tier; H2 APO lift inversely related to scale; H3 adaptive regime retains significantly more F1 under shift than both static regimes, at measurable re-optimization cost; classical baseline quantifies non-adaptable degradation.

**Explicitly out of scope (design decisions, have rationale ready):** SLM fine-tuning (would confound prompt-level vs weight-level adaptation, blow scope, reintroduce the labeling/retraining cost the loop avoids); company/production data; multi-turn dialogue state.

**SLM candidates:** Llama 3.2 1B / Gemma 2 2B (tiny), Qwen 2.5 3B / Phi-4-mini (small), Qwen 2.5 7B / Llama 3.1 8B (mid). Criteria: open weights, ≤16GB VRAM quantized, 3 tiers, structured-output capable, workable license. Pre-select via published scores + pilot; clean-data run is the definitive task-specific selection (benchmark contamination caveat).

**Literature approach:** requirements-driven (L1-L11 anchor claims table in the proposal: data growth, paradigm evolution, prompt sensitivity, APO methods, distribution shift/robustness, closed-loop systems + loop instability, benchmarks, SLMs, inference economics, EU AI Act governance, closest prior work). Verified peer-reviewed anchors: Sclar 2024 (ICLR), Khattab 2023 DSPy (ICLR24), Opsahl-Ong 2024 MIPROv2 (EMNLP), Pryzant 2023 (EMNLP), Zhou 2022 APE (ICLR23), OPRO (ICLR24), Li 2023 robust PO under shift (EMNLP), Larson 2019 CLINC150 (EMNLP), Casanueva 2020 Banking77 (ACL ws), Wei 2018 AirDialogue (EMNLP), Hsieh 2023 (ACL Findings), Gilardi 2023 (PNAS). Venue-to-verify: TextGrad journal, FrugalGPT, Loukas 2023. Grey (motivation only): Gartner/IDC, Epoch AI.

## Artifacts produced so far

1. `thesis_proposal_draft.md` — send-ready draft: title, basis/continuity (30% reuse), synopsis (incl. three-stage sentence), 6-step motivation chain, research aim + guiding question, factorial design (3.1 factors, 3.2 nine-step procedure, 3.3 hypotheses, 3.4 model selection, 3.5 implementation scope incl. out-of-scope rationale), L1-L11 literature requirements table, preliminary references appendix.
2. `spike_spec.md` — feasibility spike spec for Claude Code: confirms A1 (local SLM + valid JSON), A2 (DSPy/MIPROv2 end-to-end vs macro F1), A3 (noise measurably degrades), A4 stretch (re-optimization recovers). Qwen 2.5 3B via Ollama, 150 dev / 300 test AirDialogue slices, fixed labels [book, cancel, change], hard caps, SPIKE_REPORT.md deliverable.

## Style/working preferences relevant for a new session

- No em dashes or en dashes ever in outputs.
- Wants sparring: genuine pushback, not agreement; but also over-pivots happen, so confirm hierarchy changes before rewriting ("X stays the spine, Y becomes a dimension").
- Academic register for prof-facing docs; no internal commentary in deliverables.
- Values: interview-sellable framing ("IP enterprises care about regardless of vertical"), cost-conscious/skeptic-of-hype positioning, forward-looking systems angle.
- Prof feedback so far is supportive; wants quantitative SLM justification and the grid made explicit.

## Open items

- Prof discussion of the current draft (decisions to extract: corpus confirmation, adaptive regime as core, FOM Exposé formalities/registration timing).
- Run the spike (before deep literature).
- Verify flagged venues (TextGrad, FrugalGPT, Loukas).
- Literature phase L1-L11, then formal Exposé, then implementation.
- Maintain a decisions log (this file is its seed).
