# Report Rewrite Design — F1 Strategy Simulator Dissertation
**Date:** 2026-04-10  
**Status:** Approved  
**North star:** COMP3931 marking guidance — aim for ≥80% (Excellent) across all criteria  
**Stretch goal:** IJSPR publication readiness

---

## Context

The dissertation currently covers Phases 1–3 only. The full research arc spans Phases 1–10, with the most original contributions (incentive structure sweep, scaling boundary discovery, curriculum decomposition, difference rewards negative result) entirely absent from Chapters 2, 3, and 4. Appendix A (Self-appraisal) and Appendix B (External Materials) are both compulsory and currently missing.

The rewrite strategy is **Option B: RQ-centric restructure**. Chapter 4 is reorganised around the three research questions rather than ten phases sequentially. Every phase contributes evidence to a research question. The reader follows a logical argument, not a lab notebook.

---

## Guiding principles

1. **Take the reader on a journey** — accessible to a non-expert at the start of each section, deepening into technical detail as the argument builds.
2. **Every claim is backed** — by data from the experiments, a self-contained justification, or a citation. No unsupported assertions.
3. **Negative results are contributions** — Phase 10 (difference rewards) and Phase 6 (scaling failure) are presented as findings, not failures.
4. **Leeds Harvard referencing throughout** — `\bibliographystyle{agsm}` in config.tex; `agsm.bst` must be present in the report folder.
5. **Impersonal technical language** — "we" is acceptable; "I did this" is not.
6. **Don't stress about hitting exactly 30 pages** — write elegantly and completely; the user will review and prune.

## Prose style rules (mandatory for all LaTeX content)

These rules apply to every sentence written in Chapters 1 to 4 and Appendix A. Check each sentence before writing it.

**Punctuation:**
- No semi-colons in prose.
- No colons in prose. Colons are permitted in headings, table headers, and LaTeX list environments only.
- No em-dashes.

**Vocabulary to avoid (signs of AI writing):**
- Banned words: delve, nuanced, multifaceted, robust, leverage, pivotal, crucial, vital, comprehensive, groundbreaking, revolutionary, streamline, navigate, tapestry, foster, underscore, at the intersection of.
- Banned openers: "Additionally,", "Furthermore,", "Moreover,", "It is worth noting that", "It should be noted that", "In order to" (use "to"), "Due to the fact that" (use "because").

**Sentence construction:**
- One idea per sentence. If a sentence contains a comma joining two independent clauses, split it into two sentences.
- Short sentences are preferred. If a sentence exceeds 25 words, consider splitting it.
- Prefer active voice.

**British English spellings throughout:**
- behaviour, colour, recognise, optimise, generalise, analyse, modelling, labelling, programme (not program, except when referring to computer code).

---

## Reference additions (currently in refs.bib but uncited)

These must be woven into the rewrite at the locations specified:

| Key | Location | Purpose |
|-----|----------|---------|
| `tan1993multiagent` | Ch 1 §1.3 + Ch 4 §4.2 | Foundational IQL paper — anchors IL-MARL discussion |
| `wolpert2002optimal` | Ch 1 §1.3 + Ch 4 §4.4 | Primary citation for difference rewards formula (Phase 10) |
| `devlin2014potential` | Ch 4 §4.4 + §4.7 | Potential-based improvement; contextualises Phase 10 negative result |
| `agogino2004unifying` | Ch 4 §4.3 + §4.7 | Credit assignment framing for Phase 8 finding |
| `yang2018meanfield` | Ch 4 §4.4 | Mean-field approximation used in Phase 10A-i |
| `claus1998dynamics` | Ch 4 §4.3 | Cooperative learner dynamics; supports Phase 4 zero-sum collapse |
| `panait2005cooperative` | Ch 1 §1.3 | Cooperative MARL survey — strengthens literature coverage |
| `wang2022irat` | Ch 1 §1.3 + Ch 4 §4.3 | Individual reward + team reward mixing mirrors alpha mechanism |
| `schroeaderdewitt2020ippo` | Ch 4 §4.2 | IL effectiveness counterpoint in Phase 3 discussion |
| `son2019qtran` | Ch 4 §4.7 | QTRAN as alternative factorisation alongside QMIX in future work |

---

## Chapter-by-chapter plan

### Chapter 1 — Background Research (targeted updates, ~0.5 page net addition)

**What changes:**

1. **Section 1.3 MARL subsection** — add `tan1993multiagent`, `panait2005cooperative`, `wang2022irat` as cited works. One paragraph on cooperative MARL state of the art using these references.

2. **Section 1.4 — The Research Gap** — rewrite the four contributions paragraph to signal the full 10-phase arc:
   - Contribution 1: incentive structure as experimental variable (existing, keep)
   - Contribution 2: alpha sweep from fully competitive to fully cooperative (existing, expand)
   - Contribution 3: empirical discovery of the scaling boundary at N=4 (new)
   - Contribution 4: curriculum scheduling as a decomposition tool separating training-order from credit assignment failures (new)
   - Contribution 5: difference rewards negative result with mathematical proof (new)
   - Contribution 6: LLM semantic baseline as a reasoning vs learning comparison (new)

3. **Section 1.5 — Objectives** — add two objectives:
   - Obj 6: Characterise the agent-count boundary at which cooperative incentive transitions from beneficial to harmful, and determine whether this boundary is parametric or structural.
   - Obj 7: Evaluate difference rewards as an alternative cooperative formulation and establish whether the mean-field approximation preserves the cooperative gradient property.

4. **Section 1.5 — RQ hypotheses** — update H1, H2, H3 with one sentence each referencing Phase 4–10 evidence that will be presented in Chapter 4.

**What stays:** Everything else. The prose quality in Chapter 1 is already at a high standard.

---

### Chapter 2 — Methodology (one new section, ~1.5 pages)

**What changes:**

Add **Section 2.7: MARL reward modes and multi-phase experimental scope** immediately after Section 2.6 (Project Management). This section covers the methodology behind Phases 4–10:

**2.7.1 Alpha reward mixing**  
Formula: R_mixed = (1−α)·R_own + α·R_teammate  
Five alpha values tested: 0.0, 0.25, 0.50, 0.75, 1.0  
Justification: continuous sweep from purely competitive to purely cooperative allows threshold identification rather than binary comparison.

**2.7.2 Non-zero-sum game design**  
Structural change from 2-agent zero-sum to 2 DQN agents + 1 Base adversary.  
Why this matters: in zero-sum games the sum of rewards is structurally fixed; adding a common adversary allows both agents to win simultaneously, making genuine cooperative advantage measurable.  
Ablation design: replacing DQN agents with Base agents one at a time to isolate whether game structure or policy quality causes the observed effect.

**2.7.3 Multi-team scaling (Phases 6–7)**  
Extension to 4 DQN agents (2 per team) + 1 Base adversary.  
Phase 7A: 3 DQN + 1 Base (intermediate).  
Phase 7B: low-alpha sweep in 5-agent configuration.  
Rationale: identifies whether cooperation failure is agent-count-dependent.

**2.7.4 Curriculum alpha scheduling (Phase 8)**  
Three-stage schedule: 100 episodes competitive (α=0) → 300 episodes linear ramp → 100 episodes at target α.  
Motivation: tests whether training order, not incentive structure, causes the passivity failure observed at fixed α=0.75.

**2.7.5 Difference rewards formulation (Phase 10)**  
Wolpert–Tumer (2002) formula: D_i = G(z) − G(z_{-i})  
Mean-field approximation used: R_diff_i = own_delta + (own_delta − team_mean_delta)  
Gradient property under test: whether dR_i/d(d_j) > 0 (cooperative) or < 0 (competitive).

**2.7.6 LLM semantic baseline (Phase 9)**  
Zero-shot prompting via Claude Haiku. Race state serialised to natural language. No training, no replay buffer.  
Purpose: tests whether domain reasoning alone without learning matches trained RL performance; establishes a non-learning upper bound on strategy quality.

**What stays:** Sections 2.1–2.6 unchanged.

---

### Chapter 3 — Implementation and Validation (targeted additions, ~net zero change)

**What changes:**

1. **Fix the placeholder at line 232** — "Appendix Figure TBC" → either reference an actual appendix figure of the pipeline, or rewrite the sentence to describe the module relationships in prose without promising a figure. The latter is safer if time is short.

2. **Add Section 3.8: Phase 4–10 infrastructure extensions** (~1 page)  
   Organised as a brief inventory of what was built beyond the Phase 3 baseline:
   - Alpha mixing layer: implemented in simulator reward calculation, controlled via `reward.alpha` in config.json. No agent code changes required — fairness control preserved.
   - Multi-team competitor selection: `runtime_profiles.py` extended with `select_low_marl_team_competitors` supporting N-agent multi-team configurations.
   - Curriculum scheduler: episode counter with linear interpolation between alpha values; hooks into the existing `on_episode_end` callback.
   - Difference reward module: separate reward calculation path activated by `reward.mode = "difference"` in config. The algorithm confound encountered in Phase 10A-i (config.json still set to `"algo": "vanilla"` when rainbow-lite was intended) was caught by reading the `"algorithm"` field directly from output metric JSON files — illustrating the value of embedding config snapshots in telemetry output.
   - LLM agent: implements `BaseAgent` interface. Receives observation vector, serialises to natural language prompt, parses JSON response for action. Fallback to HOLD on parse failure.

3. **Compress Phase 0 test inventory** — replace the 12-test breakdown with a one-paragraph summary and a reference to Appendix C (where the full table moves). Saves approximately 0.5 pages in Chapter 3.

**What stays:** Sections 3.1–3.7 content unchanged except the placeholder fix.

---

### Chapter 4 — Results, Evaluation and Discussion (major restructure)

This is the centrepiece of the rewrite. Structure:

---

**4.1 Preliminary validation (target: ~1 page)**

A single compact table showing gate outcomes for Phases 0–2:

| Gate | Phase | Outcome | Key evidence |
|------|-------|---------|-------------|
| G0 | Phase 0 integrity | Pass | 12/12 tests; 3 defects found and corrected |
| G1 | Phase 1 smoke | Pass | Win rate >0.75 across 3 seeds; AGGRESSIVE flip identified |
| G2 | Phase 2 DQN benchmark | Pass | Rainbow-lite 0.835 CI95[0.817,0.852]; 0% collapse |

Brief prose: what each gate validated and why the gate structure prevents premature attribution of multi-agent effects to algorithm-level failures. This replaces the current Phase 1 and Phase 2 sections which were written in full detail — those details move to appendices.

---

**4.2 RQ1 — What strategic behaviours emerge and how robust are they? (target: ~3–4 pages)**

Evidence base: Phase 3 (IL-MARL, 36 core trials + 750-episode extensions + crossplay).

*Opening:* Set up the question — we have a stable algorithm (rainbow-lite from Phase 2) and a genuine MARL setting. What actually happens when two identical agents learn simultaneously?

*4.2.1 Collapse rates and the algorithm hierarchy*  
Table: vanilla (44%) = dueling (44%) >> double (11%) >> rainbow (0%).  
Mechanism: replay buffer contamination under positional asymmetry. More training makes it worse, not better — the 750-episode vanilla results proving collapse is irreversible within the training horizon once activated.  
Cite: `foerster2017stabilising`, `matignon2012independent`, `schroeaderdewitt2020ippo`

*4.2.2 Emergent equilibrium classes*  
Class 1 (degenerate): one agent converges to passivity. No strategic information.  
Class 2 (competitive): zone specialisation + hawk-dove risk equilibrium emerging from iterated best-response dynamics. Canonical example: rainbow s2/seed-202 at 750 episodes (ZDI = 0.405).  
Cite: `busoniu2008survey`, `tan1993multiagent`

*4.2.3 Non-stationarity vs stochasticity*  
Key finding: MARL non-stationarity from a concurrently adapting opponent is more destabilising than stochastic outcome noise. Rainbow handles both; vanilla handles neither.  
Crossplay bias finding (11pp structural A1 disadvantage) incorporated here.

*4.2.4 RQ1 answer*  
Concurrent training produces interpretable strategic structure — zone partitioning and hawk-dove roles — when the algorithm is collapse-resistant. H1 is confirmed with the qualification that zone specialisation is a training-budget effect at s2 (requires 750 episodes, not 500).

---

**4.3 RQ2 — How does incentive structure alter strategy, and is there a destabilisation threshold? (target: ~4–5 pages)**

Evidence base: Phases 4, 5, 6, 7, 8.

*Opening:* RQ1 established what happens with purely competitive agents (α=0). RQ2 asks: what happens when we change the rules of what agents are rewarded for?

*4.3.1 Phase 4 — Zero-sum alpha sweep: the catastrophic threshold*  
Table: collapse rates by alpha (0%→11%→11%→89%→100%).  
The transition at α=0.75 is sudden, not gradual — confirming H2's sharp threshold prediction.  
Why: in a zero-sum game, your teammate's gain is your loss. High alpha creates adversarial gradients. Agents receiving reward for something they cannot positively control converge to passivity.  
Cite: `claus1998dynamics`, `hu2003nash`

*4.3.2 Phase 5 — Non-zero-sum: the inverted-U curve*  
Changing game structure (adding Base adversary) eliminates collapse entirely at all alpha values.  
Table: joint beat-base rate by alpha (0.310 → 0.294 → 0.433 → 0.567 → 0.436).  
α=0.75 is optimal, not α=1.0 — agents need 25% individual incentive to maintain coherent individual learning gradients while benefiting from cooperative signal.  
Hawk-dove risk roles re-emerge under cooperation, now directed against the common adversary.  
Cite: `hughes2018inequity`, `nowak2006five`, `wang2022irat`

*4.3.3 Phase 6 — Scaling to 5 agents: cooperation fails completely*  
Table: both-beat-base rate by condition (all conditions worse than baseline α=0).  
α=0.75 produces −31% degradation. Cooperative team loses to competitive team in asymmetric condition.  
Root cause: credit assignment dilution. At N=3, each agent strongly influences one teammate. At N=5, 75% of reward comes from sources the agent barely controls — noisy Q-value updates produce caution rather than coordination.  
Cite: `agogino2004unifying`, `panait2005cooperative`

*4.3.4 Phase 7 — Boundary at N=4: a cliff, not a slope*  
Table: alpha=0.75 effect at N=3 (+83%), N=4 (+5%, noise), N=5 (−31%).  
Adding a single agent eliminates the entire cooperative advantage. This is a threshold effect — N=4 is the boundary.  
Low-alpha sweep confirms: even α=0.10 provides zero benefit in 5-agent game. There is no Goldilocks alpha. The failure is structural, not parametric.

*4.3.5 Phase 8 — Curriculum scheduling: decomposing the failure modes*  
Curriculum (0→0.75 ramp) recovers 86–89% of the caution pathology at N=4 and 69% of the performance loss at N=5.  
But cooperative advantage still does not emerge. Intra-team correlation remains negative.  
**Novel contribution:** this cleanly separates two failure modes previously conflated in the literature:  
1. *Training-order problem* (SOLVED by curriculum): immediate cooperative reward overwrites individual policies with passivity.  
2. *Credit assignment problem* (NOT SOLVED): at N≥4, agents cannot sufficiently influence teammates for shared reward to produce cooperative gradients. Requires architectural solutions.  
Cite: `agogino2004unifying`, `foerster2018coma`

*4.3.6 RQ2 answer*  
The destabilisation threshold exists and is structural. In zero-sum games: sharp collapse above α=0.5. In non-zero-sum games: cooperation is beneficial up to N=3, fails at N=4 regardless of alpha value or training schedule. H2 is confirmed.

---

**4.4 RQ3 — Does shared incentive produce genuine cooperative advantage? (target: ~3–4 pages)**

Evidence base: Phases 5, 9, 10.

*Opening:* RQ2 identified when and why cooperation fails. RQ3 asks the positive question: when it works, is the advantage genuine? And what happens when we try a fundamentally different cooperative formulation?

*4.4.1 Phase 5 — Proving causality of the cooperative advantage*  
The +83% joint beat-base rate at α=0.75 in the 3-agent game could be explained by game structure (adding a third agent changes the race dynamics regardless of alpha). The ablation eliminates this:  
- Two DQN agents at α=0 against Base: 0.310 joint beat-base  
- Two Base agents at α=0 against Base: significantly lower (near-random)  
- Two DQN agents at α=0.75 against Base: 0.567  
The α effect operates above and beyond the game structure effect. H3 is confirmed: genuine cooperative advantage exists in non-zero-sum settings.  
Cooperation lifts the weaker agent most — A2 jumped from 63% to 78% beat-base rate. This is coordinated resource partitioning, not redistribution.

*4.4.2 Phase 9 — LLM semantic baseline*  
Zero-shot Claude Haiku matches Rainbow-lite within 2% win rate with completely different zone selection and risk distribution.  
This result is significant in two directions:  
- It provides a non-trivial upper bound: trained RL with 500 episodes of experience does no better than zero-shot domain reasoning.  
- The strategies are orthogonal: LLM prioritises conservative attempts at easy zones; Rainbow diversifies. Two distinct routes to the same outcome performance — raising the question of whether win rate alone is the right metric for strategy evaluation.  
Cite: No standard RL citation; this is a novel comparison. Reference Claude Haiku model.

*4.4.3 Phase 10 — Difference rewards: a negative result as contribution*  
Wolpert and Tumer (2002) introduced difference rewards — D_i = G(z) − G(z_{-i}) — as a mechanism to give each agent credit only for its counterfactual contribution to the global outcome.  
The mean-field approximation implemented: R_diff_i = own_delta + (own_delta − team_mean_delta).  
**Mathematical proof that the formula creates competitive incentives:**  
∂R_i/∂d_j = −1/N < 0  
When teammate j overtakes, agent i's reward decreases because the team mean rises. This is the opposite of the cooperative gradient that Wolpert–Tumer intended. The approximation loses the cooperative property because it substitutes the team mean for the true counterfactual G(z_{-i}).  
Empirical confirmation: joint beat-base under difference rewards = 0.133 (rainbow-lite), a 16% regression from the IQL baseline of 0.158. Difference rewards rank last among all tested reward formulations.  
Algorithm effect (31%) > formula effect (16%): even with a cooperative formula, algorithm choice matters more at this scale.  
Cite: `wolpert2002optimal`, `devlin2014potential`, `yang2018meanfield`, `agogino2004unifying`

*4.4.4 RQ3 answer*  
Genuine cooperative advantage exists but is conditional: it requires non-zero-sum game structure (N=3), a cooperative gradient in the reward formula (not satisfied by the mean-field approximation), and a collapse-resistant algorithm. Incentive formula design matters as much as algorithm choice.

---

**4.5 Discussion (target: ~2 pages)**

*4.5.1 Positioning relative to the literature*  
- vs Tan (1993): IQL in cooperative games produces coordination at small N; this project extends to competitive settings and identifies the N boundary.  
- vs Wolpert & Tumer (2002): difference rewards require true counterfactual computation; mean-field approximation loses cooperative gradients — consistent with Devlin et al. (2014) finding that potential-based shaping preserves the property better.  
- vs Matignon et al. (2012): independent learner pathologies documented here (replay contamination, epsilon decay interaction) are consistent with the survey's predicted failure modes.  
- vs Foerster et al. (2017): stabilised experience replay addresses the same buffer contamination mechanism identified here; the Phase 3 finding that PER is the load-bearing anti-collapse component is consistent with their analysis.

*4.5.2 The three novel contributions*  
1. The N=4 scaling cliff — a threshold effect not a gradual degradation — identified empirically with a controlled ablation.  
2. Curriculum scheduling as a decomposition tool — separating training-order failure from credit assignment failure in a way that prior literature treated as a single problem.  
3. Mathematical proof that mean-field difference rewards produce competitive rather than cooperative incentives — a negative result that clarifies the conditions required for Wolpert–Tumer to work in practice.

*4.5.3 What the LLM result means*  
The LLM result raises a deeper question about what RL is learning when it matches zero-shot performance. In a sparse-decision five-lap race with one dominant zone, the strategy space is narrow enough that domain reasoning and Q-value optimisation converge on the same solution. This is not a failure of RL — it is evidence that the low-complexity tier has a near-optimal policy that is short enough to be described in natural language.

---

**4.6 Validity threats (target: ~1 page)**

- La Source structural dominance (difficulty 0.2 means near-certain success on every lap; win rate conflates strategy quality with zone exploitation)
- Evaluation protocol positional bias (11pp structural A1 disadvantage from fixed start position; collapse classification unaffected; future fix: round-robin start cycle in evaluation)
- Short race and sparse decisions (5 laps × 9 zones = 45 maximum decision events; zone specialisation requires 750 episodes not 500 at s2)
- Single circuit and single opponent type (findings generalise to Spa-style circuits with one dominant zone; may not hold on circuits with more balanced zone difficulties)
- Three seeds per cell (between-seed CI reflects variance well for ranking decisions at s0 where intervals are non-overlapping; less reliable at s1/s2)

---

**4.7 Future work (target: ~0.5 page)**

1. **QMIX and CTDE** — the architectural solution to the credit assignment problem identified in Phase 8. QMIX (Rashid et al., 2018) enforces a monotonicity constraint (∂Q_tot/∂Q_i ≥ 0) that preserves cooperative gradients regardless of agent count. QTRAN (Son et al., 2019) relaxes this to exact factorisation. These are the natural next step, now that curriculum scheduling has proven the training-order problem solvable independently.
2. **Potential-based difference rewards** — Devlin et al. (2014) show that potential-based shaping preserves cooperative gradients where mean-field approximation fails. Testing this formulation would complete the Phase 10 investigation.
3. **Medium and high complexity** — tyre degradation, pit stops, multi-car fields. The simulator is already designed for these tiers; the observation contracts are defined in config.json.
4. **Real telemetry integration** — calibrate zone difficulty values from lap-time sector data rather than position-change counts; test on circuits other than Spa.

---

### Appendix A — Self-appraisal (compulsory, no page limit)

**A.1 Project process reflection**

What worked well:
- The phased gate structure (G0→G1→G2) prevented premature attribution of multi-agent effects to algorithm failures. Each phase's result determined the next phase's design — a genuine empirical iteration rather than a pre-specified plan.
- Config-driven fairness control: externalising all experimental parameters to config.json meant every comparison was provably fair and reproducible.
- The decision to continue beyond Phase 3 rather than stop at the first successful result. Phases 4–10 produced the most original contributions — they would have been missed had the investigation stopped when a "good enough" result was found.
- Mathematical analysis alongside empirical results (Phase 10 gradient proof): the proof explained the empirical result and identified why the approximation failed, rather than simply reporting a performance regression.

What did not work well:
- The algorithm confound in Phase 10A-i: the config.json file still contained `"algo": "vanilla"` when rainbow-lite was intended, meaning the first batch of Phase 10 trials tested the wrong algorithm. The error was caught by reading the `"algorithm"` field in the output metric JSON — a reminder that config inspection at the start of every experiment batch is a necessary check, not an optional one.
- The report lagged the research by several phases for too long. Writing up Phases 4–10 retrospectively is harder than writing contemporaneously.

**A.2 Personal reflection and lessons learned**

- Negative results require as much rigorous analysis as positive ones. Phase 10's failure to produce cooperative behaviour via difference rewards is a contribution precisely because the mathematical analysis explains why — it is not just "this didn't work."
- The N=4 scaling cliff was not predicted. The willingness to design Phase 7 to characterise the boundary precisely (rather than treat Phase 6's failure as a dead end) was the decision that produced a genuine empirical finding rather than an inconclusive result.
- Reproducibility standards (Henderson et al., 2018; Agarwal et al., 2021) imposed upfront discipline that paid dividends at Phase 7 and 8: because all prior phases used fixed seeds and matched budgets, the Phase 7 ablation could confidently attribute the N=4 cliff to game structure rather than evaluation noise.

**A.3 Legal, social, ethical and professional issues**

*Ethical:* No human subjects were involved at any stage. No consent forms were required. The LLM baseline (Phase 9) uses the Claude Haiku API under Anthropic's standard terms of service; no personal data was transmitted.

*Legal:* All external libraries (FastF1, PyTorch, NumPy, Pandas) are open-source with permissive licences (MIT or BSD). The FastF1 data is publicly available Formula 1 telemetry for non-commercial research use, consistent with its licence terms. The Anthropic API was used under its commercial terms; no data was stored or repurposed beyond the research context.

*Professional:* The project follows the reproducibility standards established in the reinforcement learning literature (Henderson et al., 2018; Agarwal et al., 2021): matched compute budgets, fixed seeds, confidence interval reporting, and behavioural diagnostics. This is a professional obligation in empirical machine learning research, not merely a methodological preference. The git repository provides a full audit trail of experimental code and config changes.

*Societal:* The simulator models competitive decision-making between autonomous agents in a constrained environment. The broader societal context is the question of whether AI systems in competitive settings (not just motorsport, but financial markets, logistics, and any multi-agent strategic environment) develop emergent cooperative or exploitative behaviour depending on how their incentives are structured. This project provides controlled empirical evidence that incentive design — the rules of what agents are rewarded for — is at least as important as algorithm choice in determining emergent strategy. That finding has implications beyond motorsport for any domain where multiple autonomous systems interact under shared resource constraints.

---

### Appendix B — External Materials (compulsory)

| Material | Version/Date | Purpose | Licence |
|----------|-------------|---------|---------|
| FastF1 | v3.x | 2023 Belgian GP telemetry for zone difficulty calibration | MIT |
| PyTorch | v2.x | Neural network implementation for all DQN variants | BSD-3 |
| NumPy | v1.x | Numerical computation | BSD-3 |
| Pandas | v2.x | Metrics aggregation and analysis | BSD-3 |
| Anthropic Claude Haiku | API, 2024 | LLM semantic baseline (Phase 9) | Commercial API ToS |
| Git | — | Version control throughout | GPL |

Code repository available to supervisor and assessor at: [repository link to be added]

---

### Appendix C — Phase 0 test inventory (optional, moved from Chapter 3)

Full 12-test breakdown currently in Chapter 3 Section 3.4 moves here to free space in the main body.

### Appendix D — Full results tables (optional)

Detailed per-seed results tables for Phases 1–3 (currently in Chapters 3 and 4) move here. The main body retains summary tables only.

---

## Bibliography format fix

In `report/config.tex`, line 65:
```
\bibliographystyle{plainnat}   ← change to:
\bibliographystyle{agsm}
```

The `agsm.bst` file must be present in the `report/` directory. This is the standard Leeds Harvard style file.

---

## Spec self-review

- No TBD or TODO placeholders in this document.
- Sections are internally consistent: Chapter 4 structure matches the RQ framework set up in Chapter 1.
- Scope: this is a complete rewrite plan for one dissertation, not a decomposable multi-project spec.
- All requirements are unambiguous: each chapter section has a target page range, a list of content items, and citations to use.
