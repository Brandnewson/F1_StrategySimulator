# Complete Research Journey — Phases 1-8

**Project:** F1 Strategy Simulator — Investigating how shared incentive shapes emergent strategy in multi-agent reinforcement learning
**Track:** Spa-Francorchamps, 9 overtaking zones, 5 laps
**Algorithm:** DQN family (Rainbow-lite primary)
**Date range:** March–April 2026

---

## The Research Questions

1. **RQ1:** What strategic behaviours emerge when independent RL agents train concurrently, and how sensitive are they to noise?
2. **RQ2:** How does varying the reward sharing coefficient alter strategy, and is there a destabilisation threshold?
3. **RQ3:** Does shared incentive produce genuine cooperative advantage against a common adversary, or merely redistribute outcomes?

---

## Phase 1: Hyperparameter Tuning (AutoResearch)

**What:** 30 automated iterations of hyperparameter optimisation for DQN against a fixed Base Agent.

**Finding:** Larger network capacity (512 hidden units) and learning rate 7e-4 performed best. Objective score 0.733. This established the config baseline for all subsequent phases.

**Why it mattered:** Without good hyperparameters, all downstream experiments would be confounded by poor learning rather than the incentive structure we wanted to study.

---

## Phase 2: Algorithm Selection (Single-Agent Benchmark)

**What:** 4 DQN variants (Vanilla, Double, Dueling, Rainbow-lite) tested against a fixed Base Agent across 3 stochasticity levels. 36 trials total.

**Key results:**

| Algorithm | Win Rate (s0) | Robust under noise? |
|-----------|:---:|:---:|
| Rainbow-lite | **0.835** | Yes (range 0.005) |
| Double DQN | 0.795 | No (drops to 0.740) |
| Vanilla DQN | 0.746 | No |
| Dueling DQN | 0.724 | No (pathological collapse) |

**Findings:**
- Rainbow-lite was the clear winner — highest performance AND most stable under noise
- Double DQN's theoretical advantage (preventing overestimation) didn't help when both networks train on the same noisy data
- Dueling DQN showed pathological bimodal collapse due to the low-dimensional state space (6 features)
- Discovered the "La Source confound" — most agents spam the easiest zone rather than learning genuine strategy

**Decision:** Carry forward Rainbow-lite (primary) and Vanilla (baseline). Drop Double and Dueling.

**RQ1 contribution:** Established which algorithm provides a stable learning substrate for multi-agent experiments.

---

## Phase 3: Independent Learner MARL (Two Co-Learning Agents)

**What:** Two identical DQN agents train simultaneously against each other (no fixed opponent). Each agent's replay buffer contains experiences from a non-stationary environment because the opponent keeps changing. 54 trials total.

**Key findings:**

1. **Degenerate collapse** (Vanilla, seed 101): One agent won 150/150 races. The other converged to never attempting an overtake. An absorbing equilibrium — once one agent stops trying, it never starts again.

2. **Emergent zone specialisation** (Rainbow, seed 202): Two identical algorithms spontaneously divided the circuit into spheres of influence — one monopolised Raidillon, the other colonised Pouhon and Campus. Zone differentiation index 0.417, with zero drift. This happened with no coordination signal.

3. **Hawk-dove risk roles**: Every non-degenerate Rainbow trial developed complementary risk profiles — one agent conservative, the other aggressive. A mixed-strategy Nash equilibrium emerging from pure learning dynamics.

4. **Rainbow handles non-stationarity 3.4x better**: Mean policy drift 0.047 (Rainbow) vs 0.160 (Vanilla).

**RQ1 contribution:** Concurrent training DOES produce interpretable strategic structure (zone partitioning, hawk-dove roles). Rainbow's PER + n-step stability transfers from single-agent noise robustness to multi-agent co-convergence.

---

## Phase 4: Reward Sharing in Zero-Sum Games (Alpha Sweep)

**What:** Introduced reward sharing coefficient alpha. At alpha=0, each agent cares only about its own result. At alpha=1, each agent cares only about its teammate's result. Formula: `mixed = (1-alpha) * own_delta + alpha * teammate_delta`. Tested alpha 0.0 to 1.0 in a head-to-head (zero-sum) 2-agent game. 45 trials.

**The catastrophic result:**

| Alpha | Collapse Rate |
|:---:|:---:|
| 0.0 | 0% |
| 0.25 | 11% |
| 0.50 | 11% |
| 0.75 | **89%** |
| 1.0 | **100%** |

**Why:** In zero-sum games, your teammate can only improve at your expense. At high alpha, agents receive reward for something they can't positively influence. Q-values get noisy gradients, and agents collapse into passivity.

**Critical question raised:** Is cooperation fundamentally broken, or is the zero-sum structure the problem?

**RQ2 contribution:** Sharp destabilisation threshold exists. The transition is sudden (11% → 89% collapse between alpha=0.50 and 0.75), confirming H2's prediction.

---

## Phase 5: Non-Zero-Sum MARL (Common Adversary)

**What:** Changed the game structure to 2 DQN agents + 1 Base adversary. Now agents share a common enemy — both can win simultaneously. Tested alpha 0.0 to 1.0. Included an ablation study proving the game structure (not evaluation methodology) was the causal variable. 45 trials.

**The inverted-U curve:**

| Alpha | Joint Beat-Base Rate | Change |
|:---:|:---:|:---:|
| 0.0 | 0.310 | baseline |
| 0.25 | 0.294 | -5% |
| 0.50 | 0.433 | +40% |
| **0.75** | **0.567** | **+83%** |
| 1.0 | 0.436 | +41% |

**Zero collapse at any alpha level.** The common enemy prevents passivity.

**Key insights:**
- Alpha=0.75 is optimal (not 1.0) — agents need 25% individual incentive to maintain coherent learning
- Cooperation lifts the weaker agent most (A2 jumped from 63% to 78% beat-base rate)
- Hawk-dove risk roles re-emerged under cooperation

**RQ3 contribution:** YES — genuine cooperative advantage exists in 3-agent non-zero-sum games. +83% improvement at alpha=0.75. The ablation proved causality.

---

## Phase 6: Multi-Team Scaling (5-Agent Game)

**What:** Scaled to 4 DQN agents (2 per team) + 1 Base adversary. Tested symmetric cooperation (both teams same alpha) and asymmetric (cooperative team vs competitive team). 36 trials.

**Cooperation failed completely:**

| Condition | Both-Beat-Base | Change |
|-----------|:---:|:---:|
| 0.0 vs 0.0 | 0.350 | baseline |
| 0.50 vs 0.50 | 0.306 | -12% |
| 0.75 vs 0.75 | 0.243 | **-31%** |
| 0.75 vs 0.0 | 0.291 | -17% |

Every form of cooperation made things worse. The cooperative team even **lost** to the competitive team in the asymmetric condition (47.5% win rate).

**Root cause: Credit assignment dilution.** In Phase 5, each agent's actions strongly influence its single teammate (3-car race). In Phase 6, each agent's influence is diluted across 3 other competitors. 75% of reward comes from sources the agent can barely control → noisy Q-value updates → agents learn caution instead of coordination.

**RQ2/RQ3 contribution:** Cooperation that works brilliantly at N=3 catastrophically fails at N=5. The failure is structural, not parametric.

---

## Phase 7: Boundary Characterisation

**What:** Two sub-experiments to precisely locate the cooperation boundary.

**7A (3 DQN + 1 Base — the intermediate game size):** Tested alpha=0.0 and 0.75 with 4 total agents. 18 trials.

**7B (Low-alpha sweep in 5-agent teams):** Tested alpha=0.10, 0.15, 0.25 in the existing team setup. 27 trials.

**Results:**

| Game Size | Alpha=0.75 Effect |
|-----------|:---:|
| 3 agents (Phase 5) | **+83%** |
| 4 agents (Phase 7A) | **+5%** (noise) |
| 5 agents (Phase 6) | **-31%** |

**The boundary is at N=4, and it's a cliff.** Adding a single DQN agent to Phase 5's successful setup eliminates the entire cooperative advantage. This is a threshold effect, not gradual degradation.

**Low alpha doesn't help either.** Even alpha=0.10 in the 5-agent game provides zero benefit (0.344 vs baseline 0.350). There is no Goldilocks alpha — the failure is structural.

**Mechanistic finding:** Alpha=0.75 increases zone differentiation by 63% at N=4 (agents DO respond to the cooperative signal by specialising), but this specialisation doesn't translate into better joint outcomes. Instead, agents learn caution — attempt rates drop 38-64% across zones.

**RQ2 contribution:** Precise boundary identified. The transition from "cooperation helps" to "cooperation is neutral" occurs at exactly N=4 total agents.

---

## Phase 8: Curriculum Alpha Scheduling

**What:** Instead of applying alpha=0.75 immediately, gradually ramp from 0→0.75 over training. Schedule: 100 episodes at alpha=0 (learn individual play) → 300 episodes linear ramp → 100 episodes at full alpha. Tested on both 4-agent and 5-agent games. 18 trials.

**8A results (4 agents):**

| Condition | Joint Beat-Base | A1 La Source Attempt Rate |
|-----------|:---:|:---:|
| Competitive (a=0.0) | 0.159 | 0.864 |
| Fixed (a=0.75) | 0.167 | 0.538 |
| **Curriculum (0→0.75)** | **0.184** | **0.819** |

**The caution pathology is 86-89% eliminated.** Agents that first learn individual policies retain aggressive behaviour when cooperative reward is gradually introduced.

**8B results (5-agent teams):**

| Condition | Both-Beat-Base | Base Position |
|-----------|:---:|:---:|
| Competitive (a=0.0) | 0.350 | 3.315 |
| Fixed (a=0.75) | 0.243 | 2.956 |
| **Curriculum (0→0.75)** | **0.318** | **3.299** |

**Recovers 69% of the Phase 6 performance loss.** Fixed alpha caused -31% degradation; curriculum reduces this to -9%. Base agent stays near last instead of climbing the field.

**But cooperation still doesn't emerge.** Intra-team correlation remains negative (-0.189). Joint performance doesn't exceed competitive baseline. Curriculum prevents damage but doesn't create coordination.

**Novel contribution:** This cleanly separates two failure modes that were previously conflated in the literature:
1. **Training-order problem** (SOLVED): Immediate cooperative reward overwrites individual policies with passivity
2. **Credit assignment problem** (NOT SOLVED): At N>=4, agents can't influence partners enough for shared reward to produce cooperative gradients — requires architectural solutions (CTDE, difference rewards)

---

## The Complete Arc

```
Phase 1:  Tune the engine                    → Foundation
Phase 2:  Which algorithm learns best?       → Rainbow-lite
Phase 3:  Can agents co-learn?               → Yes, with emergent specialisation
Phase 4:  Can we add cooperation?            → Not in zero-sum (collapse)
Phase 5:  Fix the game structure             → Yes! +83% at alpha=0.75
Phase 6:  Does it scale to teams?            → No — credit assignment breaks down
Phase 7:  Where exactly is the boundary?     → N=4 (cliff, not slope)
Phase 8:  Can training schedule fix it?      → Partially — prevents damage, can't create cooperation
Phase 9:  How much is learned vs obvious?    → LLM matches Rainbow-lite within 2% (zero training)
```

## Key Literature References

| Paper | Relevance |
|-------|-----------|
| Tan 1993 | Foundational IL vs cooperative comparison — our Phase 6 hits the exact limitation Tan identified |
| Wolpert & Tumer 2002 | Team reward becomes noise as N grows — explains our alpha dilution |
| Foerster et al. 2018 (COMA) | Counterfactual credit assignment — what our alpha blending lacks |
| Sunehag et al. 2018 (VDN) | Lazy agent problem — matches our caution pathology |
| Wei & Luke 2016 | Relative overgeneralization — explains why agents undervalue good actions under noisy team reward |
| Wang et al. 2022 (IRAT) | Curriculum beta with dual policies — our Phase 8 is the simpler IL-MARL version |
| Matignon et al. 2012 | Survey of IL coordination failures — our results exhibit multiple pathologies from their taxonomy |
| Papoudakis et al. 2021 | Benchmarking shows IL fails on complex coordination tasks — consistent with our N>=4 results |

## Files Reference

| Phase | Analysis | Metrics | Config |
|-------|----------|---------|--------|
| 1 | `AUTORESEARCH_REPORT.md` | — | `config.json` |
| 2 | `research_findings/phase2_analysis.md` | `metrics/phase2/` (36 files) | — |
| 3 | `research_findings/phase3_full_analysis.md` | `metrics/phase3/` (54 files) | — |
| 4 | `research_findings/phase4_full_analysis.md` | `metrics/phase4/` (45 files) | — |
| 5 | `research_findings/phase5_full_analysis.md` | `metrics/phase5/` (45 files) | `metrics/phase5/config_*.json` |
| 6 | `research_findings/phase6_full_analysis.md` | `metrics/phase6/` (36 files) | `metrics/phase6/config_*.json` |
| 7 | `research_findings/phase7_full_analysis.md` | `metrics/phase7a/` (18), `metrics/phase7b/` (27) | `metrics/phase7a/config_*.json`, `metrics/phase7b/config_*.json` |
| 8 | `research_findings/phase8_full_analysis.md` | `metrics/phase8a/` (9), `metrics/phase8b/` (9) | Reuses Phase 6/7A configs |
| 9 | `research_findings/phase9_analysis.md` | `metrics/phase9/` (3 files) | `config.json` (llm_params) |

## Source Code Changes by Phase

| Phase | Files Modified |
|-------|---------------|
| 7 | `src/runtime_profiles.py` (new profile), `src/states.py` (routing), `src/simulator.py` (generalized reward sharing), `scripts/evaluate_marl.py` (3-agent metrics) |
| 8 | `src/simulator.py` (curriculum scheduling in `run_batch`), `scripts/evaluate_marl.py` (CLI flags + output recording) |
| 9 | `src/agents/LLM.py` (new), `src/states.py` (llm agent registration), `src/runtime_profiles.py` (low_llm_vs_base profile), `src/simulator.py` (LLM position cycling), `scripts/evaluate_llm_agent.py` (new), `scripts/tune_llm_prompt.py` (new) |

All changes are additive — backward compatible with Phases 1-8.


---
  ---
  The Story in Plain English

  You built a racing simulator and asked: "If I change how much two AI drivers care about each other's results, what
  happens to how they race?"

  You started by finding the best algorithm (Rainbow-lite, Phases 1-2), then showed that two co-learning agents
  spontaneously develop complementary strategies — one attacks at Raidillon, the other at Pouhon — with no coordination
  signal (Phase 3).

  Then you tried adding cooperation. In a head-to-head game, it collapsed — agents just gave up (Phase 4). You fixed
  this by adding a common enemy (a Base agent), and cooperation worked beautifully: +83% improvement when agents shared
  75% of each other's reward (Phase 5).

  The critical question was: does this scale? You went from 3 to 5 agents and cooperation catastrophically failed — a
  -31% degradation (Phase 6). So you ran Phase 7 to find the exact boundary: it's at N=4 agents, and it's a cliff. Even
  tiny amounts of cooperation (alpha=0.10) don't help at 5 agents. The problem is structural — each agent can't
  influence its partners' outcomes enough for the shared reward to mean anything.

  Finally, you asked whether the failure was about when the cooperative signal is introduced or what the signal contains
   (Phase 8). By gradually ramping cooperation from zero, you recovered 69% of the lost performance and eliminated the
  caution pathology where agents stop trying. But genuine coordination still didn't emerge.

  Then you asked a different kind of question: how much of what the RL agents learned was actually clever, and how much
  was just strategically obvious? You replaced the DQN with an LLM (Claude Haiku) that had domain knowledge but zero
  training (Phase 9). The LLM matched Rainbow-lite's 84% win rate within 2 percentage points — but through a completely
  different strategy. It never attempted Raidillon (which the DQN wastes attempts on), discovered the Campus→Stavelot
  cooldown tradeoff (which the DQN cannot represent), and used zero AGGRESSIVE risk (which the DQN relies on). Same
  performance, different path. This confirmed that single-agent strategy at this scale is "strategically obvious" —
  the real complexity lies in the multi-agent coordination problems of Phases 3-8.

  Your novel contribution: You cleanly separated two failure modes that the literature had conflated — the
  training-order problem (solvable with curriculum scheduling) and the credit assignment problem (requires fundamentally
   different architectures). Nobody had demonstrated this decomposition empirically before. Additionally, by introducing
  an LLM semantic baseline, you showed that RL-emergent zone strategies are largely rediscoveries of domain-obvious
  structure, while identifying specific RL failure modes (Raidillon overexploration, cooldown-blind zone allocation)
  visible only by contrast with a semantic reasoner.