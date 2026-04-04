# Phase 8 Full Analysis — Curriculum Alpha Scheduling

**Date:** 2026-04-04
**Algorithm:** rainbow-lite (IL-MARL)
**Sub-experiment A profile:** `low_marl_3dqn_vs_base` (3 DQN agents + 1 Base adversary)
**Sub-experiment B profile:** `low_marl_teams` (4 DQN agents in 2 teams + 1 Base adversary)
**Budget per trial:** 500 training episodes, 150 evaluation races (balanced starting positions)
**Curriculum schedule:** 100 warmup episodes (alpha=0.0) → 300 ramp episodes (linear 0→target) → 100 episodes at full target alpha
**Target alpha:** 0.75
**Seeds:** 101, 202, 303
**Stochasticity levels:** s0, s1, s2
**Total trials:** 18 (9 sub-experiment A + 9 sub-experiment B)
**Primary metric:** `joint_dqn_beat_base_rate` (8A) / `both_beat_base_rate` (8B)

---

## 1. Experimental rationale

Phase 7 established that fixed alpha=0.75 fails at N>=4 agents. The mechanistic analysis revealed a specific pathology: agents trained with immediate cooperative reward learn **caution** (attempt rates drop 38-64%) rather than **coordination**. The cooperative signal suppresses individual play before it can develop.

Phase 8 tests whether **graduated introduction** of cooperative incentive avoids this pathology. The hypothesis, informed by Wang et al. (2022) (IRAT), is that agents first trained on individual reward learn strong overtaking policies, and that gradually introducing cooperative reward allows them to layer coordination on top of existing competence rather than replacing competence with caution.

The curriculum schedule:
```
Episodes   1-100:  alpha = 0.0   (learn individual play)
Episodes 101-400:  alpha = 0.75 * (episode - 100) / 300  (linear ramp)
Episodes 401-500:  alpha = 0.75  (full cooperative incentive)
```

Evaluation always runs at the final alpha=0.75 — the curriculum only affects training.

---

## 2. Simulator verification

| Check | Status | Evidence |
|-------|--------|----------|
| `alpha_curriculum` recorded in all 18 output files | PASS | All show `{"enabled": true, "warmup_episodes": 100, "ramp_episodes": 300}` |
| Target alpha correctly recorded | PASS | 8A: `reward_sharing_alpha: 0.75`; 8B: `team_alphas: {team_a: 0.75, team_b: 0.75}` |
| All trials: balanced positions enabled | PASS | `balanced_positions: true` in all 18 files |
| All trials: 500 training, 150 evaluation | PASS | All files |
| All trials: no DNF | PASS | `dnf_rate: 0.0` for all agents in all 18 trials |
| Seed x stochasticity coverage complete | PASS | 3 x 3 = 9 per sub-experiment |

---

## 3. Sub-experiment 8A: The headline finding — attempt rates recover

### 3.1 Cross-condition comparison (3 DQN + 1 Base)

| Condition | Source | Joint Beat-Base | Base Pos | Zone Diff | A1 La Source Attempt | A1 Les Combes Attempt |
|-----------|--------|:---------------:|:--------:|:---------:|:--------------------:|:---------------------:|
| alpha=0.0 (competitive) | Phase 7A | 0.159 | 2.687 | 0.157 | 0.864 | 0.596 |
| alpha=0.75 (fixed) | Phase 7A | 0.167 | 2.726 | 0.256 | 0.538 | 0.314 |
| **alpha=0→0.75 (curriculum)** | **Phase 8A** | **0.184** | **2.710** | **0.203** | **0.819** | **0.565** |

**Curriculum eliminates the caution pathology.** Agent 1's La Source attempt rate recovers from 0.538 (fixed cooperative) to 0.819 (curriculum), nearly matching the competitive baseline of 0.864. Les Combes follows the same pattern: 0.314 → 0.565, approaching the baseline of 0.596.

Joint beat-base improves modestly from 0.167 (fixed) to 0.184 (curriculum) — a +10% improvement, though still within seed variance of the competitive baseline (0.159).

### 3.2 Raw trial data — 8A curriculum

| Seed | Stoch | A1 WR | A1 beat-base | A2 beat-base | A3 beat-base | Joint | Base pos | Zone diff | Risk diff | Drift |
|------|-------|-------|:------------:|:------------:|:------------:|:-----:|:--------:|:---------:|:---------:|:-----:|
| 101 | s0 | 0.440 | 0.553 | 0.607 | 0.593 | 0.193 | 2.753 | 0.123 | 0.054 | +0.20 |
| 202 | s0 | 0.613 | 0.627 | 0.360 | 0.533 | 0.160 | 2.520 | 0.381 | 0.128 | +0.06 |
| 303 | s0 | 0.527 | 0.600 | 0.613 | 0.620 | 0.240 | 2.833 | 0.068 | 0.063 | -0.08 |
| 101 | s1 | 0.493 | 0.533 | 0.527 | 0.540 | 0.167 | 2.600 | 0.194 | 0.083 | +0.06 |
| 202 | s1 | 0.447 | 0.560 | 0.607 | 0.580 | 0.193 | 2.747 | 0.097 | 0.134 | +0.10 |
| 303 | s1 | 0.507 | 0.553 | 0.567 | 0.613 | 0.140 | 2.733 | 0.195 | 0.046 | -0.04 |
| 101 | s2 | 0.440 | 0.433 | 0.587 | 0.633 | 0.153 | 2.653 | 0.406 | 0.139 | +0.18 |
| 202 | s2 | 0.513 | 0.653 | 0.640 | 0.573 | 0.253 | 2.867 | 0.235 | 0.180 | +0.00 |
| 303 | s2 | 0.567 | 0.580 | 0.573 | 0.500 | 0.153 | 2.653 | 0.105 | 0.010 | -0.26 |

**Summary:** 0/9 collapse. Mean joint beat-base = 0.184. Mean zone differentiation = 0.203 (between competitive 0.157 and fixed 0.256). Zero DNF.

### 3.3 Aggregates by stochasticity

| Stoch | Competitive (7A a=0.0) | Fixed (7A a=0.75) | Curriculum (8A) |
|-------|:----------------------:|:------------------:|:---------------:|
| s0 | 0.142 | 0.120 | **0.198** |
| s1 | 0.140 | 0.207 | 0.167 |
| s2 | 0.196 | 0.173 | **0.187** |
| **Mean** | **0.159** | **0.167** | **0.184** |

Curriculum outperforms or matches fixed alpha at every stochasticity level except s1. The s0 result (0.198 vs 0.120) is the strongest: curriculum is +65% above fixed in the deterministic setting where the caution pathology was most severe.

---

## 4. Sub-experiment 8B: Curriculum partially recovers team performance

### 4.1 Cross-condition comparison (5-agent teams)

| Condition | Source | Avg Both-Beat-Base | Base Pos | Mean Intra-A | Mean Intra-B | Zone Diff |
|-----------|--------|:------------------:|:--------:|:------------:|:------------:|:---------:|
| alpha=0.0/0.0 (competitive) | Phase 6 | 0.350 | 3.315 | -0.133 | -0.116 | 0.122 |
| alpha=0.75/0.75 (fixed) | Phase 6 | 0.243 | 2.956 | -0.122 | -0.109 | 0.195 |
| **alpha=0→0.75/0→0.75 (curriculum)** | **Phase 8B** | **0.318** | **3.299** | **-0.183** | **-0.194** | **0.086** |

**Curriculum recovers 69% of the Phase 6 performance loss.** Fixed alpha=0.75 caused a -31% degradation (0.350 → 0.243). Curriculum reduces this to only -9% (0.350 → 0.318). The improvement is 0.075 in absolute terms (0.243 → 0.318).

The Base agent position tells the same story: with fixed alpha=0.75, Base climbed to position 2.956 (near third). With curriculum, Base stays at 3.299 — near last, close to the competitive baseline of 3.315. **Curriculum prevents the Base agent from exploiting cooperative DQN teams.**

### 4.2 Raw trial data — 8B curriculum

| Seed | Stoch | TA win rate | TA beat-base | TB beat-base | Base pos | ZD | RD | Intra-A | Intra-B |
|------|-------|:-----------:|:------------:|:------------:|:--------:|:--:|:--:|:-------:|:-------:|
| 101 | s0 | 0.520 | 0.340 | 0.333 | 3.327 | 0.051 | 0.052 | -0.105 | -0.178 |
| 202 | s0 | 0.460 | 0.287 | 0.327 | 3.227 | 0.125 | 0.037 | -0.091 | -0.156 |
| 303 | s0 | 0.473 | 0.227 | 0.373 | 3.233 | 0.076 | 0.027 | -0.236 | -0.178 |
| 101 | s1 | 0.427 | 0.267 | 0.300 | 3.133 | 0.046 | 0.045 | -0.129 | -0.201 |
| 202 | s1 | 0.560 | 0.367 | 0.373 | 3.447 | 0.052 | 0.123 | -0.071 | -0.049 |
| 303 | s1 | 0.400 | 0.280 | 0.413 | 3.353 | 0.144 | 0.043 | -0.295 | -0.222 |
| 101 | s2 | 0.500 | 0.253 | 0.367 | 3.360 | 0.179 | 0.107 | -0.273 | -0.337 |
| 202 | s2 | 0.493 | 0.187 | 0.293 | 3.087 | 0.058 | 0.154 | -0.284 | -0.176 |
| 303 | s2 | 0.567 | 0.333 | 0.400 | 3.527 | 0.039 | 0.158 | -0.165 | -0.246 |

**Summary:** 0/9 collapse. Mean both-beat-base = 0.318. Zero DNF. Intra-team cooperation universally negative.

### 4.3 Aggregates by stochasticity

| Stoch | Competitive (Phase 6) | Fixed a=0.75 (Phase 6) | Curriculum (8B) |
|-------|:---------------------:|:----------------------:|:---------------:|
| s0 | 0.350 | 0.243 | **0.314** |
| s1 | 0.350 | 0.243 | **0.333** |
| s2 | 0.350 | 0.243 | 0.306 |
| **Mean** | **0.350** | **0.243** | **0.318** |

Curriculum consistently outperforms fixed cooperative alpha across all stochasticity levels, recovering to near-baseline levels.

---

## 5. Mechanistic analysis: What curriculum changes

### 5.1 The caution pathology is eliminated

The defining pathology of fixed alpha=0.75 at N>=4 was that agents learned to hold position rather than attempt overtakes. Curriculum training reverses this:

**Agent 1 mean attempt rates across all Phase 8A trials vs baselines:**

| Zone | Competitive (a=0.0) | Fixed (a=0.75) | Curriculum (0→0.75) | Recovery |
|------|:--------------------:|:--------------:|:-------------------:|:--------:|
| La Source (d=0.2) | 0.864 | 0.538 | **0.819** | 86% |
| Les Combes (d=0.7) | 0.596 | 0.314 | **0.565** | 89% |

"Recovery" measures how far curriculum returns toward the competitive baseline: `(curriculum - fixed) / (competitive - fixed)`. At 86-89%, curriculum almost fully eliminates the attempt-rate suppression.

### 5.2 But coordination does not emerge

Despite maintaining aggressive individual play, curriculum agents do not achieve meaningful coordination:

| Metric | Competitive | Fixed a=0.75 | Curriculum |
|--------|:-----------:|:------------:|:----------:|
| 8A joint beat-base | 0.159 | 0.167 | 0.184 |
| 8B both-beat-base | 0.350 | 0.243 | 0.318 |
| 8B intra-team corr | -0.125 | -0.116 | -0.189 |

In 8A, curriculum slightly improves joint beat-base but remains statistically indistinguishable from competitive baseline. In 8B, curriculum recovers from the damage of fixed cooperation but does not exceed competitive performance. Intra-team cooperation is actually **more negative** under curriculum (-0.189) than under either baseline, suggesting that agents who maintain individual aggression are even more antagonistic within teams.

### 5.3 The asymmetry: damage prevention vs. benefit creation

Curriculum alpha achieves one thing and fails at another:

**What it achieves:** Prevents the cooperative reward signal from overwriting individually-learned aggressive policies. Agents trained first on individual reward develop strong overtaking behaviour. The gradual introduction of cooperative reward does not erase this.

**What it cannot achieve:** Creating genuine coordination between agents. The cooperative signal at N>=4 remains too noisy to produce meaningful cooperative gradients, regardless of when it is introduced. Agents who maintain individual aggression simply compete with each other more effectively — they don't cooperate.

**The curriculum resolves the training-order problem but not the credit assignment problem.**

---

## 6. Cross-phase summary: The complete alpha investigation

| Phase | Game | Alpha | Method | Joint/Both Beat-Base | vs Competitive Baseline |
|-------|------|-------|--------|:--------------------:|:-----------------------:|
| 5 | 3 agents (2 DQN + Base) | 0.0 | Fixed | 0.310 | baseline |
| 5 | 3 agents (2 DQN + Base) | 0.75 | Fixed | **0.567** | **+83%** |
| 7A | 4 agents (3 DQN + Base) | 0.0 | Fixed | 0.159 | baseline |
| 7A | 4 agents (3 DQN + Base) | 0.75 | Fixed | 0.167 | +5% (noise) |
| **8A** | **4 agents (3 DQN + Base)** | **0→0.75** | **Curriculum** | **0.184** | **+16%** |
| 6 | 5 agents (4 DQN + Base) | 0.0 | Fixed | 0.350 | baseline |
| 6 | 5 agents (4 DQN + Base) | 0.75 | Fixed | 0.243 | -31% |
| **8B** | **5 agents (4 DQN + Base)** | **0→0.75** | **Curriculum** | **0.318** | **-9%** |

---

## 7. Hypothesis evaluation

**H8: Curriculum alpha scheduling recovers cooperative advantage by preserving individual competence during early training.**

PARTIALLY CONFIRMED. Curriculum scheduling eliminates the caution pathology (attempt rates recover 86-89%) and prevents most of the performance degradation caused by fixed cooperative reward at scale (69% recovery in the 5-agent game). However, it does not produce genuine cooperative advantage beyond the competitive baseline at any game size tested. The benefit is **damage prevention**, not cooperation.

This separates the problem into two components:

1. **Training-order problem** (SOLVED by curriculum): Immediate cooperative reward overwrites individual policies with passive behaviour. Gradual introduction avoids this.

2. **Credit assignment problem** (NOT solved by curriculum): At N>=4, no agent can meaningfully influence its partners' outcomes. The cooperative signal remains noise regardless of when it is introduced. This requires architectural solutions (CTDE, difference rewards) rather than scheduling solutions.

---

## 8. Implications for research questions

**RQ2 (Threshold and destabilisation):** Phase 8 adds a new dimension to the answer. The threshold at N=4 identified in Phase 7 has two components: a training-order component (solvable) and a structural credit assignment component (not solvable with IL-MARL). Curriculum scheduling addresses the first but not the second.

**RQ3 (Genuine cooperative advantage):** Curriculum alpha does not produce genuine cooperative advantage at N>=4. The only game size where reward sharing produces positive cooperative surplus remains N=3 (Phase 5). However, curriculum prevents the cooperative incentive from becoming a net liability at larger scales, which is itself a practically useful finding.

---

## 9. Summary

1. **Curriculum alpha eliminates the caution pathology.** Agents that first learn individual policies retain aggressive overtaking behaviour when cooperative reward is gradually introduced. Attempt rates recover 86-89% toward competitive baselines. This confirms the training-order hypothesis: immediate cooperative reward suppresses individual competence.

2. **Curriculum recovers 69% of the Phase 6 performance loss.** In the 5-agent team game, fixed alpha=0.75 caused -31% degradation; curriculum reduces this to -9%. The Base agent stays near last position rather than climbing the field. This is a practically significant improvement.

3. **Curriculum does not create cooperation.** Intra-team correlation remains negative. Joint performance does not exceed competitive baselines. Agents who maintain individual aggression compete with each other more effectively, but they do not coordinate.

4. **The failure decomposes into two problems.** (a) Training-order: cooperative reward overwrites individual policies — solved by curriculum. (b) Credit assignment: at N>=4, agents cannot influence partners enough for shared reward to produce cooperative gradients — unsolved, requires architectural intervention (CTDE, difference rewards).

5. **Novel contribution.** No prior work has tested curriculum alpha scheduling with IL-MARL in a competitive simulator. The finding that graduated incentive introduction prevents caution without enabling cooperation is a clean empirical separation of the training-order and credit-assignment failure modes in cooperative MARL. This extends Wang et al. (2022), who used curriculum beta with dual policies and CTDE, by showing that the simpler IL-MARL version achieves damage prevention but not genuine cooperation.
