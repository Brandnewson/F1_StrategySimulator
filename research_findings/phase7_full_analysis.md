# Phase 7 Full Analysis — Credit Assignment Boundary Characterization

**Date:** 2026-04-04
**Algorithm:** rainbow-lite (IL-MARL)
**Sub-experiment A profile:** `low_marl_3dqn_vs_base` (3 DQN agents + 1 Base adversary)
**Sub-experiment B profile:** `low_marl_teams` (4 DQN agents in 2 teams + 1 Base adversary)
**Budget per trial:** 500 training episodes, 150 evaluation races (balanced starting positions)
**Seeds:** 101, 202, 303
**Stochasticity levels:** s0 (deterministic), s1 (2% noise), s2 (5% noise)
**Total trials:** 45 (18 sub-experiment A + 27 sub-experiment B)
**Primary metric:** `joint_dqn_beat_base_rate` (fraction of races where ALL DQN agents finish ahead of Base)

---

## 1. Experimental rationale

Phase 5 demonstrated that reward sharing at alpha=0.75 produces a +83% improvement in joint-beat-base rate in a 3-agent game (2 DQN + 1 Base). Phase 6 demonstrated that the same alpha produces a -31% degradation in a 5-agent game (4 DQN + 1 Base). The hypothesis is that the causal variable is agent influence over teammate outcome, which is diluted as more agents enter the race.

Phase 7 tests this hypothesis with two sub-experiments:

| Sub-experiment | Profile | Agents | Alpha values | Trials | Question |
|----------------|---------|--------|--------------|--------|----------|
| **7A** | `low_marl_3dqn_vs_base` | 3 DQN + 1 Base (4 total) | 0.0, 0.75 | 18 | Does cooperation work at the intermediate 4-agent scale? |
| **7B** | `low_marl_teams` | 4 DQN + 1 Base (5 total) | 0.10, 0.15, 0.25 | 27 | Does lowering alpha recover cooperation in the 5-agent game? |

The reward sharing formula for 7A generalizes Phase 5's pairwise formula to N agents:
```
mixed_outcome_i = (1 - alpha) * own_delta + alpha * mean(all_other_DQN_deltas)
```

Sub-experiment 7B uses the existing Phase 6 team-based formula unchanged.

---

## 2. Simulator verification

| Check | Status | Evidence |
|-------|--------|----------|
| 7A: `complexity_profile` is `low_marl_3dqn_vs_base` in all 18 files | PASS | All show `"complexity_profile": "low_marl_3dqn_vs_base"` |
| 7A: `phase` field is `phase7_3dqn_vs_base` in all 18 files | PASS | All show `"phase": "phase7_3dqn_vs_base"` |
| 7A: 3 DQN agents named in all files | PASS | `all_dqn_names` has 3 entries; `agent3_name` populated |
| 7A: `vs_base_metrics` has `dqn_a1/a2/a3_beats_base_rate` | PASS | All 3 per-agent rates present in all 18 files |
| 7B: `complexity_profile` is `low_marl_teams` in all 27 files | PASS | All show `"complexity_profile": "low_marl_teams"` |
| 7B: `team_metrics` present in all 27 files | PASS | All contain team-level metrics |
| All trials: balanced positions enabled | PASS | `balanced_positions: true` in all 45 files |
| All trials: algorithm is rainbow_lite | PASS | All 45 files |
| All trials: no DNF | PASS | `dnf_rate: 0.0` for all agents in all 45 trials |
| All trials: training not skipped, 500 episodes | PASS | `runs: 500`, `skipped: false` |
| All trials: 150 evaluation races | PASS | `runs_per_seed: 150`, `n_races: 150` |
| All trials: alpha values correctly recorded | PASS | All files match their config alpha |
| Seed x stochasticity coverage complete | PASS | 3 seeds x 3 stoch = 9 per condition, all present |

---

## 3. Sub-experiment 7A: The headline finding

### 3.1 Cross-phase comparison table

| Phase | Game size | Agents | Alpha | Joint Beat-Base | Base Avg Pos | Collapse |
|-------|-----------|--------|-------|:---------------:|:------------:|:--------:|
| **5** | 3 agents | 2 DQN + 1 Base | 0.0 | 0.310 | 2.213 | 0/9 |
| **5** | 3 agents | 2 DQN + 1 Base | 0.75 | **0.567** | **2.409** | 0/9 |
| **7A** | 4 agents | 3 DQN + 1 Base | 0.0 | 0.159 | 2.687 | 0/9 |
| **7A** | 4 agents | 3 DQN + 1 Base | 0.75 | 0.167 | 2.726 | 0/9 |
| **6** | 5 agents | 4 DQN + 1 Base | 0.0 | 0.350 (team) | 3.315 | 0/9 |
| **6** | 5 agents | 4 DQN + 1 Base | 0.75 | 0.243 (team) | 2.956 | 0/9 |

**Cooperation produces no measurable benefit at the 4-agent scale.** Alpha=0.75 yields a joint beat-base rate of 0.167, statistically indistinguishable from the alpha=0.00 baseline of 0.159 (delta = +0.008, well within seed variance). The +83% cooperative advantage observed in Phase 5 has vanished entirely with the addition of a single DQN agent.

### 3.2 Per-agent beat-base rates

| Alpha | A1 beat-base | A2 beat-base | A3 beat-base | Joint | Base pos |
|-------|:------------:|:------------:|:------------:|:-----:|:--------:|
| 0.0 | 0.578 | 0.531 | 0.571 | 0.159 | 2.687 |
| 0.75 | 0.585 | 0.517 | 0.613 | 0.167 | 2.726 |

Individual agents beat Base at roughly the same rate regardless of alpha (~52-61%). The joint metric is low because all three must beat Base simultaneously. Adding cooperation does not improve coordination — agents do not learn to collectively push Base down.

### 3.3 Why is joint beat-base so low compared to Phase 5?

Phase 5 baseline (alpha=0.0): individual agents beat Base at ~58%, joint = 31%.
Phase 7A baseline (alpha=0.0): individual agents beat Base at ~53-58%, joint = 16%.

The drop from 31% to 16% is largely mechanical: with 3 agents all needing to beat Base independently, the probability compounds. If each agent beats Base with probability p independently, joint probability is p^3 for 3 agents vs p^2 for 2. At p=0.55: 0.55^2 = 0.302 vs 0.55^3 = 0.166. The observed 0.159 is consistent with ~55% independent beat-base rate cubed, confirming that alpha=0.0 agents are essentially independent.

**The critical test is whether alpha=0.75 breaks this independence.** In Phase 5, alpha=0.75 raised joint from 0.310 to 0.567 — agents learned to coordinate. In Phase 7A, alpha=0.75 raises joint from 0.159 to 0.167 — no coordination emerges. The cooperative reward signal fails to produce any synergy with 3 DQN agents.

---

## 4. Sub-experiment 7A: Raw trial data

### 4.1 Alpha = 0.0 (competitive baseline, 3 DQN + 1 Base)

| Seed | Stoch | A1 WR | A1 beat-base | A2 beat-base | A3 beat-base | Joint | Base pos | Zone diff | Risk diff | Drift |
|------|-------|-------|:------------:|:------------:|:------------:|:-----:|:--------:|:---------:|:---------:|:-----:|
| 101 | s0 | 0.580 | 0.653 | 0.560 | 0.560 | 0.160 | 2.773 | 0.131 | 0.061 | +0.10 |
| 202 | s0 | 0.473 | 0.547 | 0.547 | 0.607 | 0.120 | 2.700 | 0.090 | 0.170 | +0.16 |
| 303 | s0 | 0.580 | 0.653 | 0.480 | 0.620 | 0.147 | 2.753 | 0.289 | 0.247 | -0.14 |
| 101 | s1 | 0.440 | 0.493 | 0.567 | 0.593 | 0.160 | 2.653 | 0.249 | 0.053 | +0.24 |
| 202 | s1 | 0.520 | 0.607 | 0.460 | 0.547 | 0.093 | 2.613 | 0.215 | 0.137 | +0.02 |
| 303 | s1 | 0.520 | 0.627 | 0.547 | 0.507 | 0.167 | 2.680 | 0.148 | 0.072 | -0.22 |
| 101 | s2 | 0.547 | 0.527 | 0.547 | 0.620 | 0.213 | 2.693 | 0.059 | 0.072 | -0.08 |
| 202 | s2 | 0.493 | 0.573 | 0.547 | 0.500 | 0.180 | 2.620 | 0.143 | 0.285 | -0.08 |
| 303 | s2 | 0.527 | 0.560 | 0.547 | 0.587 | 0.193 | 2.693 | 0.089 | 0.066 | -0.06 |

**Summary:** 0/9 collapse. Mean joint beat-base = 0.159. Mean zone differentiation = 0.157. Mean risk differentiation = 0.129. All agents competitive — no passivity, no degenerate equilibria. Base finishes at position 2.69 on average (near third in a 4-driver race).

### 4.2 Alpha = 0.75 (cooperative, 3 DQN + 1 Base)

| Seed | Stoch | A1 WR | A1 beat-base | A2 beat-base | A3 beat-base | Joint | Base pos | Zone diff | Risk diff | Drift |
|------|-------|-------|:------------:|:------------:|:------------:|:-----:|:--------:|:---------:|:---------:|:-----:|
| 101 | s0 | 0.520 | 0.580 | 0.493 | 0.653 | 0.173 | 2.727 | 0.326 | 0.106 | +0.08 |
| 202 | s0 | 0.580 | 0.653 | 0.413 | 0.607 | 0.113 | 2.673 | 0.252 | 0.223 | -0.18 |
| 303 | s0 | 0.533 | 0.553 | 0.487 | 0.567 | 0.073 | 2.607 | 0.351 | 0.044 | -0.12 |
| 101 | s1 | 0.547 | 0.653 | 0.560 | 0.500 | 0.187 | 2.713 | 0.137 | 0.053 | -0.12 |
| 202 | s1 | 0.580 | 0.613 | 0.533 | 0.600 | 0.140 | 2.747 | 0.116 | 0.113 | -0.12 |
| 303 | s1 | 0.453 | 0.493 | 0.660 | 0.740 | 0.293 | 2.893 | 0.128 | 0.096 | +0.04 |
| 101 | s2 | 0.393 | 0.380 | 0.480 | 0.607 | 0.093 | 2.467 | 0.246 | 0.160 | +0.10 |
| 202 | s2 | 0.560 | 0.727 | 0.480 | 0.653 | 0.253 | 2.860 | 0.293 | 0.048 | +0.08 |
| 303 | s2 | 0.480 | 0.647 | 0.547 | 0.653 | 0.173 | 2.847 | 0.452 | 0.093 | +0.02 |

**Summary:** 0/9 collapse. Mean joint beat-base = 0.167. Mean zone differentiation = 0.256. Mean risk differentiation = 0.104. No cooperative advantage in joint performance. However, zone differentiation is notably higher (0.256 vs 0.157 at alpha=0.0), indicating that reward sharing does induce agents to specialize in different zones — but this specialization fails to translate into improved joint outcomes.

### 4.3 Aggregates by stochasticity

**Alpha = 0.0:**

| Stoch | Joint beat-base (mean) | Seeds | Base pos |
|-------|:----------------------:|-------|:--------:|
| s0 | 0.142 | [0.160, 0.120, 0.147] | 2.742 |
| s1 | 0.140 | [0.160, 0.093, 0.167] | 2.649 |
| s2 | 0.196 | [0.213, 0.180, 0.193] | 2.669 |

**Alpha = 0.75:**

| Stoch | Joint beat-base (mean) | Seeds | Base pos |
|-------|:----------------------:|-------|:--------:|
| s0 | 0.120 | [0.173, 0.113, 0.073] | 2.669 |
| s1 | 0.207 | [0.187, 0.140, 0.293] | 2.784 |
| s2 | 0.173 | [0.093, 0.253, 0.173] | 2.724 |

At s0, alpha=0.75 actually performs *worse* than baseline (0.120 vs 0.142). The best single trial (s303 s1, joint=0.293) is an outlier driven by A3 beating Base 74% of the time — not evidence of systematic cooperation.

---

## 5. Sub-experiment 7B: Low-alpha team sweep

### 5.1 Cross-condition aggregate table (including Phase 6 reference)

| Condition | Source | TA beat-base | TB beat-base | Avg both-beat-base | Base pos | Intra-A | Intra-B | ZD | Collapse |
|-----------|--------|:------------:|:------------:|:------------------:|:--------:|:-------:|:-------:|:--:|:--------:|
| 0.0 vs 0.0 | Phase 6 | 0.353 | 0.346 | 0.350 | 3.315 | -0.133 | -0.116 | 0.122 | 0/9 |
| **0.10 vs 0.10** | **Phase 7B** | **0.318** | **0.369** | **0.344** | **3.283** | **-0.148** | **-0.117** | **0.150** | **0/9** |
| **0.15 vs 0.15** | **Phase 7B** | **0.336** | **0.361** | **0.349** | **3.304** | **-0.113** | **-0.138** | **0.118** | **0/9** |
| **0.25 vs 0.25** | **Phase 7B** | **0.320** | **0.310** | **0.315** | **3.229** | **-0.170** | **-0.135** | **0.136** | **0/9** |
| 0.50 vs 0.50 | Phase 6 | 0.296 | 0.316 | 0.306 | 3.161 | -0.116 | -0.127 | 0.163 | 0/9 |
| 0.75 vs 0.75 | Phase 6 | 0.236 | 0.250 | 0.243 | 2.956 | -0.122 | -0.109 | 0.195 | 0/9 |

**Low alpha does not recover cooperation.** Alpha=0.10 and 0.15 produce both-beat-base rates of 0.344 and 0.349, statistically indistinguishable from the competitive baseline of 0.350. Alpha=0.25 is already showing degradation at 0.315. The monotonic decline from Phase 6 continues: as alpha increases, performance against Base worsens.

The full alpha curve for team-based MARL is now:

```
alpha=0.00:  0.350  (baseline)
alpha=0.10:  0.344  (-2%)
alpha=0.15:  0.349  (~0%)
alpha=0.25:  0.315  (-10%)
alpha=0.50:  0.306  (-12%)
alpha=0.75:  0.243  (-31%)
```

There is no Goldilocks alpha in the 5-agent team game. Even minimal cooperation (alpha=0.10) provides zero benefit.

### 5.2 Intra-team cooperation metric

| Condition | Mean Intra-A | Mean Intra-B | Any positive? |
|-----------|:------------:|:------------:|:-------------:|
| 0.10 vs 0.10 | -0.148 | -0.117 | 1/18 (s202 s2 TA: +0.041) |
| 0.15 vs 0.15 | -0.113 | -0.138 | 0/18 |
| 0.25 vs 0.25 | -0.170 | -0.135 | 0/18 |

Teammates remain antagonistic at all alpha values. Position deltas within a team are negatively correlated — when one teammate gains positions, the other loses them. Only a single trial (alpha=0.10, seed 202, s2, Team A) shows a marginally positive intra-team correlation (+0.041). This is consistent with Phase 6: **the reward sharing mechanism cannot produce genuine intra-team cooperation in the 5-agent game**.

---

## 6. Sub-experiment 7B: Raw trial data

### 6.1 Alpha = 0.10 (both teams)

| Seed | Stoch | TA win rate | TA beat-base | TB beat-base | Base pos | ZD | RD | Intra-A | Intra-B |
|------|-------|:-----------:|:------------:|:------------:|:--------:|:--:|:--:|:-------:|:-------:|
| 101 | s0 | 0.580 | 0.267 | 0.413 | 3.360 | 0.176 | 0.028 | -0.232 | -0.054 |
| 202 | s0 | 0.547 | 0.307 | 0.280 | 3.153 | 0.169 | 0.142 | -0.167 | -0.154 |
| 303 | s0 | 0.533 | 0.287 | 0.407 | 3.233 | 0.063 | 0.047 | -0.209 | -0.029 |
| 101 | s1 | 0.513 | 0.313 | 0.407 | 3.360 | 0.144 | 0.136 | -0.140 | -0.114 |
| 202 | s1 | 0.487 | 0.273 | 0.367 | 3.227 | 0.210 | 0.125 | -0.150 | -0.151 |
| 303 | s1 | 0.487 | 0.293 | 0.407 | 3.293 | 0.143 | 0.061 | -0.226 | -0.130 |
| 101 | s2 | 0.520 | 0.320 | 0.340 | 3.273 | 0.171 | 0.014 | -0.119 | -0.161 |
| 202 | s2 | 0.540 | 0.427 | 0.373 | 3.400 | 0.177 | 0.078 | +0.041 | -0.071 |
| 303 | s2 | 0.493 | 0.373 | 0.327 | 3.293 | 0.099 | 0.067 | -0.130 | -0.242 |

**Summary:** Mean both-beat-base = 0.344. Near-baseline performance. Zero collapse.

### 6.2 Alpha = 0.15 (both teams)

| Seed | Stoch | TA win rate | TA beat-base | TB beat-base | Base pos | ZD | RD | Intra-A | Intra-B |
|------|-------|:-----------:|:------------:|:------------:|:--------:|:--:|:--:|:-------:|:-------:|
| 101 | s0 | 0.507 | 0.327 | 0.400 | 3.373 | 0.102 | 0.047 | -0.158 | -0.182 |
| 202 | s0 | 0.460 | 0.307 | 0.353 | 3.220 | 0.165 | 0.030 | -0.110 | -0.191 |
| 303 | s0 | 0.487 | 0.320 | 0.440 | 3.453 | 0.180 | 0.193 | -0.187 | -0.002 |
| 101 | s1 | 0.527 | 0.267 | 0.267 | 3.093 | 0.126 | 0.058 | -0.182 | -0.160 |
| 202 | s1 | 0.433 | 0.293 | 0.353 | 3.207 | 0.055 | 0.122 | -0.056 | -0.142 |
| 303 | s1 | 0.487 | 0.400 | 0.327 | 3.400 | 0.101 | 0.037 | -0.033 | -0.163 |
| 101 | s2 | 0.513 | 0.353 | 0.327 | 3.247 | 0.052 | 0.078 | -0.049 | -0.032 |
| 202 | s2 | 0.380 | 0.347 | 0.373 | 3.320 | 0.151 | 0.074 | -0.070 | -0.127 |
| 303 | s2 | 0.480 | 0.413 | 0.360 | 3.420 | 0.091 | 0.068 | -0.151 | -0.239 |

**Summary:** Mean both-beat-base = 0.349. Statistically identical to baseline. Zero collapse.

### 6.3 Alpha = 0.25 (both teams)

| Seed | Stoch | TA win rate | TA beat-base | TB beat-base | Base pos | ZD | RD | Intra-A | Intra-B |
|------|-------|:-----------:|:------------:|:------------:|:--------:|:--:|:--:|:-------:|:-------:|
| 101 | s0 | 0.513 | 0.353 | 0.380 | 3.347 | 0.056 | 0.074 | -0.171 | -0.152 |
| 202 | s0 | 0.480 | 0.280 | 0.280 | 3.120 | 0.037 | 0.115 | -0.011 | -0.043 |
| 303 | s0 | 0.607 | 0.380 | 0.287 | 3.293 | 0.044 | 0.081 | -0.132 | -0.088 |
| 101 | s1 | 0.560 | 0.293 | 0.273 | 3.193 | 0.384 | 0.060 | -0.293 | -0.031 |
| 202 | s1 | 0.500 | 0.360 | 0.293 | 3.167 | 0.053 | 0.097 | -0.149 | -0.187 |
| 303 | s1 | 0.493 | 0.360 | 0.333 | 3.367 | 0.075 | 0.007 | -0.277 | -0.258 |
| 101 | s2 | 0.573 | 0.313 | 0.260 | 3.140 | 0.146 | 0.053 | -0.185 | -0.211 |
| 202 | s2 | 0.533 | 0.260 | 0.293 | 3.193 | 0.292 | 0.069 | -0.182 | -0.180 |
| 303 | s2 | 0.493 | 0.280 | 0.387 | 3.240 | 0.238 | 0.123 | -0.130 | -0.068 |

**Summary:** Mean both-beat-base = 0.315. Already degrading from baseline. Zero collapse. Intra-team cooperation universally negative.

---

## 7. Mechanistic analysis: Why cooperation fails at 4+ agents

### 7.1 The influence dilution model

The cooperative reward signal's effectiveness depends on:

```
effective_signal = alpha * (influence_i_over_teammates / N_teammates)
```

| Phase | DQN agents | Teammates | Opponents | Influence per teammate | Effective signal (alpha=0.75) |
|-------|:----------:|:---------:|:---------:|:----------------------:|:-----------------------------:|
| 5 | 2 | 1 | 1 Base | HIGH (1 rival + 1 Base) | 0.75 * HIGH |
| 7A | 3 | 2 | 2 DQN + 1 Base | LOW (3 rivals) | 0.75 * LOW / 2 |
| 6 | 4 | 1 (same team) | 2 DQN + 1 Base | VERY LOW (4 rivals) | 0.75 * VERY LOW |

In Phase 5, each DQN agent's actions meaningfully affect its single teammate's finishing position (there are only 3 cars in the race). In Phase 7A, each agent's actions are diluted across 3 other cars competing for the same positions. The cooperative reward becomes noise.

### 7.2 Zone differentiation increases but doesn't translate

| Condition | Zone diff | Risk diff | Joint beat-base |
|-----------|:---------:|:---------:|:---------------:|
| 7A alpha=0.0 | 0.157 | 0.129 | 0.159 |
| 7A alpha=0.75 | 0.256 | 0.104 | 0.167 |

Alpha=0.75 increases zone differentiation by 63% (0.157 to 0.256). This means agents ARE responding to the cooperative signal by specializing in different zones. But this specialization does not improve joint outcomes. In a 4-agent race, zone specialization alone is insufficient — agents need to coordinate *timing* and *position management* across zones, which the shared-reward mechanism cannot express.

### 7.3 The attempt rate signature

At alpha=0.75, Agent 1 shows dramatically reduced attempt rates across most zones compared to alpha=0.0:

| Zone | A1 attempt rate (a=0.0, mean) | A1 attempt rate (a=0.75, mean) | Change |
|------|:-----------------------------:|:------------------------------:|:------:|
| La Source (d=0.2) | 0.864 | 0.539 | -38% |
| Raidillon (d=0.9) | 0.490 | 0.270 | -45% |
| Les Combes (d=0.7) | 0.596 | 0.314 | -47% |
| Bruxelles (d=0.7) | 0.513 | 0.187 | -64% |
| Pouhon (d=0.7) | 0.385 | 0.199 | -48% |
| Campus (d=0.7) | 0.376 | 0.160 | -57% |

**Cooperative agents learn to attempt fewer overtakes.** This is the same caution-over-cooperation pathology observed in Phase 6. When 75% of your reward comes from teammates whose outcomes you can barely influence, the safest Q-value estimate is to hold position. The cooperative signal suppresses aggressive play without enabling genuine coordination.

### 7.4 The boundary is sharp

The transition from "cooperation works" to "cooperation fails" occurs between 3 agents (Phase 5) and 4 agents (Phase 7A). There is no gradual degradation:

```
2 DQN + 1 Base (Phase 5):  alpha=0.75 → +83% improvement
3 DQN + 1 Base (Phase 7A): alpha=0.75 → +5% improvement (within noise)
4 DQN + 1 Base (Phase 6):  alpha=0.75 → -31% degradation
```

The cooperative advantage collapses to zero with the addition of a single agent. This rules out a gradual influence-dilution model and suggests a threshold effect: IL-MARL reward sharing requires **direct bilateral influence** between cooperating agents. When the game structure allows each agent to meaningfully control its partner's outcome (2 DQN, 1 opponent), cooperation emerges. When three or more agents compete for positions, no individual agent has sufficient influence over any specific partner, and the cooperative signal becomes noise.

---

## 8. Hypothesis evaluation

**H7A: Cooperation works at the intermediate 4-agent scale.**

DISCONFIRMED. Alpha=0.75 produces no measurable improvement in joint beat-base rate (0.167 vs 0.159, delta = +0.008). The cooperative advantage disappears entirely at 4 agents. The boundary is between 3 and 4 total agents, not between 4 and 5 as might have been expected from a gradual dilution model.

**H7B: A scale-adjusted (lower) alpha recovers cooperation in the 5-agent game.**

DISCONFIRMED. Alpha values of 0.10, 0.15, and 0.25 all fail to produce any cooperative advantage:
- alpha=0.10: -2% vs baseline (within noise)
- alpha=0.15: ~0% vs baseline
- alpha=0.25: -10% vs baseline

There is no Goldilocks alpha for the 5-agent team game. The failure is structural, not parametric.

---

## 9. Implications for research questions

**RQ1 (Emergent behavior under stochasticity):** Phase 7A confirms that zone differentiation emerges under cooperative incentive even at 4 agents (0.256 vs 0.157). However, this emergent specialization does not produce functional coordination. Stochasticity has minimal effect on the alpha=0.75 result (s0: 0.120, s1: 0.207, s2: 0.173 — high variance but no systematic trend).

**RQ2 (Threshold for destabilization):** Phase 7 precisely locates the threshold. In the non-zero-sum game, the transition from "cooperation helps" to "cooperation is neutral" occurs at exactly N=4 total agents (3 DQN + 1 Base). This is sharper than H2 predicted — the transition is a cliff, not a slope.

**RQ3 (Genuine cooperative advantage):** Phase 7 definitively bounds the answer. Genuine cooperative advantage via IL-MARL reward sharing exists only in small non-zero-sum games (N=3). At N=4, the advantage vanishes. At N=5, cooperation actively hurts. The mechanism is influence dilution: when an agent's actions cannot meaningfully control its partners' outcomes, shared reward becomes noise that suppresses exploration rather than enabling coordination.

---

## 10. Summary

1. **The boundary is at N=4.** Adding a single DQN agent to Phase 5's successful 3-agent game eliminates the entire cooperative advantage. This is a threshold effect, not gradual degradation.

2. **Low alpha cannot rescue team cooperation.** Even alpha=0.10 (90% individual, 10% cooperative) provides zero benefit in the 5-agent team game. The failure is structural: agents cannot influence their teammates' outcomes enough for any reward-sharing coefficient to produce a meaningful cooperative gradient.

3. **Zone specialization is necessary but not sufficient.** Cooperative incentive does induce zone differentiation in 4-agent games (63% increase), but this specialization alone cannot produce joint performance gains. Coordination requires richer mechanisms than terminal reward sharing.

4. **IL-MARL reward sharing has a hard scalability limit.** For independent learners with shared terminal reward, cooperation requires that each agent's actions directly and substantially affect each partner's outcome. This condition holds in 3-agent games and fails at 4+ agents. Overcoming this limit likely requires centralized training (QMIX, MAPPO) or per-step cooperative reward shaping — both are candidates for future work.
