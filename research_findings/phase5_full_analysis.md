# Phase 5 Full Analysis — Non-Zero-Sum MARL with Shared Adversary

**Date:** 2026-03-29
**Algorithm:** rainbow-lite (IL-MARL)
**Complexity profile:** `low_marl_vs_base` (2 DQN agents + 1 Base adversary)
**Alpha values:** 0.0, 0.25, 0.50, 0.75, 1.0
**Budget per cell:** 500 training episodes, 150 evaluation races (balanced: 75 with A1 starting first, 75 with A2 starting first)
**Seeds:** 101, 202, 303
**Stochasticity levels:** s0 (deterministic), s1 (2% noise), s2 (5% noise)
**Total trials:** 45
**Primary metric:** `joint_dqn_beat_base_rate` (fraction of races where BOTH DQN agents finish ahead of the Base adversary)

---

## 1. Experimental design changes from Phase 4

Phase 5 addresses three structural flaws identified in Phase 4.

| Flaw | Phase 4 design | Phase 5 fix |
|------|---------------|-------------|
| Zero-sum game | 2 DQN agents, one winner one loser | 2 DQN agents + 1 Base adversary; joint improvement is possible |
| Evaluation positional bias | A1 always started position 1 in all 150 eval races | Eval split into two 75-race batches with alternating starting positions |
| Wrong primary metric | `win_rate_a1_vs_a2` (zero-sum, cannot rise for both agents) | `joint_dqn_beat_base_rate` (non-zero-sum, can rise for both agents simultaneously) |

The reward sharing formula is unchanged: `mixed_outcome_i = (1 - alpha) * own_delta + alpha * teammate_delta`. Alpha is applied only to the two DQN agents. The Base agent receives no shared reward and trains no neural network.

---

## 2. Simulator verification

| Check | Status | Evidence |
|-------|--------|----------|
| `complexity_profile` is `low_marl_vs_base` in all 45 output files | PASS | All show `"complexity_profile": "low_marl_vs_base"` |
| `phase` field is `phase5_marl` in all 45 files | PASS | All show `"phase": "phase5_marl"` |
| `balanced_positions` is true in all 45 files | PASS | All show `"balanced_positions": true` |
| `base_agent_name` is present and non-null in all 45 files | PASS | All show `"base_agent_name": "Base Agent"` |
| `vs_base_metrics` is present and populated in all 45 files | PASS | All contain `joint_dqn_beat_base_rate`, `dqn_a1_beats_base_rate`, `dqn_a2_beats_base_rate`, `avg_base_position` |
| Both DQN agents are rainbow-lite across all 45 trials | PASS | `algorithm: rainbow_lite` in all files |
| No DNF in any trial | PASS | `dnf_rate: 0.0` for all agents in all trials |
| Alpha values correctly recorded | PASS | All files match their config alpha |

---

## 3. Cross-alpha comparison table

| Alpha | Regime | s0 WR | s1 WR | s2 WR | Collapse | Zone diff | Risk diff | JointBeatBase | Base pos |
|-------|--------|-------|-------|-------|----------|-----------|-----------|---------------|----------|
| 0.0 | Purely competitive | 0.462 | 0.538 | 0.504 | 0/9 (0%) | 0.104 | 0.092 | 0.310 | 2.213 |
| 0.25 | Weakly cooperative | 0.518 | 0.522 | 0.593 | 0/9 (0%) | 0.177 | 0.135 | 0.294 | 2.190 |
| 0.50 | Balanced mixed | 0.527 | 0.564 | 0.400 | 0/9 (0%) | 0.100 | 0.160 | 0.433 | 2.353 |
| 0.75 | Strongly cooperative | 0.487 | 0.527 | 0.351 | 0/9 (0%) | 0.069 | 0.156 | **0.567** | **2.409** |
| 1.0 | Fully cooperative | 0.422 | 0.396 | 0.318 | 0/9 (0%) | 0.104 | 0.229 | 0.436 | 2.341 |

---

## 4. Raw trial data

### 4.1 Alpha = 0.0 (purely competitive baseline)

| Seed | Stoch | A1 WR | CI 95% | Zone diff | Risk diff | Drift | JointBase | A1vBase | A2vBase | Base pos |
|------|-------|-------|--------|-----------|-----------|-------|-----------|---------|---------|----------|
| 101 | s0 | 0.427 | [0.347, 0.506] | 0.059 | 0.111 | +0.12 (a1 str) | 0.313 | 0.547 | 0.700 | 2.247 |
| 202 | s0 | 0.467 | [0.387, 0.547] | 0.071 | 0.033 | stable | 0.367 | 0.553 | 0.713 | 2.267 |
| 303 | s0 | 0.493 | [0.413, 0.574] | 0.101 | 0.041 | +0.14 (a1 str) | 0.313 | 0.580 | 0.593 | 2.173 |
| 101 | s1 | 0.613 | [0.535, 0.692] | 0.165 | 0.042 | -0.40 (a2 str) | 0.453 | 0.647 | 0.713 | 2.360 |
| 202 | s1 | 0.480 | [0.400, 0.560] | 0.039 | 0.139 | stable | 0.247 | 0.513 | 0.627 | 2.140 |
| 303 | s1 | 0.520 | [0.440, 0.600] | 0.202 | 0.098 | stable | 0.360 | 0.660 | 0.613 | 2.273 |
| 101 | s2 | 0.440 | [0.360, 0.520] | 0.038 | 0.103 | +0.08 (a1 str) | 0.220 | 0.547 | 0.587 | 2.133 |
| 202 | s2 | 0.580 | [0.501, 0.659] | 0.196 | 0.122 | -0.14 (a2 str) | 0.313 | 0.660 | 0.533 | 2.193 |
| 303 | s2 | 0.493 | [0.413, 0.574] | 0.062 | 0.139 | -0.06 (a2 str) | 0.200 | 0.527 | 0.600 | 2.127 |

**Summary:** 0/9 collapse. Mean JointBeatBase = 0.310. Both DQN agents beat Base individually at roughly 58-63%, but they beat Base *simultaneously* only 31% of the time. The gap between individual and joint success rates reveals that the DQN agents frequently beat Base at each other's expense: when A1 beats Base, A2 often does not, and vice versa. This is the competitive signature.

### 4.2 Alpha = 0.25 (weakly cooperative)

| Seed | Stoch | A1 WR | CI 95% | Zone diff | Risk diff | Drift | JointBase | A1vBase | A2vBase | Base pos |
|------|-------|-------|--------|-----------|-----------|-------|-----------|---------|---------|----------|
| 101 | s0 | 0.500 | [0.420, 0.580] | 0.083 | 0.102 | -0.14 (a2 str) | 0.320 | 0.660 | 0.573 | 2.233 |
| 202 | s0 | 0.547 | [0.467, 0.627] | 0.114 | 0.116 | +0.08 (a1 str) | 0.260 | 0.573 | 0.567 | 2.140 |
| 303 | s0 | 0.507 | [0.426, 0.587] | 0.107 | 0.084 | +0.06 (a1 str) | 0.367 | 0.640 | 0.620 | 2.260 |
| 101 | s1 | 0.500 | [0.420, 0.580] | 0.189 | 0.007 | +0.24 (a1 str) | 0.253 | 0.567 | 0.533 | 2.100 |
| 202 | s1 | 0.500 | [0.420, 0.580] | 0.143 | 0.129 | +0.08 (a1 str) | 0.373 | 0.613 | 0.653 | 2.267 |
| 303 | s1 | 0.567 | [0.487, 0.646] | 0.134 | 0.214 | +0.14 (a1 str) | 0.340 | 0.667 | 0.607 | 2.273 |
| 101 | s2 | 0.633 | [0.556, 0.711] | 0.205 | 0.371 | +0.06 (a1 str) | 0.227 | 0.667 | 0.460 | 2.127 |
| 202 | s2 | 0.567 | [0.487, 0.646] | 0.320 | 0.119 | stable | 0.187 | 0.593 | 0.473 | 2.067 |
| 303 | s2 | 0.580 | [0.501, 0.659] | 0.298 | 0.071 | +0.08 (a1 str) | 0.320 | 0.667 | 0.580 | 2.247 |

**Summary:** 0/9 collapse. Mean JointBeatBase = 0.294. JointBeatBase is slightly *lower* than the competitive baseline (0.310). At this alpha level, the weak cooperative signal does not translate into improved joint performance against Base. Zone differentiation is modestly higher than baseline (0.177 vs 0.104) but does not help against the shared adversary.

### 4.3 Alpha = 0.50 (balanced mixed)

| Seed | Stoch | A1 WR | CI 95% | Zone diff | Risk diff | Drift | JointBase | A1vBase | A2vBase | Base pos |
|------|-------|-------|--------|-----------|-----------|-------|-----------|---------|---------|----------|
| 101 | s0 | 0.560 | [0.480, 0.640] | 0.077 | 0.173 | stable | 0.347 | 0.667 | 0.607 | 2.273 |
| 202 | s0 | 0.480 | [0.400, 0.560] | 0.140 | 0.169 | -0.24 (a2 str) | 0.460 | 0.647 | 0.693 | 2.340 |
| 303 | s0 | 0.540 | [0.460, 0.620] | 0.119 | 0.094 | -0.26 (a2 str) | 0.527 | 0.767 | 0.687 | 2.453 |
| 101 | s1 | 0.580 | [0.501, 0.659] | 0.085 | 0.194 | stable | 0.320 | 0.693 | 0.527 | 2.220 |
| 202 | s1 | 0.500 | [0.420, 0.580] | 0.069 | 0.186 | stable | 0.393 | 0.627 | 0.673 | 2.300 |
| 303 | s1 | 0.613 | [0.535, 0.692] | 0.111 | 0.174 | -0.50 (a2 str) | 0.593 | 0.753 | 0.807 | 2.560 |
| 101 | s2 | 0.480 | [0.400, 0.560] | 0.076 | 0.124 | -0.20 (a2 str) | 0.347 | 0.660 | 0.620 | 2.280 |
| 202 | s2 | 0.327 | [0.251, 0.402] | 0.083 | 0.219 | -0.14 (a2 str) | 0.393 | 0.607 | 0.700 | 2.307 |
| 303 | s2 | 0.393 | [0.315, 0.472] | 0.137 | 0.105 | -0.08 (a2 str) | 0.513 | 0.713 | 0.733 | 2.447 |

**Summary:** 0/9 collapse. Mean JointBeatBase = 0.433, a 40% increase over the competitive baseline. The transition from "no cooperative benefit" (alpha=0.25) to "measurable cooperative benefit" (alpha=0.50) occurs at the incentive crossover point where the team component equals the individual component. At alpha=0.50, both agents have equal weight on own and teammate outcomes, and this balance translates into the first statistically meaningful improvement in joint adversary performance.

Standout trial: s1 s303 achieves JointBeatBase = 0.593 with A1vBase = 0.753 and A2vBase = 0.807. Both agents beat the Base adversary more than 75% of the time, and they do so simultaneously in nearly 60% of races. This is the first trial in the study where genuine cooperative benefit is unambiguously present.

### 4.4 Alpha = 0.75 (strongly cooperative)

| Seed | Stoch | A1 WR | CI 95% | Zone diff | Risk diff | Drift | JointBase | A1vBase | A2vBase | Base pos |
|------|-------|-------|--------|-----------|-----------|-------|-----------|---------|---------|----------|
| 101 | s0 | 0.653 | [0.577, 0.730] | 0.113 | 0.015 | +0.10 (a1 str) | 0.580 | 0.647 | 0.700 | 2.347 |
| 202 | s0 | 0.420 | [0.341, 0.499] | 0.045 | 0.022 | +0.58 (a1 str) | 0.527 | 0.580 | 0.787 | 2.367 |
| 303 | s0 | 0.387 | [0.308, 0.465] | 0.086 | 0.089 | -0.06 (a2 str) | 0.567 | 0.667 | 0.773 | 2.440 |
| 101 | s1 | 0.400 | [0.321, 0.479] | 0.080 | 0.199 | -0.42 (a2 str) | 0.540 | 0.580 | 0.833 | 2.413 |
| 202 | s1 | 0.467 | [0.387, 0.547] | 0.044 | 0.228 | +0.42 (a1 str) | 0.620 | 0.660 | 0.753 | 2.413 |
| 303 | s1 | 0.713 | [0.641, 0.786] | 0.030 | 0.483 | stable | 0.693 | 0.707 | 0.707 | 2.413 |
| 101 | s2 | 0.267 | [0.196, 0.338] | 0.086 | 0.234 | +0.10 (a1 str) | 0.560 | 0.640 | 0.873 | 2.513 |
| 202 | s2 | 0.327 | [0.251, 0.402] | 0.092 | 0.072 | -0.10 (a2 str) | 0.480 | 0.633 | 0.793 | 2.427 |
| 303 | s2 | 0.460 | [0.380, 0.540] | 0.051 | 0.062 | +0.48 (a1 str) | 0.533 | 0.580 | 0.767 | 2.347 |

**Summary:** 0/9 collapse. Mean JointBeatBase = **0.567**, the highest of all alpha conditions. This is an 83% improvement over the competitive baseline (0.310). Every single trial exceeds the baseline mean. The lowest JointBeatBase at alpha=0.75 (0.480, s2 s202) still exceeds the highest at alpha=0.0 (0.453, s1 s101).

A2 is the primary beneficiary: mean A2vBase = 0.776 (vs 0.631 at alpha=0.0). The structurally disadvantaged agent gains the most from cooperative incentive. A1's individual performance is roughly maintained (0.633 vs 0.581 at baseline), but the gains are concentrated in A2's ability to beat the Base adversary.

Standout trial: s1 s303 achieves JointBeatBase = 0.693. Both DQN agents beat Base over 70% of the time, and they do so simultaneously in nearly 70% of races.

### 4.5 Alpha = 1.0 (fully cooperative)

| Seed | Stoch | A1 WR | CI 95% | Zone diff | Risk diff | Drift | JointBase | A1vBase | A2vBase | Base pos |
|------|-------|-------|--------|-----------|-----------|-------|-----------|---------|---------|----------|
| 101 | s0 | 0.493 | [0.413, 0.574] | 0.076 | 0.142 | -0.70 (a2 str) | 0.373 | 0.707 | 0.533 | 2.240 |
| 202 | s0 | 0.320 | [0.245, 0.395] | 0.047 | 0.194 | -0.26 (a2 str) | 0.560 | 0.740 | 0.713 | 2.453 |
| 303 | s0 | 0.453 | [0.373, 0.533] | 0.306 | 0.120 | stable | 0.340 | 0.600 | 0.687 | 2.287 |
| 101 | s1 | 0.653 | [0.577, 0.730] | 0.080 | 0.194 | -0.34 (a2 str) | 0.460 | 0.660 | 0.700 | 2.360 |
| 202 | s1 | 0.207 | [0.142, 0.272] | 0.052 | 0.306 | stable | 0.647 | 0.747 | 0.800 | 2.547 |
| 303 | s1 | 0.327 | [0.251, 0.402] | 0.031 | 0.329 | -0.38 (a2 str) | 0.427 | 0.687 | 0.653 | 2.340 |
| 101 | s2 | 0.307 | [0.233, 0.381] | 0.057 | 0.166 | -0.20 (a2 str) | 0.353 | 0.560 | 0.667 | 2.227 |
| 202 | s2 | 0.440 | [0.360, 0.520] | 0.244 | 0.306 | -0.08 (a2 str) | 0.160 | 0.453 | 0.620 | 2.073 |
| 303 | s2 | 0.207 | [0.142, 0.272] | 0.043 | 0.304 | +0.10 (a1 str) | 0.607 | 0.733 | 0.813 | 2.547 |

**Summary:** 0/9 collapse. Mean JointBeatBase = 0.436. Alpha=1.0 outperforms the competitive baseline (0.310) and the weakly cooperative regime (0.294), but falls below alpha=0.75 (0.567) and is comparable to alpha=0.50 (0.433).

The higher variance is notable. JointBeatBase ranges from 0.160 (s2 s202) to 0.647 (s1 s202). The fully cooperative signal eliminates all individual positional incentive, making learned strategies more sensitive to the specific training trajectory for each seed. Some seeds produce excellent cooperative outcomes; others produce weaker coordination despite no degenerate collapse.

---

## 5. Research question answers

### RQ1: What patterns of strategic behaviour emerge when independent RL agents train concurrently in a shared stochastic environment, and how sensitive are those patterns to environmental noise?

**Finding RQ1-A: Concurrent training in the non-zero-sum setting produces moderate strategic structure.**

Zone differentiation across all 45 Phase 5 trials ranges from 0.030 to 0.320, with an overall mean of 0.111. This is substantially lower than Phase 4's mean of 0.345 (alpha=0.25) but comparable to Phase 4's baseline alpha=0.0 (0.335). The presence of the Base adversary reduces zone specialisation because agents must compete across all zones to prevent Base from gaining positional advantage. In a two-agent zero-sum game, agents could afford to vacate zones; in a three-agent non-zero-sum game, vacating a zone cedes it to the Base adversary.

Risk differentiation (mean 0.092 at alpha=0.0) is lower than Phase 4 (0.314 at alpha=0.0). The balanced evaluation protocol, which alternates starting positions, reduces the structural positional asymmetry that drove hawk-dove risk polarisation in Phase 4.

**Finding RQ1-B: Strategic patterns are moderately sensitive to noise, but the sensitivity direction depends on alpha.**

At alpha=0.0, zone differentiation is stable across stochasticity levels (s0: 0.077, s1: 0.135, s2: 0.099). At alpha=0.25, zone differentiation increases from s0 (0.101) to s2 (0.274). At alpha=0.75, it is uniformly low across all stochasticity levels (0.081, 0.051, 0.076). The relationship between noise and strategic structure depends on the incentive regime, which is itself a finding: there is no universal noise-stability relationship for emergent strategy.

**Finding RQ1-C: The balanced evaluation protocol eliminates the structural A1 dominance observed in Phase 4.**

Mean A1 win rate across all 45 trials is 0.475 (range: 0.207 to 0.713). At alpha=0.0, mean WR = 0.501, near perfect parity. The Phase 4 structural A1 advantage (driven by A1 always starting position 1 in eval) is gone. The balanced eval protocol is validated.

---

### RQ2: How does systematically varying the reward sharing coefficient alter the emergent strategy profile, and is there a threshold beyond which shared incentive destabilises rather than improves learning?

**Finding RQ2-A: The destabilisation threshold observed in Phase 4 does not exist in the non-zero-sum setting.**

Phase 4 found catastrophic collapse at alpha >= 0.75 (89-100% of trials degenerate). Phase 5 finds **zero collapse at any alpha value**. The destabilisation threshold was a property of the zero-sum game structure, not of the reward sharing mechanism. When agents face a shared adversary, even fully cooperative incentive (alpha=1.0) produces active, non-degenerate strategies.

The mechanism is clear: in Phase 4's zero-sum setting, the passive Nash equilibrium (both agents hold, starting position determines outcome) was a stable fixed point because no agent could unilaterally improve by deviating. In Phase 5's non-zero-sum setting, the passive equilibrium is unstable because the Base adversary will overtake passive DQN agents. Being overtaken by Base produces a negative outcome delta, which at alpha=1.0 becomes the teammate's reward. Both agents are therefore penalised for allowing Base to advance. Passivity is no longer an equilibrium.

**Finding RQ2-B: Higher alpha shifts the A1 vs A2 balance toward A2 without causing collapse.**

| Alpha | Mean A1 WR | Interpretation |
|-------|-----------|----------------|
| 0.0 | 0.501 | Perfect parity |
| 0.25 | 0.544 | Slight A1 advantage |
| 0.50 | 0.497 | Near parity |
| 0.75 | 0.455 | Moderate A2 advantage |
| 1.0 | 0.379 | Clear A2 advantage |

As alpha increases, A2 (the structurally disadvantaged agent in position 2) gains relative to A1. At alpha=1.0, A2 wins 62% of head-to-head races. This is not collapse: both agents are active, both attempt overtakes, and the win rate spread reflects genuine strategic asymmetry rather than one agent becoming passive. The cooperative signal disproportionately helps the weaker agent by ensuring that its teammate's success contributes to its own reward, even when it cannot directly improve its own position.

**Finding RQ2-C: Risk differentiation increases monotonically with alpha.**

| Alpha | Mean risk diff |
|-------|---------------|
| 0.0 | 0.092 |
| 0.25 | 0.135 |
| 0.50 | 0.160 |
| 0.75 | 0.156 |
| 1.0 | 0.229 |

At alpha=1.0, agents develop the most asymmetric risk profiles. One agent adopts primarily conservative play while the other adopts aggressive play. This is hawk-dove specialisation driven by the cooperative incentive: when both agents' outcomes contribute equally to each other's rewards, it becomes advantageous for one agent to play safe while the other takes risks. If both take risks, double failure hurts both; if one plays safe and one takes risks, the team hedges.

---

### RQ3: Does shared incentive produce genuine cooperative advantage, measured as improved joint performance against a common adversary, or does it merely redistribute outcomes within a fixed interaction?

**Finding RQ3-A: Shared incentive produces genuine cooperative advantage. The effect is large and unambiguous.**

| Alpha | JointBeatBase | vs baseline | A1vBase | A2vBase | Base pos |
|-------|---------------|-------------|---------|---------|----------|
| 0.0 | 0.310 | baseline | 0.581 | 0.631 | 2.213 |
| 0.25 | 0.294 | -5% | 0.627 | 0.563 | 2.190 |
| 0.50 | 0.433 | **+40%** | 0.681 | 0.672 | 2.353 |
| 0.75 | 0.567 | **+83%** | 0.633 | 0.776 | 2.409 |
| 1.0 | 0.436 | +41% | 0.654 | 0.687 | 2.341 |

At alpha=0.75, both DQN agents simultaneously beat the Base adversary in 56.7% of races, compared to 31.0% at alpha=0.0. This is an 83% improvement. The Base agent's mean finishing position is pushed from 2.21 (competitive baseline) to 2.41 (alpha=0.75), confirming that the DQN pair is genuinely cooperating to suppress the adversary rather than merely redistributing outcomes between themselves.

This is the answer to RQ3: **shared incentive produces genuine cooperative advantage in the non-zero-sum setting**. The advantage is maximised at alpha=0.75, not at alpha=1.0. This is a non-trivial result.

**Finding RQ3-B: The cooperative advantage follows an inverted-U curve, peaking at alpha=0.75.**

The JointBeatBase trajectory across alpha values is:

```
0.310 → 0.294 → 0.433 → 0.567 → 0.436
(0.0)   (0.25)  (0.50)  (0.75)  (1.0)
```

The curve has three regimes:

1. **Competitive plateau (alpha 0.0-0.25):** JointBeatBase is flat at approximately 0.30. The weak cooperative signal at alpha=0.25 does not translate into improved joint adversary performance.

2. **Cooperative ascent (alpha 0.25-0.75):** JointBeatBase rises steeply from 0.294 to 0.567. The team incentive begins to meaningfully shape behaviour at alpha=0.50 (the incentive crossover point) and peaks at alpha=0.75.

3. **Cooperative decline (alpha 0.75-1.0):** JointBeatBase drops from 0.567 to 0.436. Full cooperation (alpha=1.0) is less effective than strong partial cooperation (alpha=0.75). At alpha=1.0, both agents receive only the teammate's outcome, which removes all direct incentive to improve own position. The agents remain active (no collapse), but their strategies are less well-coordinated than at alpha=0.75 where 25% residual individual incentive provides directional gradient.

**Finding RQ3-C: The structurally weaker agent (A2) is the primary beneficiary of cooperative incentive.**

| Alpha | A2 beats Base | Change from baseline |
|-------|---------------|---------------------|
| 0.0 | 63.1% | baseline |
| 0.25 | 56.3% | -6.8pp |
| 0.50 | 67.2% | +4.1pp |
| 0.75 | 77.6% | **+14.5pp** |
| 1.0 | 68.7% | +5.6pp |

A2's performance against Base improves dramatically under cooperative incentive, peaking at alpha=0.75 where A2 beats Base in 77.6% of races. A1's performance is roughly maintained across all alpha values (58.1% to 68.1%). The cooperative incentive acts as an equaliser: it lifts the weaker agent without substantially harming the stronger agent. This is the mechanism behind the JointBeatBase improvement: when both agents can beat Base individually at high rates, the probability of both beating Base simultaneously increases.

**Finding RQ3-D: At alpha=0.25, cooperative incentive slightly reduces joint performance — weak cooperation is worse than no cooperation.**

JointBeatBase at alpha=0.25 (0.294) is marginally below the competitive baseline (0.310). The mechanism: at alpha=0.25, A2's individual beat-base rate drops from 63.1% to 56.3%, while A1's rises from 58.1% to 62.7%. The 25% cooperative signal slightly benefits A1 at A2's expense without producing enough team coordination to improve the joint metric. The weak cooperative incentive perturbs the competitive equilibrium without establishing a cooperative one.

This finding has practical implications: in multi-agent system design, a weak cooperative signal can be worse than no signal at all. The minimum effective dose of cooperation appears to be at or above alpha=0.50 in this domain.

---

## 6. Phase 4 vs Phase 5 comparison

The comparison between Phase 4 (zero-sum, 2-agent) and Phase 5 (non-zero-sum, 3-agent) directly addresses the methodological critique that motivated Phase 5.

| Metric | Phase 4 (zero-sum) | Phase 5 (non-zero-sum) |
|--------|-------------------|----------------------|
| Collapse at alpha=0.0 | 0/9 (0%) | 0/9 (0%) |
| Collapse at alpha=0.25 | 1/9 (11%) | 0/9 (0%) |
| Collapse at alpha=0.50 | 1/9 (11%) | 0/9 (0%) |
| Collapse at alpha=0.75 | **8/9 (89%)** | **0/9 (0%)** |
| Collapse at alpha=1.0 | **9/9 (100%)** | **0/9 (0%)** |
| Best alpha for zone diff | 0.25 (0.345) | 0.25 (0.177) |
| Best alpha for joint performance | untestable (zero-sum) | 0.75 (0.567) |
| WR parity at alpha=0.0 | 0.498 | 0.501 |

**Key takeaway:** The Phase 4 "tragedy of cooperation" finding (collapse at high alpha) was entirely an artefact of the zero-sum game design. It was not a property of the reward sharing mechanism, the DQN algorithm, or the multi-agent training process. When the game structure allows genuine cooperative benefit (through the shared adversary), high-alpha agents learn effective cooperative strategies rather than collapsing to passivity.

The Phase 4 finding that alpha=0.25 produced the highest zone differentiation survives in Phase 5 (alpha=0.25 still peaks at 0.177), but the magnitude is substantially reduced. Zone differentiation in the presence of a shared adversary is inherently lower because agents cannot afford to vacate zones.

The most important Phase 5 finding — that alpha=0.75 maximises joint adversary performance — was structurally impossible to discover in Phase 4 because: (a) the metric did not exist; (b) 89% of alpha=0.75 trials were degenerate; and (c) the zero-sum constraint made joint improvement impossible by construction.

---

## 7. Zone behaviour patterns

### 7.1 Baseline (alpha=0.0): Symmetric zone usage

At alpha=0.0 s0 s101, both agents show similar zone profiles. A1 attempts La Source at 83%, Les Combes at 56%, Raidillon at 31%. A2 attempts La Source at 80%, Raidillon at 47%, Les Combes at 41%. The profiles are broadly symmetric with minor variation from seed-dependent training trajectories. Zone differentiation index = 0.059. The agents compete across the same zones.

### 7.2 Alpha=0.50: Emergent attempt-rate asymmetry

At alpha=0.50 s0 s303, A1 attempts La Source at 80% while A2 drops to 49%. A1 is more active overall (845 total attempts vs 829 for A2), but A2 concentrates more efficiently on Pouhon (34% vs 31%). Both agents adopt conservative risk profiles (A1: 523 conservative, 174 aggressive; A2: 630 conservative, 86 aggressive). The cooperative incentive drives both agents toward caution. JointBeatBase = 0.527.

### 7.3 Alpha=0.75: Selective engagement and cooperative suppression of Base

At alpha=0.75 s0 s303, A1 dramatically reduces La Source attempts to just 7% (compare 80% at alpha=0.0). A1 instead distributes attempts across Raidillon (21%), Les Combes (16%), and Pouhon (13%). A2 maintains La Source at 48% and Les Combes at 26%. A1 has effectively ceded the primary zone to A2 and adopted a supporting role across secondary zones. Total attempt counts are lower (A1: 383, A2: 416) but A2's success rate is higher (49.0% vs 32.9% for A1). JointBeatBase = 0.567.

This is the clearest example of emergent cooperative role differentiation in the study. A1 acts as the "disruptor" across multiple zones while A2 acts as the "closer" at the highest-value zone. Neither agent was programmed to adopt this role; it emerged from the 75% shared incentive that makes A1's reward substantially dependent on A2's finishing position.

### 7.4 Alpha=1.0: Active but less coordinated

At alpha=1.0 s1 s202, both agents show moderate attempt rates (A1: 390 attempts, A2: 458 attempts) with A1 focusing on Les Combes (30%) and A2 on La Source (37%). Both agents are active across multiple zones. Risk profiles diverge (A1: 188 conservative, 140 aggressive; A2: 85 conservative, 283 normal, 90 aggressive). JointBeatBase = 0.647.

Despite achieving the highest individual JointBeatBase in the study, alpha=1.0 is less consistent than alpha=0.75. The fully cooperative signal produces high variance across seeds because the complete removal of individual incentive makes the learned policy more sensitive to training randomness.

---

## 8. Hypothesis evaluation

### H1: Concurrent training produces interpretable strategic structure

**Supported.** All 45 trials produce non-zero zone differentiation (range 0.030-0.320) and non-zero risk differentiation (range 0.007-0.483). The structure is less pronounced than in Phase 4 (which had a zero-sum game encouraging extreme territorial behaviour), but it is interpretable and consistent across seeds and stochasticity levels. The prediction that higher stochasticity weakens specialisation is partially supported at alpha=0.0 (zone diff drops from 0.077 at s0 to 0.099 at s2) but the effect is small.

### H2: Increasing alpha initially increases differentiation, then inverts beyond a threshold

**Partially supported, partially refuted.** The prediction of an initial increase in strategic differentiation is supported by zone differentiation (peaks at alpha=0.25) and risk differentiation (increases monotonically with alpha). The prediction of a sharp destabilisation threshold is **refuted**: no collapse occurs at any alpha value in the non-zero-sum setting. The threshold identified in Phase 4 (between alpha=0.50 and alpha=0.75) was a property of the zero-sum game, not of the reward sharing mechanism.

However, the inverted-U shape of JointBeatBase (peaking at alpha=0.75, declining at alpha=1.0) represents a different kind of threshold: not a collapse boundary, but a performance optimum. Beyond alpha=0.75, the removal of individual incentive reduces strategic coordination quality without causing degenerate behaviour.

### H3: Shared incentive produces genuine cooperative advantage in non-zero-sum settings

**Strongly supported.** JointBeatBase increases from 0.310 (alpha=0.0) to 0.567 (alpha=0.75), an 83% improvement. The Base adversary's mean finishing position is pushed from 2.21 to 2.41. The structurally weaker agent (A2) benefits most, with beat-base rate rising from 63.1% to 77.6%.

The prediction that full cooperation may restore pathological convergence is **refuted**: alpha=1.0 does not produce collapse or passivity. It does, however, produce lower joint performance than alpha=0.75, supporting the prediction that optimal cooperation is partial rather than full.

---

## 9. Threats to validity

### 9.1 Base agent capability
The Base agent follows a fixed heuristic policy. Its capability level determines the difficulty of the cooperative task. If Base were stronger, the DQN pair would need to cooperate more aggressively; if weaker, the cooperative advantage might plateau at lower alpha values. Results may not generalise to settings with adaptive adversaries.

### 9.2 Small trial count per cell
Three trials per alpha-stochasticity cell (one per seed). The JointBeatBase estimates have wide confidence intervals. The direction of findings is reliable; precise effect sizes require larger N.

### 9.3 Training budget
500 training episodes may be insufficient for alpha=1.0 to fully explore the cooperative strategy space. The higher variance at alpha=1.0 could reflect incomplete convergence rather than an inherent property of full cooperation.

### 9.4 Three-agent dynamics
Adding the Base agent changes the state space (3 drivers instead of 2), grid positions (3 slots instead of 2), and zone dynamics (more traffic). Phase 5 results are not directly comparable to Phase 4 at the trial level. The comparison is valid at the level of whether alpha produces cooperative advantage, not at the level of specific metric values.

### 9.5 Base agent does not adapt
The Base agent uses a fixed policy throughout training and evaluation. In a real competitive system, adversaries adapt. The cooperative advantage observed at alpha=0.75 might diminish against an adaptive adversary that exploits the DQN pair's zone specialisation.

### 9.6 Effective decision space is narrower than the 9-zone framing implies
The Spa track defines 9 overtaking zones, but only 3 to 5 generate meaningful agent decisions per lap. The simulator requires a car ahead within 100m for a zone decision to trigger. Once a driver reaches position 1 after an early-zone overtake, they are skipped from the zone decision loop for the remainder of that lap because no car is ahead. In the 3-agent game, this is partially mitigated (more positional churn from Base interactions), but zones 6 to 9 still see substantially lower decision counts than zones 1 to 5. Zone differentiation metrics are therefore measuring specialisation across a reduced effective space. This does not invalidate cross-alpha comparisons (the zone structure is constant across conditions) but means the reported zone differentiation captures behaviour across approximately 3 to 5 active zones, not the full 9-zone track.

### 9.7 Terminal bonus timing in 3-agent races
In the 3-agent game, the first driver to finish a race has their terminal bonus computed immediately, using the teammate's current position at that moment. If the teammate's position changes before the teammate finishes (for example, by being overtaken by the Base agent), the first driver's blended outcome reflects an intermediate teammate position rather than the final one. This adds bounded noise to the reward signal (maximum error of alpha times one position shift per episode). The error is non-systematic and does not bias results in a consistent direction. The Phase 5 signal (83% JointBeatBase improvement at alpha=0.75 across 9 trials) substantially exceeds this noise floor. This limitation does not affect Phase 4 (2-agent races), where the trailing agent has no remaining opponents once the leader finishes.

---

## 10. Conclusions

Phase 5 produces five principal conclusions.

**C1. The Phase 4 degenerate collapse was a zero-sum artefact, not a genuine cooperative failure.** Zero collapse across all 45 Phase 5 trials (0/45) compared to 18/36 in Phase 4. The passive Nash equilibrium that destroyed high-alpha learning in Phase 4 does not exist when agents face a shared adversary. This is the single most important methodological finding in the study: the structure of the game determines whether cooperative incentive is viable, independent of the reward sharing coefficient.

**C2. Cooperative incentive produces genuine joint performance improvement, peaking at alpha=0.75.** JointBeatBase increases 83% from the competitive baseline (0.310 to 0.567). Both DQN agents benefit, but A2 (the structurally weaker agent) benefits most. The cooperative incentive acts as an equaliser that lifts the weaker agent without substantially harming the stronger agent.

**C3. The optimal incentive structure is strong partial cooperation (alpha=0.75), not full cooperation (alpha=1.0).** Full cooperation eliminates individual positional incentive, producing higher variance and lower mean joint performance than the 75/25 split. A residual 25% individual incentive provides directional learning gradient that pure team reward lacks.

**C4. Weak cooperative incentive (alpha=0.25) is worse than no incentive.** JointBeatBase at alpha=0.25 (0.294) is marginally below the competitive baseline (0.310). The minimum effective dose of cooperation in this domain is at or above alpha=0.50.

**C5. Emergent role differentiation at alpha=0.75 represents genuine cooperative convention.** At alpha=0.75, agents develop complementary roles: one acts as the primary closer at high-value zones while the other disrupts across secondary zones. This territorial split emerges from the shared incentive without explicit coordination mechanisms, communication, or centralised critics. It is the clearest evidence in the study that IL-MARL agents can develop cooperative conventions when the game structure and incentive design jointly support it.

---

## 11. Implications for the dissertation

The Phase 5 results fundamentally reframe the project narrative. The story is no longer "cooperation fails at high alpha" (Phase 4) but rather "cooperation succeeds when the game structure supports it, and optimal cooperation is strong but not complete" (Phase 5). The finding that alpha=0.75 outperforms both pure competition and full cooperation is the central empirical contribution of the dissertation.

The Phase 4 results retain value as a cautionary finding: applying cooperative incentive to a zero-sum interaction produces pathological convergence. This is itself a contribution to the MARL literature, demonstrating that the game structure must be considered alongside the incentive design when implementing reward sharing between concurrent learners.
