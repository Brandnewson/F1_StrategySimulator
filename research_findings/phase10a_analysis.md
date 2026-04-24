# Phase 10A-i Analysis: Difference Rewards at N=4

**Date:** 2026-04-05 (updated 2026-04-06 with redux results)
**Setup:** 3 DQN agents + 1 Base adversary, `reward_mode=difference`, 500 training runs, 150 eval runs per trial

---

## 1. Executive Summary

Phase 10A-i tested difference rewards (Wolpert & Tumer, 2002) as an alternative credit assignment mechanism to alpha-blending at N=4 agents. Three experiment batches were conducted:

1. **Original (vanilla DQN, 9 trials):** Joint beat-base = 0.087 --- accidental algorithm mismatch
2. **Redux attempt 1 (vanilla DQN, 9 trials):** Joint beat-base = 0.081 --- config not updated, served as replication
3. **Redux corrected (rainbow_lite, 9 trials):** Joint beat-base = **0.133** --- valid comparison to Phase 7A

With rainbow_lite (matching Phase 7A baselines), difference rewards produce a **16% regression** from the IQL baseline (0.133 vs 0.159). The algorithm confound accounted for roughly half the originally observed gap. The remaining deficit is attributable to the reward formula's competitive incentive structure.

**Key finding:** The difference reward formula `R_i = 2*d_i - mean` has a negative gradient on teammate performance (`dR_i/d(d_j) = -1/N`), creating intra-team competition rather than cooperation. Despite this, rainbow_lite's sample efficiency partially compensates, recovering joint beat-base from 0.087 (vanilla) to 0.133 --- but not enough to match IQL at 0.159.

**Verdict:** Difference rewards as implemented do not improve coordination at N=4. The competitive reward signal degrades joint performance even with the correct algorithm. Proceed to QMIX (Phase 10A-ii), which guarantees cooperative incentives by architectural construction.

---

## 2. Results: Rainbow_lite Redux (Corrected)

### 2.1 Joint Beat-Base Rate (Primary Metric)

| Stochasticity | Seed 101 | Seed 202 | Seed 303 | Mean |
|:---:|:---:|:---:|:---:|:---:|
| s0 | 0.127 | 0.147 | 0.087 | **0.120** |
| s1 | 0.180 | 0.127 | 0.133 | **0.147** |
| s2 | 0.167 | 0.113 | 0.107 | **0.129** |
| **Grand mean** | | | | **0.133** |

**Comparison across all conditions:**

| Condition | Algorithm | Joint Beat-Base | vs IQL |
|-----------|:---------:|:---:|:---:|
| Phase 10A-i Original | vanilla | 0.087 | -45% |
| Phase 10A-i Replication | vanilla | 0.081 | -49% |
| **Phase 10A-i Redux** | **rainbow_lite** | **0.133** | **-16%** |
| Phase 7A IQL (alpha=0.0) | rainbow_lite | 0.159 | baseline |
| Phase 7A RS (alpha=0.75) | rainbow_lite | 0.167 | +5% |
| Phase 8A Curriculum | rainbow_lite | 0.184 | +16% |

**Algorithm vs formula decomposition:**
- IQL baseline (rainbow_lite, alpha=0): 0.159
- Difference rewards + rainbow_lite: 0.133 (formula effect: **-16%**)
- Difference rewards + vanilla: 0.084 (formula + algorithm: **-47%**)
- Algorithm effect alone: accounts for ~31 percentage points of the 47% total regression

### 2.2 Individual Beat-Base Rates

| Agent | Redux (rainbow_lite) | Phase 7A IQL | Change |
|:-----:|:---:|:---:|:---:|
| A1 | 0.522 | 0.582 | -10.3% |
| A2 | 0.571 | 0.533 | +7.1% |
| A3 | 0.621 | 0.571 | +8.8% |

A1 degrades while A2 and A3 improve. Notably, **A3 (team_b) has the highest individual beat-base at 0.621**, suggesting the competitive reward signal between team_a agents (A1 and A2) hurts their coordination while leaving the cross-team A3 relatively unaffected.

### 2.3 Objective Score (A1 vs A2 Win Rate)

| Condition | Mean | Std |
|-----------|:---:|:---:|
| 10A-i Redux (rainbow_lite) | 0.486 | 0.050 |
| 10A-i Original (vanilla) | 0.426 | 0.046 |
| Phase 7A IQL | 0.520 | 0.044 |
| Phase 7A RS | 0.516 | 0.059 |
| Phase 8A Curriculum | 0.505 | 0.056 |

Rainbow_lite partially corrects the A2 dominance seen in vanilla (0.426 -> 0.486), though A2 still has a slight edge. PER's ability to upweight rare successful transitions helps the weaker agent learn more effectively.

### 2.4 Average Positions

| Agent | Redux (rainbow_lite) | Original (vanilla) | Phase 7A IQL |
|:-----:|:---:|:---:|:---:|
| A1 | 2.53 | 2.63 | 2.40 |
| A2 | 2.42 | 2.33 | 2.52 |
| Base | 2.72 | 2.53 | 2.69 |

Critical improvement: **Base returns to 2.72** (even slightly worse than Phase 7A's 2.69). Rainbow_lite + difference rewards are at least as effective as IQL at pushing Base down the field. The joint beat-base deficit comes from intra-team coordination, not from failing to beat Base individually.

### 2.5 Strategy Differentiation

| Stochasticity | Risk Diff | Zone Diff | Interpretation |
|:---:|:---:|:---:|:---:|
| s0 | 0.137 | 0.107 | low-moderate |
| s1 | 0.152 | 0.285 | moderate-high |
| s2 | 0.109 | 0.167 | moderate |

Zone differentiation peaks at s1, with two trials showing "high" differentiation (0.331, 0.432). This suggests s1 noise pushes agents into zone-specialization niches.

### 2.6 Non-Stationarity Signals

| Trial | Early A1 | Late A1 | Drift | Interpretation |
|-------|:---:|:---:|:---:|:---|
| s0/101 | 0.58 | 0.46 | -0.12 | A2 strengthening |
| s0/202 | 0.48 | 0.68 | +0.20 | A1 strengthening |
| s0/303 | 0.40 | 0.46 | +0.06 | A1 strengthening |
| s1/101 | 0.44 | 0.58 | +0.14 | A1 strengthening |
| s1/202 | 0.60 | 0.38 | -0.22 | A2 strengthening |
| s1/303 | 0.58 | 0.44 | -0.14 | A2 strengthening |
| s2/101 | 0.38 | 0.50 | +0.12 | A1 strengthening |
| s2/202 | 0.40 | 0.44 | +0.04 | Stable |
| s2/303 | 0.48 | 0.46 | -0.02 | Stable |

4/9 A1 strengthening, 3/9 A2 strengthening, 2/9 stable. Much more balanced than the vanilla runs (which showed 7/9 A2 strengthening). Rainbow_lite's PER reduces the winner-take-most lock-in by ensuring both agents learn from high-value transitions.

---

## 3. Algorithm Confound: Resolved

### 3.1 Decomposition of Effects

The three experiment batches allow clean decomposition:

```
IQL baseline (rainbow_lite):             0.159
  - Formula effect (difference reward):  -0.026  (-16%)
  = Difference + rainbow_lite:           0.133

IQL baseline (rainbow_lite):             0.159
  - Formula effect:                      -0.026  (-16%)
  - Algorithm effect (vanilla):          -0.049  (-31%)
  = Difference + vanilla:                0.084
```

The algorithm effect (-31%) is nearly **twice** the formula effect (-16%). This confirms:
1. Rainbow_lite's PER and n-step returns are critical at N=4 where coordination events are rare
2. The difference reward formula does hurt, but less severely than losing rainbow_lite

### 3.2 Vanilla Replication Value

The accidental vanilla replication (two independent runs at 0.087 and 0.081) provides strong evidence that vanilla + difference is reliably ~0.08. Cross-run variance of <10% confirms reproducibility.

---

## 4. Root Cause Analysis: The Reward Formula

### 4.1 What Was Implemented

```python
# simulator.py lines 1170-1177
if self.reward_mode == "difference":
    all_dqn = [driver] + other_dqn
    all_deltas = [float(int(d.starting_position) - int(d.position)) for d in all_dqn]
    team_mean = sum(all_deltas) / len(all_deltas)
    marginal = own_delta - team_mean
    return own_delta + marginal
```

For agent i with position-change delta d_i in a team of N agents:

```
R_i = d_i + (d_i - mean(d_1, ..., d_N))
    = 2*d_i - (1/N) * sum(d_j)
```

For N=3: `R_1 = (5*d_1 - d_2 - d_3) / 3`

### 4.2 Mathematical Properties

**Property 1: Zero-sum redistribution.**
`sum(R_i) = sum(d_i)`. The total reward pie equals IQL; difference rewards only redistribute.

**Property 2: Negative teammate gradient.**
`dR_i / d(d_j) = -1/N` for j != i. Agents are penalized when teammates improve.

**Property 3: Own-performance amplification.**
`dR_i / d(d_i) = (2N-1)/N`. For N=3: 5/3 = 1.67x amplification.

### 4.3 Comparison with Wolpert & Tumer (2002)

The literature defines: `D_i = G(z) - G(z_{-i})`

The implementation uses a mean-field approximation that loses the key property: true difference rewards are always non-negative when agent i contributes positively, regardless of teammates. The implemented version penalizes below-average agents even when they contribute positively to the team.

### 4.4 Why Rainbow_lite Partially Compensates

PER upweights high-TD-error transitions, which correspond to rare coordination successes. This partially counteracts the competitive signal by ensuring both agents learn from the same successful coordination events. The winner-take-most dynamic is dampened (4/9 balanced non-stationarity vs vanilla's 7/9 skewed).

---

## 5. Stochasticity Effects

| Level | Joint Beat-Base | Interpretation |
|:---:|:---:|:---|
| s0 | 0.120 | Deterministic: formula effect most visible |
| s1 | 0.147 | Best performance: noise aids exploration |
| s2 | 0.129 | Moderate noise: some recovery |

s1 produces the best result (0.147, with one trial hitting 0.180 --- matching Phase 8A curriculum). This suggests small noise helps agents escape local optima created by the competitive reward signal.

---

## 6. Zone-Level Behavioral Analysis

### 6.1 La Source (Zone 1, Difficulty 0.2)

| Agent | Attempt Rate | Success Rate |
|:-----:|:---:|:---:|
| A1 (redux rainbow) | 0.83 | 0.85 |
| A2 (redux rainbow) | 0.90 | 0.88 |
| A1 (Phase 7A IQL) | 0.86 | 0.83 |

Rainbow_lite recovers La Source exploitation (0.83-0.90 attempt rates) close to Phase 7A levels (0.86). The vanilla run showed depressed attempt rates (0.60) for A1.

### 6.2 High-Attempt Agents in s1 Trials

A notable pattern in s1/s202 and s1/s303: A1 develops extremely aggressive zone coverage (attempt rates 0.70-1.00 across ALL zones). This "hyper-aggressive" strategy achieves more overtake attempts (~1470-1480) than any other condition. Despite low per-zone success rates, the sheer volume produces acceptable individual performance.

---

## 7. Summary of Findings

### Quantitative Conclusions

1. **Difference rewards (rainbow_lite) produce 0.133 joint beat-base** --- a 16% regression from IQL (0.159)
2. **Algorithm choice matters more than reward formula**: vanilla -> rainbow_lite accounts for 31% vs the formula's 16%
3. **The formula's negative teammate gradient is the structural cause** of the remaining deficit
4. **s1 stochasticity partially compensates** (0.147 mean, peak trial at 0.180)
5. **A3 (cross-team) benefits** from intra-team A1/A2 competition (0.621 beat-base vs 0.571 baseline)

### Phase 10A-i Position in the Research Arc

| Phase | Method | Joint Beat-Base | Credit Quality |
|:-----:|--------|:---:|:---:|
| 7A | IQL (alpha=0.0) | 0.159 | None (individual only) |
| 7A | RS (alpha=0.75) | 0.167 | Flat sharing |
| 8A | Curriculum (0->0.75) | 0.184 | Scheduled sharing |
| **10A-i** | **Difference rewards** | **0.133** | **Competitive marginal** |

Difference rewards rank **last** among all tested credit assignment methods. The formula creates competitive rather than cooperative incentives, making it worse than no credit assignment (IQL) and far worse than alpha-blending approaches.

---

## 8. Next Actions

### 8.1 Proceed to QMIX (Phase 10A-ii)

The difference reward experiments provide the scientific justification for architectural CTDE:

- **Reward shaping alone cannot solve credit dilution.** Even the simplest Wolpert-Tumer approximation creates competitive incentives in our multi-agent racing domain.
- **QMIX's monotonicity constraint** (`dQ_tot/dQ_i >= 0`) guarantees cooperative incentives by construction.
- **State-dependent mixing weights** learn situation-appropriate credit attribution --- something no fixed reward formula can do.

### 8.2 QMIX Implementation Plan

| Step | Task | Detail |
|:----:|------|--------|
| 1 | Implement MixingNetwork | Hypernetwork-conditioned monotonic mixing (~150-200 lines) |
| 2 | Implement global state extraction | Concatenate all agents' observations for mixing network input |
| 3 | Modify training loop | Centralized TD loss through mixing network |
| 4 | Add `reward_mode=qmix` path | Wire mixing network into evaluate_marl.py |
| 5 | Run 9-trial matrix | 3 stochasticity x 3 seeds, matching prior phases |
| 6 | Compare vs all baselines | 7A, 8A, 10A-i in unified comparison |

### 8.3 If Revisiting Difference Rewards (Low Priority)

Two corrected formulas for completeness:

**Option A: Leave-one-out counterfactual (faithful Wolpert-Tumer)**
```python
team_total = sum(all_deltas)
team_without_i = team_total - own_delta
D_i = team_total / N - team_without_i / (N-1)
```

**Option B: Cooperative floor (hybrid)**
```python
marginal = own_delta - team_mean
return own_delta + max(0, marginal)  # No penalty for below-average
```

---

## Appendix A: Full Results Table (Rainbow_lite Redux)

| Trial | Stoch | Seed | Joint Beat-Base | A1 Beat | A2 Beat | A3 Beat | A1 Win Rate | A1 Pos | A2 Pos | Base Pos | Risk Diff | Zone Diff |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | s0 | 101 | 0.127 | 0.547 | 0.567 | 0.607 | 0.507 | 2.46 | 2.46 | 2.72 | 0.111 | 0.123 |
| 2 | s0 | 202 | 0.147 | 0.593 | 0.527 | 0.647 | 0.600 | 2.27 | 2.65 | 2.77 | 0.170 | 0.070 |
| 3 | s0 | 303 | 0.087 | 0.487 | 0.647 | 0.600 | 0.427 | 2.66 | 2.24 | 2.73 | 0.129 | 0.127 |
| 4 | s1 | 101 | 0.180 | 0.587 | 0.607 | 0.647 | 0.513 | 2.39 | 2.35 | 2.84 | 0.122 | 0.092 |
| 5 | s1 | 202 | 0.127 | 0.507 | 0.580 | 0.667 | 0.493 | 2.57 | 2.45 | 2.75 | 0.167 | 0.331 |
| 6 | s1 | 303 | 0.133 | 0.473 | 0.487 | 0.740 | 0.493 | 2.63 | 2.56 | 2.70 | 0.166 | 0.432 |
| 7 | s2 | 101 | 0.167 | 0.493 | 0.527 | 0.547 | 0.460 | 2.60 | 2.41 | 2.57 | 0.199 | 0.219 |
| 8 | s2 | 202 | 0.113 | 0.527 | 0.580 | 0.573 | 0.447 | 2.60 | 2.35 | 2.68 | 0.062 | 0.077 |
| 9 | s2 | 303 | 0.107 | 0.480 | 0.613 | 0.580 | 0.433 | 2.65 | 2.30 | 2.67 | 0.066 | 0.077 |

## Appendix B: Vanilla Replication Results (Two Independent Runs)

| Batch | Trials | Grand Mean Joint Beat-Base | A1 Win Rate | Interpretation |
|-------|:---:|:---:|:---:|:---|
| Original | 9 | 0.087 | 0.426 | Strong A2 dominance (7/9) |
| Replication | 9 | 0.081 | 0.495 | More balanced, A1 slightly favored (5/9) |

Confirms vanilla + difference reward result is reproducible at ~0.08 regardless of seed.

## Appendix C: Cross-Phase Algorithm x Reward Interaction

```
                     vanilla    rainbow_lite    Delta
IQL (alpha=0):        --          0.159          --
Difference reward:    0.084       0.133        +58%
                      -----       -----
Formula effect:        --         -16%           --
Algorithm effect:               +58% (from vanilla to rainbow)
```

The algorithm x reward interaction is subadditive: rainbow_lite helps more under difference rewards (+58%) than the baseline algorithm gap (~20-30% in single-agent Phase 2). This suggests PER specifically compensates for the competitive reward signal by ensuring both agents learn from coordination successes.
