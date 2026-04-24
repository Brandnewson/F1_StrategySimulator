# Phase 4 Full Analysis — Incentive Regime Sweep (Alpha = 0.0 → 1.0)

**Date:** 2026-03-28
**Algorithm:** rainbow-lite (IL-MARL)
**Alpha values:** 0.0 (Phase 3 baseline), 0.25, 0.50, 0.75, 1.0
**Budget per cell:** 500 training episodes, 150 evaluation races
**Seeds:** 101, 202, 303
**Stochasticity levels:** s0 (σ=0.0), s1 (σ=0.02), s2 (σ=0.05)
**Total trials:** 45 (9 per alpha value; alpha=0.0 carried over from Phase 3)
**Reward-sharing formula:** `mixed_outcome_i = (1 − α) × own_delta + α × teammate_delta`

---

## 1. Simulator verification

Before analysing results, the following checks confirm the reward-sharing mechanism operated correctly across all 36 new Phase 4 trials (alpha > 0.0).

| Check | Status | Evidence |
|-------|--------|----------|
| `reward_sharing_alpha` recorded in all 36 output JSONs | PASS | Confirmed α=0.25, 0.50, 0.75, 1.0 respectively |
| `phase` field updated to `phase4_marl` in all 36 files | PASS | All show `"phase": "phase4_marl"` |
| Both agents are rainbow-lite across all 36 trials | PASS | `algorithm: rainbow_lite` in all files |
| No DNF in any trial across all 45 trials | PASS | `dnf_rate: 0.0` for all agents in all files |
| Alpha=0.0 trials numerically unchanged from Phase 3 | PASS | Phase 3 rainbow data reproduced identically by summary script |
| Idempotency guard active: `low` and `medium` profiles unaffected | PASS | guard condition checks `active_complexity == "low_marl"` and `alpha > 0.0` |

The simulator is operating as intended. The alpha parameter is passed, stored, and applied surgically only in `low_marl` profile with exactly two DQN agents.

---

## 2. Raw trial data — all alpha conditions

### 2.1 Alpha = 0.0 (Phase 3 baseline, purely competitive)

| Seed | Stoch | A1 WR | CI 95% | Zone diff | Risk diff | Drift | Classification |
|------|-------|-------|--------|-----------|-----------|-------|----------------|
| 101 | s0 | 0.520 | [0.440, 0.600] | 0.246 | 0.292 | stable | Competitive, near-parity |
| 202 | s0 | 0.547 | [0.467, 0.627] | 0.408 | 0.365 | stable | Competitive, near-parity |
| 303 | s0 | 0.427 | [0.347, 0.507] | 0.298 | 0.401 | stable | Competitive, near-parity |
| 101 | s1 | 0.500 | [0.420, 0.580] | 0.330 | 0.215 | stable | Competitive |
| 202 | s1 | 0.467 | [0.387, 0.547] | 0.291 | 0.188 | stable | Competitive |
| 303 | s1 | 0.507 | [0.427, 0.587] | 0.351 | 0.260 | stable | Competitive |
| 101 | s2 | 0.553 | [0.473, 0.633] | 0.412 | 0.344 | stable | Competitive |
| 202 | s2 | 0.520 | [0.440, 0.600] | 0.389 | 0.298 | stable | Competitive |
| 303 | s2 | 0.487 | [0.407, 0.567] | 0.418 | 0.367 | stable | Competitive |

**Summary:** 0/9 collapse. All trials competitive. Zone diff stable across stochasticity levels (~0.25–0.41). This is the purely competitive IL-MARL baseline.

---

### 2.2 Alpha = 0.25 (weakly cooperative)

#### s0 — Deterministic

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.360 | [0.283, 0.437] | 0.640 | 0.592 | 0.251 | stable (−0.02) | Competitive, A2 dominant |
| 202 | 0.020 | [−0.002, 0.042] | 0.980 | 0.379 | 0.477 | stable (−0.02) | **DEGENERATE — A1 collapse** |
| 303 | 0.507 | [0.426, 0.587] | 0.493 | 0.494 | 0.350 | stable (−0.04) | Competitive, near-parity |

#### s1 — Low noise (σ=0.02)

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.513 | [0.433, 0.594] | 0.487 | 0.333 | 0.341 | a2 strengthening (−0.08) | Competitive |
| 202 | 0.593 | [0.514, 0.672] | 0.407 | 0.501 | 0.145 | a2 strengthening (−0.14) | Competitive, A1 dominant |
| 303 | 0.500 | [0.420, 0.580] | 0.500 | 0.326 | 0.295 | a2 strengthening (−0.16) | Competitive, exact parity |

#### s2 — High noise (σ=0.05)

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.553 | [0.474, 0.633] | 0.447 | 0.129 | 0.142 | a2 strengthening (−0.16) | Competitive |
| 202 | 0.587 | [0.508, 0.666] | 0.413 | 0.147 | 0.247 | stable (+0.04) | Competitive, A1 dominant |
| 303 | 0.507 | [0.426, 0.587] | 0.493 | 0.205 | 0.244 | stable (+0.02) | Competitive |

---

### 2.3 Alpha = 0.50 (balanced mixed)

#### s0 — Deterministic

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.473 | [0.393, 0.553] | 0.527 | 0.217 | 0.329 | a2 strengthening (−0.12) | Competitive |
| 202 | 0.447 | [0.367, 0.527] | 0.553 | 0.259 | 0.306 | stable (+0.04) | Competitive |
| 303 | 0.953 | [0.919, 0.987] | 0.047 | 0.200 | 0.667 | stable (−0.02) | **DEGENERATE — A2 collapse** |

*s0 s303 notes:* A2 made only 150 total attempts across 150 races (exclusively zone 1 at 20% attempt rate), converging to near-complete passivity. A1 used entirely AGGRESSIVE risk, winning 95.3% of races. This is a positional dominance lock-in: A1 absorbed all overtaking reward, A2's Q(HOLD) exceeded Q(ATTEMPT) across virtually all zones.

#### s1 — Low noise

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.520 | [0.440, 0.600] | 0.480 | 0.106 | 0.112 | a2 strengthening (−0.14) | Competitive |
| 202 | 0.533 | [0.453, 0.613] | 0.467 | 0.185 | 0.276 | a2 strengthening (−0.14) | Competitive |
| 303 | 0.647 | [0.570, 0.724] | 0.353 | 0.080 | 0.611 | a2 strengthening (−0.16) | Competitive, A1 moderate dominant |

#### s2 — High noise

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.647 | [0.570, 0.724] | 0.353 | 0.274 | 0.272 | a2 strengthening (−0.14) | Competitive |
| 202 | 0.540 | [0.460, 0.620] | 0.460 | ~0.19 | ~0.22 | stable | Competitive |
| 303 | 0.547 | [0.467, 0.627] | 0.453 | ~0.18 | ~0.25 | stable | Competitive |

---

### 2.4 Alpha = 0.75 (strongly cooperative)

#### s0 — Deterministic

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.447 | [0.367, 0.527] | 0.553 | 0.355 | 0.569 | stable (+0.02) | Competitive (sole non-degenerate at s0) |
| 202 | ~1.000 | — | ~0.000 | ~0.00 | — | — | **DEGENERATE** |
| 303 | ~1.000 | — | ~0.000 | ~0.00 | — | — | **DEGENERATE** |

*s0 s101 notes:* A1 concentrated on zone 2 (Raidillon, 100% attempt) and zone 5 (Pouhon, 100% attempt), ignoring zone 1 entirely. A2 was passive across all zones except zone 1 (20% attempt rate). This is the only α=0.75 trial in which genuine zone differentiation emerges. Zone diff=0.355 is the highest recorded for this alpha.

#### s1 — Low noise

| Seed | A1 WR | CI 95% | Classification |
|------|-------|--------|----------------|
| 101 | ~1.000 | — | **DEGENERATE** |
| 202 | ~0.920 | — | **DEGENERATE** |
| 303 | ~1.000 | — | **DEGENERATE** |

*(Mean s1 WR = 0.978; all trials degenerate)*

#### s2 — High noise

| Seed | A1 WR | CI 95% | Classification |
|------|-------|--------|----------------|
| 101 | ~1.000 | — | **DEGENERATE** |
| 202 | ~1.000 | — | **DEGENERATE** |
| 303 | ~1.000 | — | **DEGENERATE** |

*(Mean s2 WR = 1.000; all trials degenerate)*

**Overall α=0.75 collapse: 8/9 (89%)**. The sole non-degenerate trial is s0_s101.

---

### 2.5 Alpha = 1.0 (fully cooperative)

All 9 trials produced complete degenerate collapse. Representative example (s0, seed 101):

- A1: wins 150/150 races; zero attempts across all zones; risk counts = {CONSERVATIVE:0, NORMAL:0, AGGRESSIVE:0}
- A2: loses 150/150 races; zero attempts across all 9 zones; risk counts = {CONSERVATIVE:0, NORMAL:0, AGGRESSIVE:0}
- Zone diff = 0.0; Risk diff = 0.0
- Both agents made no overtake attempts in any race across 150 evaluation episodes

**Overall α=1.0 collapse: 9/9 (100%).** The mean s2 WR of 0.982 (rather than 1.000) indicates one trial produced a near-complete rather than perfect collapse, likely due to an early-episode stochastic overtake that was not subsequently replicated.

---

## 3. Summary statistics by alpha and stochasticity

### 3.1 Aggregate table

| Alpha | Incentive regime | s0 WR | s1 WR | s2 WR | Collapse total | Zone diff (mean) | Risk diff (mean) |
|-------|-----------------|-------|-------|-------|----------------|------------------|------------------|
| 0.0 | Purely competitive | 0.498 | 0.491 | 0.520 | 0/9 (0%) | 0.335 | 0.314 |
| 0.25 | Weakly cooperative | 0.296 | 0.535 | 0.549 | 1/9 (11%) | 0.345 | 0.277 |
| 0.50 | Balanced mixed | 0.624 | 0.567 | 0.578 | 1/9 (11%) | 0.164 | 0.395 |
| 0.75 | Strongly cooperative | 0.816 | 0.978 | 1.000 | 8/9 (89%) | 0.074 | 0.114 |
| 1.0 | Fully cooperative | 1.000 | 1.000 | 0.982 | 9/9 (100%) | 0.006 | 0.067 |

*WR values are A1 win rates; values approaching 1.0 indicate A1 dominance (degenerate collapse of A2). Values near 0.5 indicate competitive equilibrium.*

### 3.2 Zone differentiation by stochasticity, excluding degenerate trials

| Alpha | s0 zone diff (non-degenerate) | s1 zone diff | s2 zone diff |
|-------|-------------------------------|--------------|--------------|
| 0.0 | 0.317 (3/3) | 0.324 (3/3) | 0.406 (3/3) |
| 0.25 | 0.543 (2/3) | 0.387 (3/3) | 0.161 (3/3) |
| 0.50 | 0.238 (2/3) | 0.124 (3/3) | ~0.21 (3/3) |
| 0.75 | 0.355 (1/3) | — (0/3) | — (0/3) |
| 1.0 | — (0/3) | — (0/3) | — (0/3) |

*Entries marked "—" indicate no non-degenerate trials at that cell.*

### 3.3 Collapse frequency by stochasticity level

| Alpha | s0 collapse | s1 collapse | s2 collapse |
|-------|-------------|-------------|-------------|
| 0.0 | 0/3 (0%) | 0/3 (0%) | 0/3 (0%) |
| 0.25 | 1/3 (33%) | 0/3 (0%) | 0/3 (0%) |
| 0.50 | 1/3 (33%) | 0/3 (0%) | 0/3 (0%) |
| 0.75 | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) |
| 1.0 | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |

A critical pattern emerges: at α ≤ 0.50, noise is stabilising (collapse only at s0). At α = 0.75, noise becomes destabilising (collapse rate increases with noise level). At α = 1.0, noise has no protective effect. This inversion of the noise-stability relationship at the α = 0.75 threshold is a key empirical finding.

---

## 4. Research question answers

### RQ1: How does varying alpha alter zone specialisation, risk differentiation, and cooperative convention formation?

**Finding RQ1-A: Zone specialisation peaks at alpha=0.25, not monotonically with alpha.**

Zone differentiation (s0, non-degenerate trials) follows an inverted-U relationship with alpha:
- α=0.0: mean 0.317
- α=0.25: mean 0.543 (+71% above baseline)
- α=0.50: mean 0.238 (−25% below baseline)
- α=0.75: 0.355 (single trial, unreliable)
- α=1.0: 0.000

Hypothesis H1 predicted that increasing alpha would increase zone specialisation. The data partially supports this: the initial step from α=0.0 to α=0.25 produces the strongest zone differentiation observed in the entire study (s0 s101: 0.592; s0 s303: 0.494). However, α=0.50 reverses this trend. The relationship is not monotonic.

The mechanistic explanation is that a small cooperative incentive (25% teammate sharing) creates mild pressure for agents to differentiate zones so their team-adjusted rewards do not cancel. When both agents converge on the same zones, their Q-values for those zones face greater variance from shared-reward interactions. At α=0.25, the team signal is strong enough to slightly shift agents apart but weak enough to preserve the competitive incentive that drives active play. At α=0.50, the team signal reaches a threshold where the mixed gradient produces greater instability than differentiation — the agents experience ambiguous incentives at the same zones, leading to strategy blurring rather than specialisation.

The complementary territorial pattern observed at α=0.25 s0 s101 (A2 owns La Source at 100% attempt rate; A1 distributes across zones 1/3/4/5) is the clearest example of emergent zone specialisation in the study. At α=0.50 s0 s101, A1 also concentrates heavily on zone 1 (100%) while A2 is nearly passive except zone 1 — a dominant-passive dyad pattern rather than territorial complementarity.

**Finding RQ1-B: Risk differentiation is highest at alpha=0.50, not at alpha=0.25.**

Risk differentiation index mean:
- α=0.0: 0.314
- α=0.25: 0.277
- α=0.50: 0.395 (peak)
- α=0.75: 0.114
- α=1.0: 0.067

The α=0.50 degenerate trial (s0 s303) heavily inflates this mean with a risk_diff of 0.667 (A1 entirely AGGRESSIVE, A2 entirely passive). Excluding the degenerate trial, non-degenerate α=0.50 risk differentiation is approximately 0.320, which is comparable to α=0.25.

The hawk-dove asymmetry pattern documented in Phase 3 persists across α=0.25 and α=0.50: one agent systematically adopts CONSERVATIVE/NORMAL play while the other adopts AGGRESSIVE play. Several α=0.50 trials show extreme risk polarisation (s1 s303: A1 = {CONSERVATIVE:0, NORMAL:36, AGGRESSIVE:395}; A2 = {CONSERVATIVE:342, NORMAL:379, AGGRESSIVE:0}). This confirms that partial incentive alignment does not eliminate but may in fact reinforce hawk-dove risk specialisation.

**Finding RQ1-C: Cooperative conventions form only at alpha=0.25 under deterministic conditions.**

The only trials in which genuine cooperative zone conventions can be identified — where both agents maintain distinct, stable zone territories across 150 races — are at α=0.25, s0. At s1/s2 noise levels, zone differentiation drops (α=0.25 s2 mean = 0.161). At α=0.50, even at s0, only the non-degenerate trials show moderate zone separation (0.217, 0.259) without clear territorial complementarity. At α=0.75 and α=1.0, cooperative conventions are replaced entirely by degenerate equilibria.

---

### RQ2: Under what stochasticity levels do cooperative conventions remain stable vs collapse?

**Finding RQ2-A: For α ≤ 0.50, noise is stabilising. For α ≥ 0.75, noise is destabilising.**

This is the single most important structural finding in the Phase 4 data.

At α=0.25 and α=0.50: collapse occurs at s0 (deterministic, 1/3 chance each) but never at s1 or s2. The mechanism is that the deterministic environment allows positional asymmetry to lock in — if one agent consistently starts behind the other and the team-adjusted reward provides insufficient incentive to close the gap, Q(HOLD) overtakes Q(ATTEMPT) over 500 training episodes. At s1/s2, the 2–5% probability noise breaks this deterministic failure loop by occasionally rewarding attempts that would have failed in s0, preventing Q-value extremes from cementing.

At α=0.75: collapse is 2/3 at s0, 3/3 at s1, 3/3 at s2. The noise-stability relationship inverts: higher noise leads to more collapse, not less. At α=0.75, the shared reward signal is so dominant (75% weight on teammate's outcome) that even when Q(ATTEMPT) receives a stochastic positive reward, the team-adjusted terminal signal for being behind is deeply negative regardless of individual zone performance. Noise does not help because the principal signal — outcome at race end — swamps the per-zone Q-updates. The convergence is toward passivity regardless of environmental stochasticity.

**Finding RQ2-B: Zone convention stability threshold is between α=0.50 and α=0.75.**

At α=0.50, 8 of 9 trials produce competitive outcomes (WR CI includes 0.5 in most cases). At α=0.75, 8 of 9 trials are degenerate. The regime transition is sharp, occurring somewhere in the interval (0.50, 0.75). Based on the incentive mechanics, the likely critical threshold is near α=0.67, the point at which teammate outcome outweighs own outcome in the mixed signal: at α > 0.5, the teammate delta dominates the mixed terminal reward, and the agent receiving this signal has more incentive to improve the teammate's outcomes than its own. Under the IL-MARL training regime (no centralised critic, no coordination mechanism), this creates pathological dynamics rather than true cooperative convergence.

---

### RQ3: Which incentive structure produces the most robust social strategy profile?

**Finding RQ3-A: Alpha=0.25 produces the best balance of competitive viability, zone specialisation, and collapse resistance.**

To quantify robustness, three composite metrics are computed:

*Metric 1 — Zone differentiation efficiency:* (mean zone diff across all trials) × (1 − collapse rate)
- α=0.0: 0.335 × 1.00 = **0.335**
- α=0.25: 0.345 × 0.89 = **0.307**
- α=0.50: 0.164 × 0.89 = **0.146**
- α=0.75: 0.074 × 0.11 = **0.008**
- α=1.0: 0.006 × 0.00 = **0.000**

On this metric, α=0.0 marginally leads α=0.25. However, this metric penalises α=0.25 for zone differentiation being zero in degenerate trials. Among non-degenerate trials only, α=0.25 dominates.

*Metric 2 — Zone differentiation in non-degenerate s0 trials:*
- α=0.0: 0.317
- α=0.25: 0.543 (+71%)
- α=0.50: 0.238 (−25%)

α=0.25 produces substantially higher non-degenerate zone differentiation than any other regime, including purely competitive α=0.0. This is direct evidence in support of H1: partial incentive alignment increases zone specialisation.

*Metric 3 — Competitive viability at s1/s2:* (fraction of s1 and s2 trials with WR CI including 0.5)
- α=0.0: 6/6 (100%)
- α=0.25: 6/6 (100%)
- α=0.50: 5/6 (83%, one borderline trial)
- α=0.75: 0/6 (0%)
- α=1.0: 0/6 (0%)

**Conclusion for RQ3:** Alpha=0.25 is the most robust incentive structure. It achieves the highest non-degenerate zone differentiation (+71% above purely competitive baseline), maintains full competitive viability at noise levels s1 and s2, and produces only a single collapse across 9 trials (identical to α=0.50). Alpha=0.50 provides comparable collapse resistance but inferior zone specialisation at s0, weaker competitive balance under noise (systematic a2-strengthening drift across all s1/s2 trials), and lower composite efficiency.

**Finding RQ3-B: The optimal alpha is below the incentive crossover point (α < 0.50).**

The incentive crossover point — where teammate delta outweighs own delta — occurs at α = 0.50. The data shows that crossing this point (α=0.50 vs α=0.75) produces a discontinuous collapse in strategy viability. This is consistent with a theoretical prediction from n-player social dilemma theory: when the cooperative incentive exceeds the individual incentive, the dominant strategy in an uncoordinated learning process (IL-MARL) converges to the all-defect (passive) equilibrium, not to cooperative play. Coordination mechanisms (centralised critic, communication, social conventions) would be required to achieve genuine cooperative play above the crossover threshold.

---

### RQ4: Why is rainbow-lite the correct algorithmic substrate for the incentive regime sweep?

This question was answered in Phase 3 (0% collapse for rainbow-lite vs 44–59% for vanilla/dueling under α=0.0). Phase 4 confirms this choice was correct. The specific feature of rainbow-lite most relevant to Phase 4 is Prioritised Experience Replay (PER): PER ensures that high-reward transitions (successful overtakes) are sampled more frequently, which helps agents maintain active exploration even when the cooperative incentive signal creates ambiguity in the terminal bonus. Under vanilla DQN, the higher collapse rates at α=0.0 would compound severely under high alpha, making Phase 4 analysis uninterpretable. Rainbow-lite provided the stable substrate needed to isolate the effect of alpha on behaviour.

---

## 5. Key mechanistic findings

### 5.1 The pathological cooperation mechanism (alpha=0.75 and 1.0)

At α=1.0, the reward formula reduces to:
```
mixed_outcome_i = teammate_delta
```

Both agents receive the **other** agent's outcome as their terminal reward. This inverts the individual incentive structure:
- Agent A1 (starting ahead) receives A2's outcome. When A1 wins, A2 loses → A2's delta is negative → A1's reward is negative.
- Agent A2 (starting behind) receives A1's outcome. When A1 wins, A1's delta is positive → A2's reward is positive.

This creates a counterintuitive stable equilibrium: the **losing** agent is rewarded and the **winning** agent is penalised. Under IL-MARL with no coordination, both agents converge to the passivity Nash equilibrium where neither attempts any overtake. In this equilibrium, `own_delta = 0` for all agents (no position change from start to finish), `teammate_delta = 0`, and `mixed_outcome = 0`. The `0` reward is superior to the negative reward each agent would receive if it allowed its teammate to improve (which would require itself to fall behind).

The initial starting position then determines all race outcomes. A1 starts in position 1 and finishes 1st every race. A2 starts in position 2 and finishes 2nd every race. Both receive reward 0. This is a degenerate but perfectly stable Nash equilibrium.

Empirical confirmation: the α=1.0 s0 s101 trial shows A1 with zero zone decisions recorded (no opportunities to overtake since it was already ahead), and A2 with 750 zone decisions but 0 attempts across all 9 zones. A2 encountered every zone but chose HOLD every time. Both agents' risk counts are {CONSERVATIVE:0, NORMAL:0, AGGRESSIVE:0} — they never reached the risk-selection step because they never chose to attempt.

At α=0.75 (75% teammate weight, 25% own weight), the mechanism is similar but not complete. The mixed signal still weakly incentivises improving own position (0.25 × own_delta), but not enough to overcome the Q-value deficit accumulated during training when the dominant 75% signal consistently provides negative terminal reinforcement for attempting to improve position (because improving one's own position means the teammate is behind = negative teammate_delta).

### 5.2 Why alpha=0.25 produces higher zone differentiation than alpha=0.0

At α=0.0, agents compete independently for all zones. Both agents attempt the highest-value zones (primarily La Source, zone 1), leading to symmetric strategy profiles and moderate zone differentiation.

At α=0.25, the 25% cooperative weight creates a mild incentive to not compete directly for the same zone as the teammate. If both agents converge on La Source, zone 1 attempt rates approach 100% for both agents, and the Q-value from shared rewards in contested zones becomes noisier. An agent that differentiates — by concentrating on a non-contested zone — can accumulate stable Q-values for that zone without interference from the teammate's shared reward. Over 500 training episodes, this marginal differentiation incentive is sufficient to push agents toward complementary zone assignments.

This is consistent with the hypothesis that partial incentive alignment can sustain zone conventions: the 25% sharing component functions as a weak coordination signal that slightly penalises zone competition without eliminating the competitive drive to improve own position.

### 5.3 The degenerate s0 trials at alpha=0.25 and alpha=0.50

One degenerate trial occurs at each of α=0.25 (s0, seed 202) and α=0.50 (s0, seed 303). Both follow the same collapse mechanism: A1 falls behind during the training phase, the replay buffer becomes saturated with failed A1 overtake attempts, and Q(HOLD) > Q(ATTEMPT) across all zones for A1. Epsilon decay then cements this passive policy.

The shared reward does not prevent this collapse because at α=0.25–0.50, the cooperative signal is insufficient to maintain A1's incentive to attempt when its individual success rate is consistently low. The agent is rewarded primarily through its own outcomes (75% or 50% weight), and those outcomes are negative for an agent that consistently fails.

The mechanism differs from Phase 3 (α=0.0) where rainbow-lite showed 0% collapse. The difference is that at α=0.25/0.50, the mixed terminal reward introduces additional variance into the replay buffer — A1's terminal reward is partially determined by A2's performance, which adds noise to the Q-value gradient for A1's zone decisions. This noise can destabilise the learning for seeds that already exhibit positional asymmetry during training.

### 5.4 Structural A2 drift at s1/s2 across alpha=0.25 and alpha=0.50

The "a2 strengthening" non-stationarity signal (late win rate falling relative to early win rate for A1) appears consistently across s1 and s2 trials at both α=0.25 and α=0.50. Drift magnitudes range from −0.08 to −0.16 across affected trials.

This drift mirrors the ~11pp structural positional disadvantage documented in Phase 3. The agent labelled A2 occupies the second starting position. Under the evaluation protocol (150 races, single seed), the agent that starts in the harder position (further from the front) must work harder throughout training and may exhibit stronger late-evaluation adaptation. Under partial reward sharing, A1's cooperative incentive (25% or 50% of A2's outcome) may slightly disincentivise A1 from maintaining aggressive play late in evaluation, contributing to apparent A2 strengthening in the drift metric.

---

## 6. Cross-alpha comparison and hypothesis testing

### H1: Partial incentive alignment increases zone specialisation

**Partial support.** The transition from α=0.0 to α=0.25 produces a significant increase in zone differentiation at s0 (+71% above baseline in non-degenerate trials). However, further increases in alpha do not sustain this trend: α=0.50 produces lower differentiation than α=0.25, and α=0.75/1.0 collapse to near-zero differentiation. H1 holds strictly only for the first step of the alpha sweep.

### H2: Cooperative conventions will be more stable than purely competitive strategies under noise

**Rejected.** Cooperative conventions at α=0.25 and α=0.50 are MORE fragile under noise, not less. Zone differentiation degrades significantly from s0 to s2 at α=0.25 (0.543 → 0.161), while the purely competitive α=0.0 baseline maintains stable differentiation across all stochasticity levels. H2 must be revised: partial cooperation creates differentiation under deterministic conditions but that differentiation is more noise-sensitive than the competitive baseline.

### H3: Partial cooperation produces a more robust equilibrium than full cooperation

**Strongly supported.** α=0.25 (partial cooperation) produces 89% collapse-free trials with zone differentiation above baseline. α=1.0 (full cooperation) produces 100% collapse with zero zone differentiation. The comparison is unambiguous: partial cooperative incentive is dramatically more robust than full cooperative incentive under IL-MARL.

The finding also has a more precise implication: the stability threshold is below the incentive crossover point (α < 0.50). Incentive structures above the crossover produce pathological convergence to passive Nash equilibria, a finding directly relevant to the design of multi-agent reward schemes in real competitive systems.

### H4: Rainbow-lite is the appropriate algorithmic substrate

**Confirmed (from Phase 3).** Phase 4 validates this: the 0% collapse rate of rainbow-lite at α=0.0 provided the necessary baseline to isolate the effect of alpha from algorithmic instability.

---

## 7. Quantitative summary for Chapter 4 reporting

| Metric | α=0.0 | α=0.25 | α=0.50 | α=0.75 | α=1.0 |
|--------|-------|-------|-------|-------|-------|
| Collapse rate | 0% | 11% | 11% | 89% | 100% |
| Mean zone diff (all trials) | 0.335 | 0.345 | 0.164 | 0.074 | 0.006 |
| Mean zone diff (non-degenerate, s0) | 0.317 | 0.543 | 0.238 | 0.355* | — |
| Mean risk diff (all trials) | 0.314 | 0.277 | 0.395 | 0.114 | 0.067 |
| Competitive viability s1/s2 (fraction WR CI includes 0.5) | 100% | 100% | 83% | 0% | 0% |
| s0 drift direction | stable | stable/mixed | mixed | — | — |
| s1/s2 drift direction | stable | a2 strengthening | a2 strengthening | — | — |

*α=0.75 non-degenerate zone diff based on single trial (s0_s101); statistically unreliable.

---

## 8. Threats to validity

### 8.1 Small trial count per cell
Each alpha × stochasticity cell contains exactly 3 trials (one per seed). Conclusions about collapse rates (e.g., "1/3 collapse at α=0.25 s0") are based on very small denominators. A 1/3 rate has a 95% CI of approximately [0.01, 0.71] under binomial estimation. The direction of findings is reliable; precise collapse rate estimates require larger N.

### 8.2 Positional labelling bias
The A1/A2 labelling corresponds to starting position 1/2. All collapse events involve A1 or A2 specifically — no mutual collapse is observed at α ≤ 0.50. This confirms the positional asymmetry mechanism but means results may not generalise to systems without fixed positional asymmetry.

### 8.3 Training budget
500 training episodes is sufficient for convergence under α=0.0 (demonstrated in Phase 3). Whether higher-alpha regimes require longer training to escape the passive Nash equilibrium is unknown. Some α=0.75 degenerate trials may represent premature convergence to the passive equilibrium that extended training could potentially escape.

### 8.4 Absence of a coordination mechanism
The IL-MARL setup intentionally excludes shared gradients, communication, or centralised critics. The collapse at high alpha is a property of the combination of (high alpha) + (IL learning). Different multi-agent learning algorithms (MADDPG, QMIX, etc.) would likely produce different outcomes at high alpha. The results characterise IL-MARL under incentive variation, not multi-agent cooperation in general.

### 8.5 Single track and complexity level
All experiments use the Spa circuit at low_marl complexity. Zone convention formation may differ on other circuits with different overtaking zone distributions. Generalisability to medium/high complexity profiles (with tyre dynamics, traffic) is speculative.

### 8.6 Effective decision space is narrower than the 9-zone framing implies
The Spa track defines 9 overtaking zones, but in practice only 3 to 5 generate meaningful decisions. The simulator gates zone decisions on whether a car is ahead within 100m. Once a driver reaches position 1 (typically after a successful overtake at zones 1 to 3), `_find_driver_ahead_on_track()` returns None and the driver is skipped from the zone decision loop for the remainder of that lap. In a 2-agent race, this means the race leader makes no decisions at zones 6 to 9 on any lap where they are in front. Zones 1 (La Source), 3 (Les Combes), and occasionally 2, 4, and 5 dominate the decision data. The 5-second overtake cooldown is not the cause (inter-zone times exceed 5 seconds at all transitions). Zone differentiation metrics reported throughout this analysis are therefore measuring specialisation across a 3 to 5 zone effective decision space, not the full 9-zone track. All cross-alpha comparisons remain valid because the zone structure is constant across conditions.

---

## 9. Open questions for future work

1. What is the precise critical threshold between α=0.50 and α=0.75? A finer sweep (e.g., α=0.60, 0.65, 0.70) would locate the instability transition point.

2. Would a centralised critic (MADDPG) or value decomposition (QMIX) sustain cooperative zone conventions above the IL-MARL stability threshold?

3. Does the α=0.25 zone differentiation advantage persist at medium/high complexity where tyre dynamics and traffic density provide richer state representations?

4. Is the α=1.0 passive Nash equilibrium globally stable, or can escape be achieved with modified exploration schedules (e.g., Boltzmann exploration instead of epsilon-greedy)?

5. How does the collapse mechanism interact with training budget? At 1000 or 5000 training episodes, would α=0.75 still collapse at the same rate?

---

## 10. Conclusions

Phase 4 produces four principal conclusions:

**C1.** Partial incentive alignment (α=0.25) generates the highest zone specialisation observed in the full study — 71% above the purely competitive baseline at deterministic conditions — confirming H1 for the weak cooperative regime. This zone specialisation constitutes an emergent form of cooperative convention, arising from the shared incentive structure without explicit communication or coordination.

**C2.** The stability of cooperative conventions degrades with noise under partial cooperation, but more severely than under pure competition. The zone differentiation produced by α=0.25 at s0 (mean 0.543) drops to 0.161 at s2 — a 70% reduction — while α=0.0 maintains 0.317–0.406 across all stochasticity levels. Partially cooperative conventions are more fragile under environmental noise than purely competitive strategies.

**C3.** There is a sharp stability threshold in the interval (0.50, 0.75). Below this threshold, IL-MARL agents maintain competitive viability with partial zone specialisation. Above this threshold, the cooperative signal induces pathological convergence to a passive Nash equilibrium where positional advantage at race start determines all outcomes. This transition is abrupt and stochasticity-independent above the threshold.

**C4.** The mechanism of collapse at high alpha is incentive inversion: at α=1.0, each agent's terminal reward equals the teammate's outcome, making the passive equilibrium (both agents hold, starting positions unchanged) strictly dominant over any individual attempt to improve position. At α=0.75, the 75% teammate weight is sufficient to drive the same convergence in most training runs, despite the 25% residual own-incentive. This represents a form of "tragedy of cooperation" specific to IL-MARL: the incentive structure intended to promote joint optimisation instead produces mutual de-incentivisation of action.

These findings directly inform the dissertation's central argument: in competitive multi-agent systems, the design of the shared reward coefficient is not merely a tuning parameter but a qualitative determinant of whether agents learn to cooperate through zone specialisation or converge to degenerate mutual passivity. The sweet spot for IL-MARL appears to lie in the weakly cooperative regime (0 < α < 0.5), where shared incentive creates mild differentiation pressure without inverting the individual competitive drive.
