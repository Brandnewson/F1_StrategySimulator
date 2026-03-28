# Phase 4 Analysis — Alpha = 0.25 (Weakly Cooperative Incentive Regime)

**Date:** 2026-03-28
**Algorithm:** rainbow-lite
**Alpha:** 0.25 (each agent receives 75% own outcome + 25% teammate outcome)
**Budget:** 500 training episodes, 150 evaluation races per seed
**Seeds:** 101, 202, 303
**Stochasticity levels:** s0 (deterministic), s1 (2% noise), s2 (5% noise)
**Total trials:** 9

---

## 1. Simulator verification

Before analysing results, these checks confirm the reward-sharing mechanism is operating correctly.

| Check | Status | Evidence |
|-------|--------|----------|
| `reward_sharing_alpha` recorded in every output JSON | PASS | All 9 files show `"reward_sharing_alpha": 0.25` |
| `phase` field updated to `phase4_marl` | PASS | All 9 files show `"phase": "phase4_marl"` |
| Both agents are rainbow-lite | PASS | `algorithm: rainbow_lite` in all 9 files |
| No DNF in any trial | PASS | `dnf_rate: 0.0` across all agents in all trials |
| Config path points to Phase 4 directory | PASS | All reference `config_rainbow_alpha025.json` |
| Alpha override received from CLI | PASS | Confirmed by `reward_sharing_alpha: 0.25` matching the `--alpha 0.25` flag |

The simulator is working as intended. The alpha parameter is passed, stored, and applied.

---

## 2. Raw trial data

### s0 — Deterministic (probability_noise_std = 0.0)

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.360 | [0.283, 0.437] | 0.640 | 0.592 | 0.251 | stable (−0.02) | Competitive, A2 dominant |
| 202 | 0.020 | [−0.002, 0.042] | 0.980 | 0.379 | 0.477 | stable (−0.02) | **DEGENERATE — A1 collapse** |
| 303 | 0.507 | [0.426, 0.587] | 0.493 | 0.494 | 0.350 | stable (−0.04) | Competitive, near-parity |

### s1 — Low noise (probability_noise_std = 0.02)

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.513 | [0.433, 0.594] | 0.487 | 0.333 | 0.341 | a2 strengthening (−0.08) | Competitive, near-parity |
| 202 | 0.593 | [0.514, 0.672] | 0.407 | 0.501 | 0.145 | a2 strengthening (−0.14) | Competitive, A1 dominant |
| 303 | 0.500 | [0.420, 0.580] | 0.500 | 0.326 | 0.295 | a2 strengthening (−0.16) | Competitive, exact parity |

### s2 — High noise (probability_noise_std = 0.05)

| Seed | A1 WR | CI 95% | A2 WR | Zone diff | Risk diff | Drift | Classification |
|------|-------|--------|-------|-----------|-----------|-------|----------------|
| 101 | 0.553 | [0.474, 0.633] | 0.447 | 0.129 | 0.142 | a2 strengthening (−0.16) | Competitive, near-parity |
| 202 | 0.587 | [0.508, 0.666] | 0.413 | 0.147 | 0.247 | stable (+0.04) | Competitive, A1 dominant |
| 303 | 0.507 | [0.426, 0.587] | 0.493 | 0.205 | 0.244 | stable (+0.02) | Competitive, near-parity |

---

## 3. Summary statistics by stochasticity level

| Stoch | Mean WR | Zone diff mean | Risk diff mean | Collapse | Competitive |
|-------|---------|----------------|----------------|----------|-------------|
| s0 | 0.296 | 0.488 | 0.359 | 1/3 (33%) | 67% |
| s1 | 0.535 | 0.387 | 0.260 | 0/3 (0%) | 100% |
| s2 | 0.549 | 0.161 | 0.211 | 0/3 (0%) | 100% |

**Note on s0 mean WR:** The degenerate s202 trial (WR=0.020) heavily suppresses the s0 mean. Excluding it, the non-degenerate s0 mean is (0.360 + 0.507) / 2 = 0.434.

---

## 4. Comparison with Phase 3 baseline (alpha = 0.0, rainbow-lite)

| Metric | Phase 3 (α=0.0) | Phase 4 (α=0.25) | Direction |
|--------|-----------------|------------------|-----------|
| s0 collapse rate | 0/3 (0%) | 1/3 (33%) | Worse |
| s1 collapse rate | 0/3 (0%) | 0/3 (0%) | Same |
| s2 collapse rate | 0/3 (0%) | 0/3 (0%) | Same |
| s0 zone diff mean | ~0.25–0.41* | 0.488 | Higher |
| s1 zone diff mean | ~0.33* | 0.387 | Higher |
| s2 zone diff mean | ~0.41* | 0.161 | Lower |
| s0 WR spread (CI width) | ~0.145 | ~0.154 | Similar |
| s2 WR spread (CI width) | ~0.160 | ~0.160 | Similar |

*Phase 3 rainbow zone differentiation values taken from `phase3_full_analysis.md` trial tables.

---

## 5. Key findings

### Finding A1: One degenerate collapse at s0 — partial incentive alignment does not guarantee collapse prevention

Seed 202 at s0 produced a classic A1 degenerate collapse. A1 made only 16 total overtake attempts across 150 evaluation races (against ~6,300 decisions), locking in to near-complete passivity. This is mechanistically identical to the Phase 3 collapse mechanism: positional asymmetry fills A1's replay buffer with failed attempts, Q(HOLD) rises above Q(ATTEMPT), and epsilon decay cements the passive policy.

This finding is significant. Rainbow-lite at alpha=0.0 showed 0% collapse across all 9 Phase 3 trials. The fact that a single collapse occurs at alpha=0.25 (s0, seed 202) suggests that the 25% team incentive can, under certain seed initialisations, interfere with the priority weighting in the replay buffer in a way that destabilises A1's learning without fully redirecting its incentive towards teammate benefit.

Alternative interpretation: this may be a positional bias effect rather than an alpha effect. At s0, the deterministic environment means A1 starting behind and consistently losing can lock in collapse without noise providing recovery opportunities. At s1 and s2, noise breaks the deterministic failure loop.

Both interpretations require verification against alpha=0.50 and alpha=1.0 data.

### Finding A2: Zone differentiation is substantially higher at s0 than Phase 3 baseline

The two non-degenerate s0 trials show zone_diff = 0.592 and 0.494. The Phase 3 alpha=0.0 s0 trials showed zone differentiation typically in the 0.2–0.4 range. Even at alpha=0.25, agents develop more distinct territorial strategies under deterministic conditions than they did with zero team incentive. This is the first empirical evidence consistent with H1: partial incentive alignment increases zone specialisation.

At s0 seed 101, the pattern is clear. A2 concentrates on La Source (100% attempt rate, 91.6% success rate, 252/275 decisions resulting in attempts). A1 distributes effort across La Source (36%), Les Combes (41%), Pouhon (27%), and Stavelot (26%). A2 owns the primary zone; A1 disperses across secondary zones. This is a complementary territorial split.

At s0 seed 303, both agents attempt La Source at 100% but A1 uses zones 3/4/5/9 as secondaries and avoids zone 2, while A2 commits heavily to zones 2/4/7. The agents have differentiated secondary territories despite sharing the primary zone.

### Finding A3: Zone differentiation collapses dramatically at s2

The mean zone_diff at s0 (non-degenerate trials) is approximately 0.543. At s2, the mean drops to 0.161. This is the sharpest stochasticity-driven erosion of zone differentiation observed so far. At s2, both agents converge toward broadly similar attempt-rate profiles across zones — zone specialisation dissolves under high noise.

This directly addresses RQ2. For alpha=0.25, cooperative zone conventions ARE stable at s0 but break down at s2. The stability threshold appears between s1 (zone_diff mean 0.387 — still meaningfully differentiated) and s2 (zone_diff mean 0.161 — near the Phase 3 alpha=0.0 competitive baseline).

The mechanism is likely that s2 noise flattens the Q-value advantage of any particular zone by injecting enough variance that no zone reliably outperforms another. With flattened Q-value gradients, the incentive to specialise disappears.

### Finding A4: Risk profile asymmetry emerges — consistent with hawk-dove dynamics

Several trials show strongly asymmetric risk distributions between agents.

s1 seed 101: A1 — CONSERVATIVE 211, NORMAL 69, AGGRESSIVE 84 (conservative-leaning). A2 — CONSERVATIVE 131, NORMAL 0, AGGRESSIVE 378 (heavily aggressive).

s0 seed 303: A1 — CONSERVATIVE 30, NORMAL 96, AGGRESSIVE 215 (aggressive-leaning). A2 — CONSERVATIVE 438, NORMAL 11, AGGRESSIVE 265 (conservative-dominant with secondary aggression).

This asymmetric risk polarisation is consistent with the hawk-dove equilibrium class identified in Phase 3. Partial incentive alignment does not eliminate this pattern — it may reinforce it by making the cost of competing directly for the same zone higher when the teammate's outcome is partially internalised.

### Finding A5: Drift patterns at s1/s2 suggest A2 has a structural advantage

All three s1 trials show "a2 strengthening" drift (drift = −0.08, −0.14, −0.16). Two of three s2 trials are stable but the third also shows a2 strengthening. This pattern mirrors the positional evaluation bias identified in Phase 3 (~11pp structural disadvantage against the A1 label). Under partial reward sharing, this bias may be amplified if A1's team-adjusted incentive leads it to be less aggressive about defending its positional advantage.

---

## 6. Simulator health check summary

All 9 trials completed without error. No DNFs. No negative-infinity or NaN reward values. Risk attempt counts are non-zero for both agents in all non-degenerate trials. Zone behavior records are internally consistent. The output schema matches Phase 3 exactly, with the addition of `reward_sharing_alpha` and the updated `phase` field. The reward-sharing mechanism is confirmed active.

The one degenerate collapse (s0_s202) is a genuine behavioural outcome, not a simulation error. The extremely low attempt count (16 total) and stable but near-zero win rate confirm this is A1 converging to passive policy, not a crash or misconfiguration.

---

## 7. Open questions for subsequent alpha values

1. Does collapse rate increase further at alpha=0.50 and alpha=1.0, or does deeper cooperation prevent positional asymmetry locking in?
2. Does zone differentiation at s0 continue to increase from alpha=0.25 to alpha=0.50, or is alpha=0.25 near a peak?
3. Does s2 zone differentiation recover at alpha=1.0 (fully cooperative), where both agents receive the same reward signal regardless of position?
4. Is the "a2 strengthening" drift structural (positional bias) or alpha-dependent?

These questions will be addressed once alpha=0.50 and alpha=1.0 data are available.

---

## 8. Next steps

Run the 18 remaining Phase 4 trials:
- Alpha = 0.50: 9 trials (s0/s1/s2 × seeds 101/202/303)
- Alpha = 1.0: 9 trials (s0/s1/s2 × seeds 101/202/303)

After all 27 Phase 4 trials complete, run `scripts/compute_summary_table.py` to produce the full cross-alpha comparison table.
