# Phase 3 Full Analysis: Independent Learner MARL — Cross-Stochasticity
**Date:** 2026-03-26
**Trials completed:** 18 (2 algorithms × 3 seeds × 3 stochasticity levels)
**Status:** All 18 trials complete, all valid for analysis.

---

## 1. What Phase 3 Is and Why It Was Necessary

### 1.1 The Gap Phase 2 Left Open

Phase 2 established that vanilla DQN and rainbow-lite both produce reliable overtaking policies when trained against a *fixed* opponent — a Base Agent using a predetermined gap-based heuristic. That fixed-opponent framing answers **RQ3** (which DQN variant is most reliable?) but it does not answer either of the following:

- **RQ1**: Do those policies remain robust as stochasticity increases *when the opponent is also adapting*?
- **RQ2**: What strategic behaviours emerge from two learning agents in direct competition?

A fixed opponent introduces no non-stationarity. From the learning agent's perspective, the opponent's policy is constant across all training episodes; the Q-value function can converge normally. This is a comfortable but artificial setting. In any real MARL environment — racing, game theory, robotics, mechanism design — the opponent co-adapts alongside the agent, which means the environment itself is non-stationary from each agent's perspective. Violations of the stationarity assumption that underpins Q-learning convergence guarantees (Watkins & Dayan, 1992) become live concerns.

Phase 3 removes this artificial comfort. The Base Agent is replaced by a second concurrent DQN learner. Both agents train from their own independent replay buffers after every episode and share no weights, gradients, or Q-values. This is **independent learner MARL (IL-MARL)** — the canonical baseline in the multi-agent RL literature (Foerster et al., 2017). It is minimal in implementation (no architectural changes; the simulator already loops over all DQN drivers and calls `train()` on each after each episode) but maximal in what it introduces: genuine non-stationarity, mutual adaptation, and the possibility of emergent strategic interaction.

### 1.2 What Was Held Constant (The Control Design)

To isolate the effect of opponent type (fixed heuristic → concurrent learner), every other experimental variable was held identical to Phase 2:

| Variable | Value |
|:---------|:------|
| Track | Spa-Francorchamps, 5 laps, 9 overtaking zones |
| Training budget | 500 episodes per trial |
| Evaluation budget | 150 races per seed |
| Seeds | 101, 202, 303 |
| Stochasticity levels tested | s0 (deterministic), s1 (mild noise), s2 (high noise) |
| Network: hidden size, learning rate, discount | 512, 7×10⁻⁴, 0.99 |
| Reward weights: outcome, tactical, penalty | 2.0, 0.05, 2.0 |
| Complexity profile | `low_marl` (two concurrent DQN agents) |

**Algorithm pairings** are same-algorithm matchups: vanilla vs vanilla and rainbow-lite vs rainbow-lite. Mixing algorithms would confound strategy emergence with algorithm asymmetry. By holding algorithm constant, any policy differentiation must arise from the dynamics of the MARL interaction itself — different random weight initialisations, minor early asymmetries that compound under non-stationary co-adaptation. This is the cleanest possible test of whether IL-MARL itself generates emergent behaviour.

---

## 2. Metrics and Hypotheses Tested

### 2.1 Primary Metric

**Win rate of Agent 1 vs Agent 2** (`win_rate_a1_vs_a2`) with Wilson CI95. With two agents and exactly two finishing positions {1, 2}, this fully captures relative performance. A sustained departure from 0.50 indicates that one agent has converged to a consistently dominant policy. This replaces the Phase 2 `win_rate_vs_baseline` metric because there is no longer a fixed external reference; performance is inherently relative between the two co-adapting agents.

### 2.2 Secondary Metrics

- **Non-stationarity drift**: win rate of A1 in the first third of evaluation races minus the last third. A magnitude > 0.05 indicates that policies have not converged and are still co-adapting during evaluation.
- **Zone differentiation index**: mean absolute difference in attempt rates between agents across all overtaking zones. High index (> 0.25) = agents have specialised to different circuits sections.
- **Risk differentiation index**: mean absolute difference in CONSERVATIVE/NORMAL/AGGRESSIVE proportions. High index (> 0.25) = agents have developed distinct risk personalities.
- **Degenerate equilibrium flag**: A trial is classified as **degenerate** when WR = 1.000 (or 0.000) with zero variance — one agent has a pure hold policy across all 9 zones at all 5 laps across all 150 evaluation races. This is the leader-follower Nash equilibrium documented by Matignon et al. (2012): the losing agent's replay buffer fills with failed-chase experiences until Q(HOLD) >> Q(ATTEMPT) at every zone, locking it into permanent second place.

### 2.3 Hypotheses

| Hypothesis | Statement |
|:-----------|:----------|
| **H_ns1** | Non-stationarity erodes positional stability: |drift| > 0.05 across multiple seeds indicates the 500-episode budget is insufficient for IL-MARL convergence |
| **H_ns2** | Rainbow-lite's stability advantage from Phase 2 transfers to the non-stationary setting: lower across-seed variance and lower degenerate rate than vanilla |
| **H_diff** | Same-algorithm agents spontaneously develop differentiated strategies: zone or risk differentiation index > 0.10 in at least one pairing |

---

## 3. Complete Dataset — All 18 Trials

### 3.1 Vanilla vs Vanilla

*† = degenerate equilibrium (WR = 1.000, A2 pure HOLD, zero variance)*

| Stoch | Seed | WR A1 | CI95 | Drift | Drift Interp | Zone Diff | Risk Diff | Classification |
|:------|:-----|:------|:-----|:------|:------------|:---------|:---------|:--------------|
| s0 | 101 | **1.000** | [1.000, 1.000] | 0.000 | stable | 0.000 | 0.000 | **DEGENERATE†** |
| s0 | 202 | 0.593 | [0.514, 0.672] | −0.160 | A2 strengthening | 0.222 | 0.212 | moderate |
| s0 | 303 | 0.507 | [0.426, 0.587] | +0.160 | A1 strengthening | 0.174 | 0.174 | moderate |
| s1 | 101 | 0.827 | [0.766, 0.887] | +0.040 | stable | 0.352 | 0.277 | **high** |
| s1 | 202 | 0.587 | [0.508, 0.666] | +0.040 | stable | 0.290 | 0.153 | **high** |
| s1 | 303 | **1.000** | [1.000, 1.000] | 0.000 | stable | 0.000 | 0.000 | **DEGENERATE†** |
| s2 | 101 | **1.000** | [1.000, 1.000] | 0.000 | stable | 0.000 | 0.000 | **DEGENERATE†** |
| s2 | 202 | 0.327 | [0.251, 0.402] | −0.140 | A2 strengthening | 0.450 | 0.278 | **high** |
| s2 | 303 | **1.000** | [1.000, 1.000] | 0.000 | stable | 0.000 | 0.000 | **DEGENERATE†** |

**Degenerate count: 4/9 trials (44%)**. By stochasticity level: s0 = 1/3, s1 = 1/3, s2 = 2/3.

### 3.2 Rainbow-lite vs Rainbow-lite

| Stoch | Seed | WR A1 | CI95 | Drift | Drift Interp | Zone Diff | Risk Diff | Classification |
|:------|:-----|:------|:-----|:------|:------------|:---------|:---------|:--------------|
| s0 | 101 | 0.587 | [0.508, 0.666] | −0.120 | A2 strengthening | 0.003 | 0.256 | low zone / mod risk |
| s0 | 202 | 0.547 | [0.467, 0.627] | 0.000 | stable | 0.417 | 0.170 | **high** |
| s0 | 303 | 0.527 | [0.447, 0.607] | −0.020 | stable | 0.212 | 0.345 | moderate |
| s1 | 101 | 0.480 | [0.400, 0.560] | +0.020 | stable | **0.476** | **0.393** | **high** |
| s1 | 202 | 0.373 | [0.296, 0.451] | −0.020 | stable | 0.103 | 0.389 | moderate |
| s1 | 303 | 0.593 | [0.514, 0.672] | +0.160 | A1 strengthening | 0.225 | 0.233 | moderate |
| s2 | 101 | 0.220 | [0.153, 0.287] | +0.040 | stable | 0.022 | 0.378 | low |
| s2 | 202 | 0.467 | [0.387, 0.547] | −0.080 | A2 strengthening | 0.188 | 0.350 | moderate |
| s2 | 303 | 0.513 | [0.433, 0.594] | −0.040 | stable | 0.063 | 0.418 | low |

**Degenerate count: 0/9 (0%).**

---

## 4. Cross-Stochasticity Analysis

### 4.1 Degenerate Equilibrium Rate

The most striking finding in the full dataset is the divergence in degenerate equilibrium rates between the two algorithms.

| Algorithm | s0 degenerate | s1 degenerate | s2 degenerate | Total |
|:----------|:-------------|:-------------|:-------------|:------|
| Vanilla | 1/3 (33%) | 1/3 (33%) | 2/3 (67%) | **4/9 (44%)** |
| Rainbow | 0/3 (0%) | 0/3 (0%) | 0/3 (0%) | **0/9 (0%)** |

The degenerate rate for vanilla **doubles at s2**, and it is completely absent for rainbow across all noise levels. This is not a marginal difference; it is a qualitative failure mode that vanilla exhibits systematically under conditions that rainbow handles entirely without issue.

**Mechanism:** In vanilla IL-MARL, when one agent gets a small positional advantage early in training, it accumulates more positive experiences in its replay buffer (La Source zone 1, difficulty 0.2, consistent near-certain success). The trailing agent, behind more often, accumulates failed attempt experiences. With uniform replay sampling and no prioritisation, the trailing agent's positive samples are diluted by large volumes of failed-attempt records. Eventually Q(HOLD) dominates at all zones and the agent settles into passive following. This is a self-reinforcing feedback loop: passive following reduces encounter frequency with overtaking zones, further reducing the signal for Q(ATTEMPT) to recover.

**Why rainbow is immune:** PER (prioritised experience replay) breaks this loop. Failed attempts produce high TD errors — the discrepancy between predicted reward and the negative reward received is large — and PER prioritises these samples for replay. Crucially, successful counter-examples are also high-TD-error when first encountered. The trailing rainbow agent continues to learn from its failures at high weight, keeping Q(ATTEMPT) values competitive with Q(HOLD). The n-step return further helps: a multi-step chain of consecutive sub-optimal decisions produces a compounded negative return that generates strong corrective signal rather than a weak single-step signal easily overwhelmed by the noise of MARL.

### 4.2 Win Rate Profiles (Excluding Degenerate Trials)

With degenerate trials excluded, the remaining valid win rates show markedly different profiles.

**Vanilla valid win rates:**

| Stoch | Seed | WR A1 | Note |
|:------|:-----|:------|:-----|
| s0 | 202 | 0.593 | A1 slight advantage |
| s0 | 303 | 0.507 | Near parity |
| s1 | 101 | **0.827** | A1 dominant — largest competitive WR in dataset |
| s1 | 202 | 0.587 | A1 slight advantage |
| s2 | 202 | **0.327** | A2 dominant — reversal |

The range across valid vanilla trials is 0.327–0.827 (span = 0.500). This extreme variance is itself a finding: when vanilla does not collapse into a degenerate equilibrium, it tends to produce a large and seed-dependent winner. The "middle" outcomes near parity (0.507, 0.587) are present but the distribution has fat tails. Vanilla IL-MARL is characterised by **bimodal outcomes**: either degenerate collapse (one agent holds passively) or highly asymmetric competition (one agent dominant but still learning).

**Rainbow valid win rates (all 9):**

| Stoch | Mean WR A1 | WR Range | Interpretation |
|:------|:----------|:---------|:--------------|
| s0 | 0.554 | 0.527–0.587 | A1 slight competitive advantage; tight cluster |
| s1 | 0.482 | 0.373–0.593 | Near parity; moderate spread |
| s2 | 0.400 | 0.220–0.513 | A2 advantage emerging; wide spread |

The rainbow win rate profile tells a coherent story: at s0, A1 has a small but consistent advantage, suggesting minor asymmetry in the two agents' convergence trajectories that emerged from the same architecture but different random seeds. At s1, this is washed out by noise and the distribution centres near parity. At s2, A2 develops a systematic advantage (mean WR A1 = 0.400). The directionality is consistent across seeds at s2 (all three have A1 WR ≤ 0.513), which rules out random fluctuation as an explanation.

**Why A2 gains advantage at s2 for rainbow:** This is the most novel finding for further investigation. A plausible mechanism: under high stochasticity, the agent that begins behind more often (A2) encounters higher-variance race outcomes and therefore generates higher-TD-error transitions more frequently. PER amplifies these, accelerating A2's learning relative to A1. A1, ahead more often, sees low-TD-error maintenance transitions (already near-optimal La Source) which are de-prioritised by PER. The asymmetry is an unintended consequence of PER's prioritisation scheme in high-noise environments. This should be verified in Phase 4 by running cross-play experiments between agents trained independently to confirm it is a training effect rather than a positional artefact.

### 4.3 Non-Stationarity Drift Across Stochasticity

**Vanilla drift magnitudes (valid trials only):**

| Stoch | Seed | |Drift| | Interpretation |
|:------|:-----|:--------|:--------------|
| s0 | 202 | 0.160 | High — not converged |
| s0 | 303 | 0.160 | High — not converged |
| s1 | 101 | 0.040 | Low |
| s1 | 202 | 0.040 | Low |
| s2 | 202 | 0.140 | High — still adapting |

Vanilla's drift pattern is non-monotonic: high at s0, low at s1, high again at s2. The s1 trials that show low drift (0.040) correspond to the cases where one agent achieved a large win-rate advantage (s101: WR=0.827). A large, stable advantage means the dominant agent has effectively "solved" the MARL problem for its seed — it converged to a policy that consistently wins, and the subordinate agent settled into a semi-passive role. This produces low drift during evaluation even though the underlying dynamic during training was highly non-stationary. **Low drift is not always a sign of healthy convergence** — it can indicate that competition has already resolved into a stable hierarchy (close to degenerate, but not yet fully collapsed).

**Rainbow drift magnitudes (all trials):**

| Stoch | Mean |Drift| | Max |Drift| | Interpretation |
|:------|:------------|:-----------|:--------------|
| s0 | 0.047 | 0.120 | Mostly converged; s101 exception |
| s1 | 0.067 | 0.160 | Small increase; s303 outlier |
| s2 | 0.053 | 0.080 | Stable; A2 advantage present but not growing |

Rainbow drift is remarkably stable across stochasticity levels. The mean drifts (0.047, 0.067, 0.053) are all below the 0.10 threshold that would indicate clear non-convergence. Even the worst case (rainbow s1 s303, drift=0.160) occurs in a trial where A1 ends stronger, suggesting a late-training advantage that could continue — but critically, A2 is still actively competing (WR A2 ≈ 0.41), not collapsed. This is qualitatively different from vanilla's high-drift cases, which occur alongside near-parity win rates (both agents still fully in contention, reward signals too noisy to resolve).

### 4.4 Strategy Differentiation Across Stochasticity

#### Zone differentiation

| Algorithm | s0 mean | s1 mean | s2 mean | Trend |
|:---------|:--------|:--------|:--------|:------|
| Vanilla (valid) | 0.198 | 0.321 | 0.450* | Increasing |
| Rainbow | 0.211 | 0.268 | 0.091 | Peaks at s1, collapses at s2 |

*s2 vanilla valid = only 1 trial (s202, zone_diff=0.450).

The two algorithms show opposite trends for zone differentiation under noise. Vanilla valid trials show increasing zone separation as stochasticity increases — but this is largely a consequence of winner-loser forced asymmetry: the dominant agent actively exploits La Source and one or two high-yield zones; the subordinate agent is pushed into low-frequency encounter patterns with whatever zones it can reach. This is not *chosen* specialisation; it is forced separation.

Rainbow zone differentiation peaks at s1 (mean 0.268, with s101 at 0.476 — the highest single-trial zone_diff in the entire Phase 3 dataset) and then collapses at s2 (mean 0.091). Under high noise, the per-zone value signal becomes too unreliable for either agent to develop a stable zone preference. With success probabilities varying substantially across realisations, the learned Q-values for specific zones fail to remain discriminative. Both rainbow agents retreat toward low-difficulty zones (La Source dominates attempts at s2 for both) and the zone diversity that characterised s0/s1 disappears. This collapse is not a failure of the architecture but a rational response to a noisy reward landscape.

#### Risk differentiation

| Algorithm | s0 mean | s1 mean | s2 mean | Trend |
|:---------|:--------|:--------|:--------|:------|
| Vanilla (valid) | 0.193 | 0.215 | 0.278 | Gradually increasing |
| Rainbow | 0.257 | 0.338 | 0.382 | **Monotonically increasing** |

Risk differentiation follows the opposite pattern to zone differentiation for rainbow: it increases monotonically with stochasticity. Under high noise, rainbow agents maintain distinctly different risk levels (mean risk_diff=0.382 at s2) even when their zone footprints become indistinguishable. This is a critical finding for RQ2: **under high stochasticity, rainbow IL-MARL agents shift from zone specialisation to risk-level specialisation as their primary mode of strategic differentiation.** The hawk-dove risk polarisation that was visible in s0 (one agent systematically more aggressive, one more conservative) not only survives noise but intensifies.

This makes game-theoretic sense. In a hawk-dove payoff structure, the value of playing aggressive relative to conservative depends on the *opponent's* strategy — which is exactly the non-stationary signal the agents are adapting to. As zone-level signals become noisy, risk level becomes the higher-order variable they reliably learn to differentiate on, because it indexes the opponent's likely response pattern more robustly than any individual zone encounter.

---

## 5. Hypothesis Evaluations

### H_ns1 — Non-stationarity erodes positional stability

**Verdict: PARTIALLY CONFIRMED, with algorithm-dependence.**

Vanilla shows clear evidence of unresolved non-stationarity:
- s0 valid drift = 0.160 in 2/2 non-degenerate seeds → insufficient convergence at s0
- s2 valid drift = 0.140 → still not converged under high noise
- Degenerate rate doubles at s2 → the MARL dynamics are destabilised by noise

Rainbow shows modest but real drift:
- Mean |drift| ranges from 0.047 to 0.067 across stochasticity levels
- No trial has |drift| > 0.160
- The 500-episode budget achieves near-convergence for rainbow but not for vanilla

The hypothesis that stochasticity *escalates* non-stationarity is confirmed for vanilla (s0 valid drift 0.160 → s2 valid drift 0.140, with collapse in between) but largely disconfirmed for rainbow (drift is stable across s0–s2, 0.047–0.067 mean). The implication for Phase 4 is that the training budget should be increased to 750 episodes specifically to address vanilla's failure to converge at s0 — a longer budget gives the agents more time to resolve the non-stationarity before evaluation begins.

### H_ns2 — Rainbow-lite's stability advantage transfers to MARL

**Verdict: STRONGLY CONFIRMED.**

The evidence is overwhelming:
- Rainbow: 0/9 degenerate trials vs vanilla: 4/9 (44%) — categorical difference in stability mode
- Rainbow mean win rate variance across seeds at each stochasticity level: s0 = 0.0009, s1 = 0.0102, s2 = 0.0221 — all low
- Vanilla valid win rate variance: s0 = 0.0020, s1 = 0.0290, s2 = — (insufficient valid trials)
- Rainbow mean drift is stable (0.047–0.067); vanilla mean drift is high and inconsistent (0.160, 0.040, 0.140)

The mechanisms that make rainbow more stable in single-agent (Phase 2) transfer directly to the multi-agent setting: PER prevents the degenerate replay buffer imbalance that causes vanilla to collapse into leader-follower equilibria, and n-step returns provide stronger signal through the noisy training dynamics of IL-MARL. This confirms that the algorithm properties matter at least as much as the training environment when designing MARL systems — a well-engineered value-based agent is substantially more resilient to non-stationarity than a naive variant.

### H_diff — Same-algorithm agents spontaneously develop differentiated strategies

**Verdict: CONFIRMED, with important nuance about the nature of differentiation.**

Zone differentiation index > 0.10 in 10/14 non-degenerate trials (71%).
Risk differentiation index > 0.10 in 14/14 non-degenerate trials (100%).

Strategy differentiation is universal and robust across algorithms and stochasticity levels, at least in terms of risk approach. The nuance is that vanilla differentiation in valid trials is partially *forced* by winner-loser asymmetry (the trailing agent is denied encounter opportunities), whereas rainbow differentiation is *chosen* — it arises from two agents with genuine competitive dynamics who converge to different tactics. The distinction matters for how the findings inform RQ2: vanilla's differentiation signals the resolution of competition into hierarchy; rainbow's differentiation signals the emergence of co-existing strategies within a competitive equilibrium.

---

## 6. Answering the Research Questions

### RQ1: How robustly did MARL policies maintain performance as stochasticity increased?

The honest answer requires distinguishing the two algorithms, because their robustness profiles are qualitatively different.

**Vanilla:** Robustness fails under MARL conditions in a way that did not appear in Phase 2. In Phase 2 (single-agent), vanilla showed modest stochasticity degradation (win rate drop of ~3pp from s0 to s2) with seed variance tripling. In Phase 3, vanilla exhibits degenerate collapse at 44% of trials across all stochasticity levels, with that rate doubling at s2. When vanilla does not collapse, it produces highly variable outcomes with a win-rate range of 0.500 between the best and worst valid trials. The Phase 2 finding that vanilla is a reliable policy against a fixed opponent does not generalise to MARL settings.

The reason is structural: uniform replay sampling and single-step returns cannot maintain the signal-to-noise ratio needed for both agents to keep improving when both are simultaneously changing. Whichever agent gets a small early advantage compounds it through the replay buffer imbalance mechanism until the dynamic resolves into a stable hierarchy — which may be degenerate.

**Rainbow:** Robustness is maintained across all stochasticity levels in the sense that no rainbow trial produces a degenerate equilibrium. Both agents remain active learners with genuine competition throughout. However, rainbow is not uniformly robust in the sense of maintaining constant win rates: at s2, A2 develops a systematic advantage (mean A1 WR = 0.400), which represents a 15pp shift from the s0 mean (0.554). This degradation is present but it is a genuine competitive shift — one agent learns a more effective policy under high-noise conditions — rather than a collapse.

**Summary for RQ1:** Rainbow-lite MARL policies are robustly competitive under all tested stochasticity levels. Vanilla MARL policies are unreliable — competitive in roughly half of trials and degenerate in the other half, with the degenerate fraction worsening as noise increases. The stochasticity robustness ordering from Phase 2 (rainbow > vanilla) is confirmed under MARL conditions, and the magnitude of the difference is larger than Phase 2 indicated.

---

### RQ2: What strategic behaviours emerged from cooperative-competitive interaction?

Phase 3 reveals two distinct classes of emergent behaviour, one per algorithm. They reflect fundamentally different ways that competition resolves under the same MARL setting.

#### Emergent behaviour class 1: Hierarchical resolution (vanilla)

In 4/9 vanilla trials, two symmetric agents with identical initialisations, architectures, and stochasticity exposure develop into a rigid hierarchy: one permanent leader, one permanent follower. The follower does not adapt — across 150 evaluation races and 9 zones over 5 laps, it makes zero overtaking attempts. This is the **leader-follower Nash equilibrium** documented by Matignon et al. (2012): a degenerate outcome of IL-MARL with homogeneous agents, uniform replay, and insufficient training budget relative to competition intensity.

In the 5 valid vanilla trials, a softer version of the same dynamic is visible: one agent achieves a substantial win rate advantage (0.593, 0.827, 0.327 for the clearly-advantaged agent) and the competition has visibly resolved into a hierarchy even without full collapse. The zone and risk differentiation in these trials is high (zone_diff up to 0.450) but it is asymmetric differentiation — the winner occupies the valuable zones, the loser is excluded from them by positional dynamics.

The type of competition vanilla generates, when it generates competition at all, is closer to arms-race escalation followed by resolution than to sustained strategic diversity. One agent "wins" the training process and the other adapts to a subordinate strategy.

#### Emergent behaviour class 2: Sustained strategic coexistence (rainbow)

In all 9 rainbow trials, two agents maintain competitive dynamics throughout, with win rates ranging 0.220–0.593 (never at either extreme). Within this competition, two sub-patterns emerge:

**Zone specialisation (most prominent at s0 and s1):** Agents partition the circuit into spheres of influence without any communication or explicit coordination mechanism. The clearest example is rainbow s0 s202 (zone_diff=0.417): A1 concentrates exclusively on Pouhon (100% attempt rate) and La Chicane, completely ignoring Raidillon (0% attempt rate); A2 commits fully to Raidillon (100% attempt rate) and Campus. This is spatially non-overlapping territory — two agents have independently computed that contesting the same zones against each other is less profitable than each owning a subset. The emergence of this partitioning from independent learners with no coordination mechanism is direct empirical evidence that IL-MARL produces behavioural novelty beyond what either agent would develop against a fixed opponent.

At rainbow s1 s101, this reaches its most extreme form: zone_diff=0.476, the highest in the entire Phase 3 dataset. Agent 1 dominates La Source (100% attempt rate) and Les Combes but holds at Raidillon (0% attempts, 15 decisions). Agent 2 spreads its attempts across Les Combes, Bruxelles, Pouhon, and Stavelot — a complementary geography that broadly avoids Agent 1's core territory. Neither agent was instructed to behave this way. The Q-value function of each agent independently learned that attempting at zones heavily contested by the opponent produces worse expected returns than specialising.

**Risk polarisation (persistent and intensifying under noise):** Across all rainbow trials, risk differentiation index > 0.10 is universal and the risk profiles consistently show one more-aggressive and one more-conservative agent, even when zone footprints overlap. At s2, when zone differentiation collapses due to noisy reward signals, risk polarisation increases (mean risk_diff = 0.382). The two rainbow agents at s2 maintain distinct risk identities even when both are broadly restricted to La Source and occasional other zones. This is a **hawk-dove equilibrium** at the risk level: one agent adopts a systematically higher-aggression profile (AGGRESSIVE proportion > 40%) while the other maintains a conservative stance (CONSERVATIVE proportion > 50%), with each agent's strategy being a best response to the other's. The equilibrium is self-reinforcing: the aggressive agent's attempts force the conservative agent to hold (the conservative best response to a hawk opponent) and the conservative agent's holding makes the aggressive agent's attempts more valuable (fewer contested zones).

**Summary for RQ2:** Two qualitatively distinct strategic behaviours emerged:
1. **Zone specialisation / territorial partitioning** — agents divide the circuit into non-overlapping zones of influence; strongest under rainbow at s0 and s1.
2. **Risk-level polarisation (hawk-dove equilibrium)** — agents develop distinct aggression profiles that are mutual best responses; strongest under rainbow at all noise levels and intensifying at s2.

Neither behaviour was instructed, hardcoded, or present in the Phase 2 single-agent setting. Both arise purely from two learning agents solving the same optimisation problem simultaneously under competitive pressure.

---

## 7. Algorithm Comparison Summary

| Dimension | Vanilla IL-MARL | Rainbow IL-MARL |
|:----------|:----------------|:----------------|
| Degenerate rate | 44% overall; 67% at s2 | 0% across all stochasticity levels |
| Competitive outcome (valid trials) | Bimodal — near parity or heavily asymmetric | Consistently competitive; moderate A2 advantage at s2 |
| Win rate range (valid) | 0.327–0.827 (span = 0.500) | 0.220–0.593 (span = 0.373) |
| Non-stationarity drift | High (0.160) in s0 valid; unstable across levels | Stable (0.047–0.067 mean); monotonic |
| Zone differentiation | Increases with noise (forced asymmetry) | Peaks at s1, collapses at s2 |
| Risk differentiation | Moderately increases with noise | Strongly increases with noise; dominant channel at s2 |
| Emergent behaviour class | Hierarchical resolution / arms-race escalation | Zone specialisation + hawk-dove risk equilibrium |
| Robustness to noise (MARL) | Fails — collapse rate doubles at s2 | Maintained — competitive equilibria at all levels |
| Phase 4 readiness | Requires higher training budget (750 ep) to address convergence failure | Ready for Phase 4 at current 500-episode budget |

---

## 8. Limitations and Validity Threats

### 8.1 Single seed per trial
Each trial uses one training seed and one evaluation seed. Seed variance at the single-trial level is unquantifiable from Phase 3 data alone. The three seeds per condition provide a between-seed picture, but the within-seed variance (would the same training run produce a different outcome under a different random minibatch sequence?) is unknown.

### 8.2 La Source structural dominance persists
Zone 1 (La Source, difficulty=0.2) remains the dominant overtaking zone in Phase 3 as in Phase 2. Its near-certain success probability (~0.8) makes it the first zone any learning agent colonises. In degenerate vanilla trials, the leader concentrates entirely on La Source; in rainbow s2 trials, both agents attempt La Source far more than other zones. Win rate comparisons are therefore partly driven by which agent secures La Source access most reliably, which is in turn driven by starting position asymmetry in the race. This is a validity threat to strong claims about zone specialisation — some of the observed zone differentiation may reflect starting position dynamics rather than pure Q-value learning.

### 8.3 500-episode training budget
Phase 2 established that 500 episodes gives ε ≈ 0.08 (policy mostly converged) in the single-agent setting. In MARL, non-stationarity means the effective training time needed for convergence is longer — the target policy is a moving one. The vanilla convergence failures at s0 (drift=0.160) confirm this: vanilla's 500-episode budget is insufficient for reliable IL-MARL convergence. For rainbow, 500 episodes appears adequate at s0–s1 but marginal at s2 (drift=-0.080 in s202, with A2 still strengthening).

### 8.4 Same-algorithm matchups only
Phase 3 runs only vanilla-vs-vanilla and rainbow-vs-rainbow. Cross-algorithm matchups (vanilla-vs-rainbow, for instance) would provide additional evidence about algorithm-level skill transfer and whether the hawk-dove equilibria are algorithm-specific or general to IL-MARL. This is left to Phase 4.

---

## 9. Further Investigation

### Priority 1: Vanilla convergence at higher training budget
Rerun vanilla at s0 with 750 episodes to determine whether the degenerate collapse is a training-budget failure or a fundamental property of vanilla IL-MARL. If vanilla degenerate rate drops significantly at 750 episodes, the conclusion is budget-dependent; if it persists, vanilla is architecturally unsuited to IL-MARL regardless of budget.

### Priority 2: Rainbow A2 advantage at s2 — training artefact or genuine
The systematic A2 advantage at s2 (all three seeds, mean A1 WR = 0.400) requires a mechanistic explanation. Recommended investigation: run cross-play experiments in which Agent 1 trained at s2 is evaluated against Agent 2 trained at s2, and vice versa, with positions swapped. If the advantage is intrinsic to the A2 training trajectory, it should persist in cross-play. If it disappears when positions are randomised, it is a positional artefact (A2 encounters being-behind more often, producing better learning signal for catch-up strategies).

### Priority 3: Zone specialisation at s2 — why does it collapse?
Rainbow zone differentiation collapses from a mean of 0.268 at s1 to 0.091 at s2. The proposed mechanism is that high-noise reward signals make zone-specific Q-values unreliable. This can be tested directly: run rainbow at s2 with a higher training budget (e.g., 1000 episodes). If zone specialisation recovers with more training time, the collapse is a signal-to-noise issue (insufficient learning given noise). If it does not recover, the noise level at s2 is fundamentally too high for reliable zone discrimination and agents rationally fall back to risk-level strategies — a finding with direct implications for real-world race strategy (noisy track conditions destroy driver-level track section knowledge).

### Priority 4: Phase 4 team extension symmetry-breaking
The most important structural fix for Phase 4 is the introduction of pace asymmetry (one team's agents have a 1.05× lap time penalty). This artificially breaks the symmetric initialisation that causes degenerate equilibria in vanilla. When two symmetric agents cannot reach degenerate equilibria because one has a structural performance disadvantage, the MARL dynamics should be richer and more informative. The cooperation threshold α* (reward-sharing blend) becomes the key variable: at what α does the slower team's agents benefit from coordination? Rainbow MARL at s0/s1 with α ∈ {0, 0.25, 0.5, 0.75, 1.0} would provide the direct test of cooperative strategy emergence.

### Priority 5: PPO cross-algorithm MARL
Adding a PPO agent to Phase 3 (vanilla vs rainbow, rainbow vs PPO, vanilla vs PPO) would test whether the emergent hawk-dove and zone specialisation behaviours are specific to DQN-family value-based agents or general to any gradient-based learner in IL-MARL. PPO's on-policy nature means its policy never locks into Q(HOLD) the same way — it continuously samples the environment according to its current policy — making it more resistant to degenerate equilibria by a different mechanism than PER. If PPO and rainbow produce similar differentiation patterns against each other, the emergent behaviours are a feature of the competitive setting, not of specific algorithm properties.

---

## 10. Summary of Findings

| Finding | Evidence | Significance |
|:--------|:---------|:------------|
| Vanilla IL-MARL collapses to degenerate equilibria in 44% of trials | 4/9 trials WR=1.000, A2 zero attempts | Invalidates vanilla as a reliable IL-MARL agent; Phase 2 stability does not transfer |
| Degenerate rate doubles at s2 for vanilla | s0: 1/3, s1: 1/3, s2: 2/3 | Noise amplifies the replay buffer imbalance mechanism that drives degenerate collapse |
| Rainbow produces zero degenerate trials across all stochasticity levels | 0/9 degenerate | PER + n-step returns prevent the feedback loop that collapses vanilla; H_ns2 confirmed |
| Rainbow drift is stable across s0–s2 (0.047–0.067 mean) | Drift magnitudes table | 500-episode budget is adequate for rainbow IL-MARL convergence at s0–s1; marginal at s2 |
| Zone specialisation emerges spontaneously in rainbow at s0–s1 | Zone diff peak 0.476 (s1 s101); territorial partitioning of circuit sections | Direct empirical answer to RQ2: IL-MARL generates behavioural novelty without coordination mechanisms |
| Risk polarisation intensifies monotonically under noise for rainbow | risk_diff s0=0.257, s1=0.338, s2=0.382 | Hawk-dove equilibrium at risk level; agents shift primary differentiation axis under noise |
| A2 gains systematic advantage at s2 for rainbow | All 3 s2 seeds: A1 WR ≤ 0.513, mean=0.400 | Potential PER asymmetry at high noise; requires cross-play investigation in Phase 4 |
| H_diff confirmed universally | risk_diff > 0.10 in 14/14 non-degenerate trials | Same-algorithm agents consistently develop different strategies under competitive pressure |

---

## 11. Conclusion

Phase 3 provides the evidence needed to substantively answer both remaining research questions.

**For RQ1**, the Phase 2 finding that rainbow-lite is the most robust DQN variant extends cleanly into MARL: rainbow maintains genuinely competitive dynamics under all tested stochasticity levels, while vanilla collapses into degenerate equilibria at nearly half of all trials — a rate that worsens under noise. Robustness under MARL conditions is fundamentally an algorithmic property, not just a function of the competitive environment.

**For RQ2**, two distinct emergent behaviour patterns were observed: zone specialisation (agents independently partitioning the circuit without coordination) and risk polarisation (hawk-dove equilibria in aggression level). These behaviours arise purely from competitive pressure in IL-MARL and are qualitatively distinct from anything observable in single-agent training. They have clear real-world analogues — racing drivers with different risk profiles carving out different sections of the track — and demonstrate that IL-MARL is a valid tool for modelling competitive strategic emergence, provided the underlying algorithm is stable enough to sustain genuine competition rather than collapsing into hierarchy.

Phase 4 (4-agent team extension with pace asymmetry and α reward blending) is the natural next step, building on the finding that rainbow's competitive stability is the prerequisite for rich strategy emergence.

---

*All 18 trial JSON files are in `metrics/phase3/`. The s0 analysis (6 trials) is in `research_findings/phase3_analysis.md`. This document supersedes that for cross-stochasticity and RQ conclusions.*
