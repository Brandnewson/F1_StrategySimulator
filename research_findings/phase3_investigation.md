# Phase 3 — Deep Investigation: MARL Mechanics, Degenerate Collapse, and A2 Advantage
**Date:** 2026-03-26
**Purpose:** Answer the three open questions from `phase3_full_analysis.md` before proceeding to Phase 4. Covers: (1) how MARL works in this simulator, (2) the root cause of vanilla's degenerate collapse, (3) the cause of rainbow's systematic A2 advantage at s2, and (4) additional tests to fully answer the standing hypotheses.

---

## Part 1 — How This Simulator Implements MARL

### 1.1 The Single Most Important Fact: No Information Is Shared

The two agents in Phase 3 do not share anything. Not weights. Not gradients. Not replay buffer contents. Not Q-values. Not target network states. Not even epsilon schedules (they are independent). Every component of each agent's learning machinery is entirely private.

This is not a limitation or a simplification. It is the defining property of Independent Learner MARL (IL-MARL), which is itself the correct methodological baseline for this dissertation. Understanding exactly what is and is not shared is essential to interpreting every result.

---

### 1.2 The Training Loop — What Happens in Each Episode

The full sequence for one training episode is:

```
Episode starts
│
├── race_reset()
│   └── Round-robin starting grid assignment:
│       ├── DQN Agent 1 gets grid position from cycle index
│       └── DQN Agent 2 gets the next position in the cycle
│       (Ensures ~equal P1/P2 starts over 500 episodes — fair by design)
│
├── Tick loop (runs until all drivers finish 5 laps)
│   │
│   └── Each tick:
│       ├── _update_driver(driver, dt)  ← moves each car forward
│       ├── update_driver_positions()   ← recalculates P1/P2 from track position
│       ├── _check_overtakes()
│       │   └── At each overtaking zone, for each driver behind another:
│       │       ├── DQN agent observes: [zone_distance, gap_to_ahead,
│       │       │   zone_difficulty, has_car_ahead, position_norm, laps_remaining]
│       │       ├── Selects action: HOLD / CONSERVATIVE / NORMAL / AGGRESSIVE
│       │       ├── Outcome resolved by stochastic probability function
│       │       └── Reward calculated and stored as pending transition
│       └── At decision resolution: store_transition_from_context()
│           └── Each agent stores (state, action, reward, next_state, done)
│               into ITS OWN replay buffer. The other agent never sees this.
│
├── Race ends → terminal bonus awarded to each agent
│
└── _train_dqn_agents()
    └── For EACH DQN driver independently:
        ├── 10 × train_step(batch_size=64)
        │   ├── Sample 64 transitions from OWN replay buffer
        │   ├── Compute TD targets using OWN Q-network and OWN target network
        │   ├── Backpropagate loss through OWN weights
        │   └── (rainbow only) Update priorities in OWN PER buffer
        └── on_episode_end() → decay OWN epsilon by 0.995
```

The key is the phrase "OWN" repeated throughout. At no point does Agent 1's training step read anything from Agent 2's buffer, weights, or gradients. The two `train_step()` calls are sequential (one completes before the other starts) but they are functionally independent — the ordering within the episode does not matter.

---

### 1.3 How the Agents "Teach" Each Other — Indirect Coupling Through Race Outcomes

Since no explicit information is shared, the teaching mechanism is entirely **implicit**. It operates through the shared race environment, not through any direct channel.

Consider what happens from Agent 1's perspective:

- Agent 1's observation vector at each decision point contains `gap_to_ahead_norm`, `has_car_ahead`, `current_position_norm`. All three are directly determined by where Agent 2 is on the track at that moment.
- If Agent 2 just successfully overtook Agent 1, then Agent 1 now observes: `has_car_ahead = 0`, `current_position_norm = 1.0` (last position), `gap_to_ahead_norm = 0` (no one ahead). Its incentive to attempt anything is altered.
- Conversely, if Agent 1's last attempt failed and Agent 2 is now 0.08 km ahead, Agent 1 sees a closing gap and a legitimate overtaking opportunity.

The "teaching" is entirely mediated by the race state. Agent 2's policy manifests as a set of physical positions on the track across episodes. Agent 1 experiences those positions as part of its environment and learns to respond to the patterns they create. Neither agent has access to a description of the other's intentions — only the consequences.

This is exactly analogous to how two human racing drivers learn from each other without telepathy: by observing what the other car does, inferring the pattern, and adapting their own approach. The learning is slower and more noisy than cooperative approaches (like MADDPG, which allows agents to observe each other's Q-values during training) but it produces genuinely independent policies.

---

### 1.4 The Observation Vector and What It Does and Does Not Contain

The six features available to each agent at decision time (low complexity profile):

| Feature | What it measures | Does it reveal anything about the opponent's strategy? |
|:--------|:----------------|:------------------------------------------------------|
| `zone_distance_norm` | How far to the next overtaking zone | No — purely about track geography |
| `gap_to_ahead_norm` | Gap in km to the car immediately ahead | Indirectly — reflects the opponent's speed and position |
| `zone_difficulty` | Difficulty rating of the upcoming zone | No — fixed track property |
| `has_car_ahead` | Binary: is there a car to overtake? | Indirectly — depends on whether the opponent is ahead |
| `current_position_norm` | Own finishing position, normalised | Indirectly — encodes relative standing against all other drivers |
| `laps_remaining_norm` | Race time remaining, normalised | No — race clock |

Crucially, the observation vector contains **no direct information about what the opponent agent is doing**. It does not contain:
- The opponent's epsilon value (exploration rate)
- The opponent's Q-values or policy outputs
- The opponent's last action
- The opponent's reward

The agents can only observe their own positions and gaps. The opponent's strategy becomes visible only through the physical consequences it produces — a gap that closes or widens, a position that changes, a zone that becomes contested or vacated.

This is why emergent zone specialisation (Finding 2 from Phase 3) is a genuine finding rather than a trivially explained result. Agent 1 did not observe "Agent 2 always attempts Raidillon"; it experienced that it kept losing position-for-position when it also tried Raidillon and learned to redirect its attention to zones where it got better outcomes.

---

### 1.5 The Starting Position System and Why It Matters

The `race_reset()` method assigns starting grid positions using a round-robin cycle:

```python
for dqn_driver in dqn_drivers:
    pos = all_positions[self._start_pos_cycle_idx % num_drivers]
    self._start_pos_cycle_idx += 1
```

Over 500 training episodes with 2 DQN agents, this produces approximately 250 episodes where A1 starts P1 and 250 where A2 starts P1. The cycle is deterministic and perfectly balanced across the training run.

This matters because **starting position has a direct causal effect on the training signal each agent receives**. The agent starting P1:
- Rarely has `has_car_ahead = 1` at the start of the race
- Receives fewer overtaking decision opportunities per race (the leader is rarely challenged at La Source)
- Accumulates more terminal reward from simply staying ahead

The agent starting P2:
- Immediately has `has_car_ahead = 1`
- Gets many more overtaking decisions per race
- Has more varied reward signals (successful and failed attempts)

In a perfectly fair alternating system, these effects average out over 500 episodes. But early in training, when random weight initialisation creates minor asymmetries, the agent that happens to win slightly more in the first dozen episodes begins to spend more time in P1 — accumulating a positional advantage that compounds over subsequent episodes. This compounding is the origin of both degenerate equilibria and competitive asymmetry.

---

## Part 2 — Why Vanilla Degenerates: The Full Mechanism

### 2.1 The Feedback Loop

Vanilla degenerate collapse is not a random failure. It is a deterministic outcome of a self-reinforcing feedback loop that operates through the replay buffer. The loop has four stages:

**Stage 1 — Initial asymmetry from random weights:**
At the start of training, both agents have random Q-values. For a given state (zone ahead, gap, difficulty), both agents will select actions that differ from each other because their initial weights differ. In some early episodes, Agent 1 happens to select an action that produces a better outcome than Agent 2's action in the same situation. This gives A1 a slight positional advantage — perhaps a 55/45 win rate over the first 30 training episodes.

**Stage 2 — Replay buffer divergence:**
After 30 episodes, Agent 1 has accumulated ~70% "winning transitions" in its replay buffer (races where it finished P1). These transitions are reinforcing: they contain positive outcome rewards. Agent 2 has accumulated ~70% "losing transitions" — races where it finished P2 and its overtaking attempts at difficult zones produced negative tactical rewards.

With vanilla's uniform replay, Agent 2 samples these losing transitions uniformly. The Q-value update for HOLD consistently receives less negative signal than ATTEMPT (since the losing positions are often because of failed attempts at high-difficulty zones). Gradually, Q(HOLD) edges above Q(ATTEMPT) in Agent 2's network at those zones.

**Stage 3 — The hold policy locks in:**
Once Q(HOLD) > Q(ATTEMPT) at a zone, the agent selects HOLD. It never generates a new ATTEMPT transition at that zone. Its replay buffer gets no new successful attempt data for that zone. With Q(ATTEMPT) decaying and never being reinforced by new experience, the gap between Q(HOLD) and Q(ATTEMPT) widens. This is self-reinforcing: not attempting means no new positive attempt data; no new positive data means Q(ATTEMPT) only receives negative or stale updates; stale updates reinforce the HOLD preference.

**Stage 4 — Full collapse:**
Once HOLD becomes dominant at multiple zones, the agent never again generates attempts at those zones. Q(HOLD) is updated positively (the agent never falls further behind from not attempting) and Q(ATTEMPT) continues to decay from stale negative data. Within 100–150 training episodes of Stage 3 beginning, Q(HOLD) dominates at all 9 zones. The agent has settled into a permanent passive follower.

**Why this does not happen to rainbow:** Rainbow's PER assigns priority proportional to TD error magnitude. A failed attempt at a hard zone produces a large TD error (expected reward was, say, +0.05; actual reward was −0.15; |TD error| = 0.20). This transition is replayed many times. The agent keeps learning about the failure rather than letting it fade from uniform sampling. More importantly, when a rare success at that zone occurs (stochasticity means some attempts succeed even in unfavourable conditions), the positive surprise also produces a high TD error and is immediately over-sampled. The Q-value estimate for ATTEMPT remains calibrated to the true distribution of outcomes, preventing HOLD from permanently dominating.

The n-step return compounds this: three consecutive sub-optimal decisions produce a larger negative discounted return than one, making the replay signal for bad attempt chains more corrective than single-step would suggest. This makes bad period of training more recoverable.

### 2.2 What the Data Shows as Evidence for This Mechanism

In vanilla s1 s202 (degenerate-adjacent, WR=0.587):
- A2 risk: CONSERVATIVE=0, NORMAL=0, AGGRESSIVE=287 (100% aggressive)
- A2 zone_behavior: La Source = 750 decisions, 0 attempts
- This is a partial collapse: A2 has learned Q(HOLD) > Q(ATTEMPT) at La Source (750 encounters, zero attempts) but still attempts at Raidillon and Les Combes

This is Stage 3 of the feedback loop caught mid-progress. La Source, with its low difficulty (0.2), is the first zone any agent colonises because it is the most reliably rewarding. Once the leader holds La Source through consistent wins, the follower's La Source attempt data becomes dominated by failures (because the leader is ahead and A2's gap is usually too large). The Q(HOLD) > Q(ATTEMPT) transition at La Source occurs first, then propagates to other zones over further episodes.

In fully degenerate trials (all at WR=1.000), this process has completed: A2 shows exactly 750 decisions and 0 attempts at every zone — the maximum possible decisions (9 zones × 5 laps × 150 evaluation races) with zero attempts. The loop has run to completion.

### 2.3 Why Stochasticity Makes It Worse

At s2 (noise_std=0.05 on the probability function), the outcome of any given attempt is less predictable. An agent attempting at Raidillon (difficulty=0.9) might succeed sometimes even when the probability is theoretically 0.05. But it will also fail when the probability should have been higher. Under high noise:

- Successful attempts generate unexpectedly large positive TD errors (surprised by success) — but vanilla's uniform sampling means these are diluted by the large volume of negative data
- Failed attempts generate unexpectedly large negative TD errors — and these are also uniformly sampled

The net effect for vanilla is that the signal-to-noise ratio of the replay buffer worsens under s2. The agent cannot reliably distinguish "this zone is genuinely bad" from "this zone had a bad run of noise". It tends to over-weight the failures (which are more numerous) and under-weight the successes (which are rarer and uniformly sampled). This accelerates the HOLD convergence, explaining why the degenerate rate doubles at s2 (67% vs 33% at s0 and s1).

---

## Part 3 — Why Rainbow's Agent 2 Has a Systematic Advantage at S2

### 3.1 The Observation

In all three rainbow s2 trials, A2 outperforms A1:

| Trial | A1 WR | Interpretation |
|:------|:------|:--------------|
| s2 s101 | 0.220 | A2 dominates strongly |
| s2 s202 | 0.467 | A2 slight advantage |
| s2 s303 | 0.513 | Near parity, A2 marginal edge |

This is a consistent directional pattern (A2 ≥ A1 all three times) that does not appear at s0 or s1 for rainbow. The question is whether A2's advantage is:
- **(a)** A genuine training asymmetry — A2 learns a better policy at high noise
- **(b)** A positional artefact — the round-robin starting position system gives A2 a systematic training advantage under s2 conditions
- **(c)** Statistical noise — the three-seed sample is too small to distinguish from chance

### 3.2 The Two Candidate Mechanisms

**Mechanism A — PER self-reinforcement from behind:**
Under s2 high noise, the agent that starts P2 more often generates higher-TD-error transitions because:
1. Starting P2 → more overtaking decisions per race → more varied outcomes → higher average |TD error|
2. PER over-samples these high-TD-error transitions → faster Q-value update
3. Faster Q-value update → better catch-up policy → more races won from behind

If A2 happened to spend slightly more episodes behind in the early phases of training (due to random initialisation asymmetry in the opposite direction at s2 vs s0), this mechanism predicts A2 develops a better behind-position policy. The high noise at s2 amplifies this because it increases TD error magnitudes for both success and failure surprises.

**Mechanism B — Round-robin grid asymmetry under S2:**
The starting grid cycle assigns positions deterministically. With 500 training episodes and 2 agents, the assignments are perfectly alternated. But consider what happens to the LEARNING signal at P1 vs P2 under s2:

At P1 (leading): very few overtaking decisions, mostly maintenance behaviour. The terminal bonus (finishing P1) is large but not varied — it is +2.0 regardless of how the agent achieved P1. Low TD error, low PER priority at s2.

At P2 (trailing): many overtaking decisions, diverse outcomes under s2 noise. Some surprises from stochasticity, high |TD error|. PER magnifies these.

If A2's label causes it to be assigned the starting position second in the round-robin cycle, it gets P2 in slightly more early-training episodes where positional hierarchy has not yet been established. If those early episodes compound (early P2 → more learning → slight edge → early P1 in later episodes → terminal bonus → advantage grows), A2 could systematically end up better trained by episode 500 at s2 where the P2-learning boost is maximal.

### 3.3 How to Resolve This — The Cross-Play Experiment

The diagnostic test is straightforward: **evaluate trained A1 against trained A2 with all starting positions deliberately swapped.** Run 150 evaluation races where A1 always starts P2 and A2 always starts P1, plus 150 where positions are randomised normally. Compare win rates.

If A2's advantage persists even when A1 is given P1:
→ The advantage is intrinsic to A2's learned policy (Mechanism A confirmed)
→ A2 has genuinely learned a better high-noise policy

If A2's advantage disappears or reverses when positions are swapped:
→ The advantage is a positional artefact (Mechanism B confirmed or partially confirmed)
→ The policy quality is symmetric; starting position drives the outcome

If results are mixed (advantage partially survives position swap):
→ Both mechanisms contribute
→ A2 has a slight genuine policy edge amplified by starting position effects

This cross-play test requires one new script that loads the saved model checkpoints from the s2 trials and runs evaluation with forced starting positions. This is discussed in Part 4.

---

## Part 4 — Additional Tests to Fully Answer the Standing Hypotheses

### 4.1 Summary of Open Hypothesis Questions

| Hypothesis | Current status | What is still open |
|:-----------|:--------------|:-------------------|
| **H_ns1** — 500 episodes insufficient for vanilla convergence | Confirmed for vanilla; confirmed adequate (barely) for rainbow at s0–s1 | Does 750 episodes resolve vanilla's collapse? Is rainbow sufficient at s2? |
| **H_ns2** — Rainbow stability transfers to MARL | Strongly confirmed: 0% degenerate vs 44% | Is the A2 advantage at s2 a training artefact or genuine? |
| **H_diff** — Agents spontaneously differentiate | Confirmed for risk (100% non-degenerate trials); confirmed for zones at s0–s1 | Why does rainbow zone differentiation collapse at s2? Is it recoverable with more training? |

---

### 4.2 Test Battery 1 — Vanilla Degenerate Collapse Investigation

**Purpose:** Determine whether vanilla collapse is a budget failure (fixable with more training) or an architectural failure (fundamental to uniform replay in MARL).

**Test V1 — Vanilla at 750 training episodes:**
Run the same 9 vanilla trials (3 seeds × 3 stochasticity levels) with `--train-runs 750`. Compare degenerate rate:
- If degenerate rate drops from 44% to < 15%: collapse is budget-dependent. Report that vanilla needs 750+ episodes for MARL and proceed with that budget.
- If degenerate rate stays above 30%: vanilla has a structural MARL failure from uniform replay. This is itself a publishable finding — vanilla DQN should not be used as an IL-MARL agent at this training scale.

Commands:
```bash
# s0 seeds
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 750 --eval-runs 150 --train-seed 101 --eval-seeds 101 --stochasticity-level s0 --out metrics/phase3/vanilla_750_s0_s101.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 750 --eval-runs 150 --train-seed 303 --eval-seeds 303 --stochasticity-level s0 --out metrics/phase3/vanilla_750_s0_s303.json

# s2 seeds (where collapse is worst)
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 750 --eval-runs 150 --train-seed 101 --eval-seeds 101 --stochasticity-level s2 --out metrics/phase3/vanilla_750_s2_s101.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 750 --eval-runs 150 --train-seed 303 --eval-seeds 303 --stochasticity-level s2 --out metrics/phase3/vanilla_750_s2_s303.json
```

These 4 trials target the 4 degenerate cases and will directly answer whether budget is the cause.

**Test V2 — Replay buffer size inspection:**
After a degenerate trial completes training, the ratio of HOLD to ATTEMPT transitions in A2's replay buffer directly evidences the imbalance mechanism. This requires adding a diagnostic counter to `evaluate_marl.py` (or parsing the saved training logs) to compute: for A2, what fraction of stored transitions have `action_idx == 0` (HOLD)? If > 70% HOLD in a degenerate trial vs < 50% in a valid trial, the mechanism is confirmed without needing a structural change.

---

### 4.3 Test Battery 2 — Rainbow A2 Advantage Investigation

**Purpose:** Determine whether rainbow A2's s2 advantage is a genuine policy edge or a positional/training artefact.

**Test R1 — Cross-play evaluation (position swap):**
This uses the saved model checkpoints from the rainbow s2 trials. Write a small evaluation script (`scripts/evaluate_crossplay.py`) that:
1. Loads the saved A1 weights from a rainbow s2 trial (e.g., `models/DQN_A1_rainbow_lite_trained.pth`)
2. Loads the saved A2 weights from the same trial
3. Runs 150 evaluation races with normal (round-robin) starting positions — records A1 WR
4. Runs 150 evaluation races with FORCED positions: A1 always starts P1 — records A1 WR
5. Runs 150 evaluation races with FORCED positions: A2 always starts P1 — records A2 WR

If `A2 WR when forced to start P1` ≈ `A2 WR normally` → policy edge (Mechanism A)
If `A2 WR when forced to start P2` >> `A2 WR when forced to start P1` → positional artefact (Mechanism B)

**Test R2 — Rainbow s2 extended seed set:**
The 3-seed observation that A2 consistently wins is suggestive but not statistically conclusive (p-value for 3 consistent results under null is 0.125 — not significant). Run 2 additional seeds (404, 505) at rainbow s2 to increase the sample:

```bash
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 404 --eval-seeds 404 --stochasticity-level s2 --out metrics/phase3/rainbow_marl_s2_s404.json

conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 505 --eval-seeds 505 --stochasticity-level s2 --out metrics/phase3/rainbow_marl_s2_s505.json
```

With 5 seeds at s2: if 5/5 show A2 ≥ A1, the binomial probability under the null hypothesis is 0.5^5 = 0.031 — statistically significant at p < 0.05.

**Test R3 — Rainbow at 750 episodes for s2:**
Separately, test whether rainbow at 750 training episodes resolves the A2 advantage at s2 (because both agents have had more time to converge to symmetric equilibria):

```bash
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 750 --eval-runs 150 --train-seed 101 --eval-seeds 101 --stochasticity-level s2 --out metrics/phase3/rainbow_750_s2_s101.json
```

If A2's advantage disappears at 750 episodes: the advantage is a transitional artefact of the training process, not a stable policy property. Both agents converge to symmetric equilibria given enough time.

---

### 4.4 Test Battery 3 — Zone Differentiation Collapse at S2

**Purpose:** Determine whether rainbow's zone differentiation collapse at s2 (mean 0.091 vs 0.268 at s1) is signal-noise driven (fixable with more training or larger buffers) or reflects a genuine inability to sustain zone-specific learning under s2-level noise.

**Test Z1 — Rainbow at 750 episodes, s2:**
This shares trials with R3 above. Examine `zone_differentiation_index` in the 750-episode s2 output. If zone_diff recovers above 0.20, the collapse is training-time limited. If it stays below 0.10, s2 noise is genuinely too high for zone-specific policy learning.

**Test Z2 — Intermediate stochasticity s1.5:**
s1 has `probability_noise_std=0.02` and s2 has `0.05`. A custom s1.5 level at `0.035` would locate the threshold at which zone differentiation begins to collapse. This requires adding a new stochasticity level to the config:

```json
"s1_5": {
    "base_probability_scale": 1.0,
    "risk_modifier_scale": 1.0,
    "gap_modifier_scale": 1.0,
    "probability_noise_std": 0.035,
    "min_success_probability": 0.02,
    "max_success_probability": 0.95
}
```

Rationale: if zone_diff collapses sharply between s1 and s2, there is a threshold somewhere in that noise range. Identifying it would tell you the precise noise tolerance of zone-specific Q-value learning — a theoretically interesting finding about the limits of tabula-rasa IL-MARL in noisy environments.

---

### 4.5 Test Battery 4 — Cross-Algorithm Matchup

**Purpose:** Test whether the emergent zone specialisation and hawk-dove patterns are specific to same-algorithm competition or general to any two IL-MARL learners.

**Test C1 — Vanilla A1 vs Rainbow A2:**
This asymmetric matchup directly answers whether rainbow's stability comes from its algorithm properties alone or requires the opponent to also be stable. If rainbow A2 learns a strong policy even against vanilla A1's erratic behaviour, it demonstrates that rainbow's PER mechanism is robust to opponent policy instability.

This requires a new config (`config_mixed_marl.json`) where Agent 1 uses `algo: "vanilla"` and Agent 2 uses `algo: "rainbow_lite"`.

Expected outcome: Rainbow A2 should win most races (its more stable policy should dominate vanilla A1's erratic training). But the interesting question is: does vanilla A1 collapse into a degenerate HOLD policy against a rainbow opponent, or does rainbow's more active and varied behaviour create a richer environment that keeps vanilla competitive?

---

### 4.6 Priority Ordering

| Priority | Test | Runs needed | Key question it answers |
|:---------|:-----|:-----------|:------------------------|
| 1 | V1 — Vanilla 750 episodes | 4 | Is vanilla's collapse fixable with budget? |
| 2 | R2 — Rainbow s2 seeds 404, 505 | 2 | Is A2 advantage statistically significant? |
| 3 | R1 — Cross-play position swap | 1 script + 3 eval runs | Is A2 advantage genuine policy or artefact? |
| 4 | R3/Z1 — Rainbow 750 s2 | 3 | Does more training resolve s2 issues? |
| 5 | C1 — Mixed vanilla/rainbow | 3 | Are emergent behaviours algorithm-specific? |
| 6 | Z2 — s1.5 stochasticity | 3 | Where is the zone differentiation threshold? |

---

## Part 5 — Implications for the Dissertation

### 5.1 What These Investigations Give Chapter 4

**If V1 shows vanilla still degenerates at 750 episodes:** This becomes a strong finding for RQ3. Vanilla DQN is not just less reliable than rainbow in single-agent settings (Phase 2) — it is architecturally unsuited to IL-MARL due to uniform replay. The dissertation can argue that prioritised experience replay is not merely a performance enhancement in single-agent RL but a **structural requirement** for stable multi-agent training. This links to the broader MARL literature on non-stationarity mitigation and would be a novel empirical contribution.

**If R1 shows A2's advantage is genuine (policy, not position):** This becomes a finding about PER learning dynamics in competitive settings. The agent that faces more adversity during training (more behind, more high-TD-error surprises) learns a stronger catch-up policy — an "underdog advantage" from PER's prioritisation scheme under high noise. This links to curriculum learning theory and the observation that harder training environments can produce more generalisable policies.

**If R1 shows it is a positional artefact:** This is still important — it exposes a subtle bias in the evaluation methodology and argues for forced-symmetric evaluation procedures in future MARL work.

**If C1 shows vanilla collapses against rainbow:** This confirms that vanilla's instability is exacerbated rather than resolved by a more capable opponent. A stronger opponent accelerates the replay buffer imbalance. This is theoretically coherent and has practical implications: naive IL-MARL agents should not be paired against significantly more capable agents during training.

### 5.2 What Remains Regardless of Investigation Outcomes

Irrespective of the above findings, the following Phase 3 results stand firm and are sufficient to answer RQ1 and RQ2 for the dissertation:

- **RQ1**: Rainbow MARL maintains competitive dynamics under all stochasticity levels. Vanilla does not at s2.
- **RQ2**: Zone specialisation (territorial partitioning) and risk polarisation (hawk-dove equilibria) emerged independently in rainbow IL-MARL without any coordination mechanism.
- The 500-episode training budget is sufficient for rainbow at s0–s1 and marginal at s2.
- Vanilla's degenerate collapse is a real phenomenon with a mechanistically explained cause (uniform replay buffer imbalance under positional asymmetry), regardless of whether it is fixable with more budget.

The investigations above strengthen and nuance these answers. They are not prerequisites for the dissertation's core claims.

---

*Recommended next step: Run V1 (4 trials) and R2 (2 trials) first — these are the two tests with highest impact-to-compute ratio. Results from those 6 trials will inform whether the full test battery is needed or whether the evidence is sufficient for the dissertation's purposes.*
