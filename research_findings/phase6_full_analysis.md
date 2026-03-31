# Phase 6 Full Analysis — Multi-Team Intra-Team Dynamics

**Date:** 2026-03-31
**Algorithm:** rainbow-lite (IL-MARL)
**Complexity profile:** `low_marl_teams` (4 DQN agents in 2 teams + 1 Base adversary)
**Budget per trial:** 500 training episodes, 150 evaluation races (balanced starting positions)
**Seeds:** 101, 202, 303
**Stochasticity levels:** s0, s1, s2
**Total trials:** 36 (4 conditions x 9 trials each)
**Primary metric:** `team_a_vs_team_b_win_rate` (best-finisher comparison between teams)

---

## 1. Experimental conditions

| Condition | Team A alpha | Team B alpha | Purpose |
|-----------|-------------|-------------|---------|
| Baseline | 0.0 | 0.0 | No cooperation, competitive reference |
| Symmetric moderate | 0.50 | 0.50 | Both teams moderately cooperative |
| Symmetric strong | 0.75 | 0.75 | Both teams at Phase 5's optimal alpha |
| Asymmetric | 0.75 | 0.0 | Does the cooperative team beat the competitive team? |

---

## 2. Cross-condition aggregate table

| Condition | TA win rate | TA avg pos | TB avg pos | TA beat Base | TB beat Base | Base pos | Intra-A | Intra-B | ZD | RD | Collapse |
|-----------|-----------|-----------|-----------|-------------|-------------|---------|---------|---------|----|----|----------|
| 0.0 vs 0.0 | 0.496 | 2.927 | 2.916 | 0.353 | 0.346 | 3.315 | -0.133 | -0.116 | 0.122 | 0.096 | 0/9 |
| 0.50 vs 0.50 | 0.456 | 3.013 | 2.907 | 0.296 | 0.316 | 3.161 | -0.116 | -0.127 | 0.163 | 0.066 | 0/9 |
| 0.75 vs 0.75 | 0.484 | 3.055 | 2.967 | 0.236 | 0.250 | **2.956** | -0.122 | -0.109 | 0.195 | 0.104 | 0/9 |
| 0.75 vs 0.0 | 0.475 | **3.116** | **2.816** | 0.245 | 0.338 | 3.136 | -0.152 | -0.168 | 0.126 | 0.087 | 0/9 |

---

## 3. The headline finding: cooperation hurts

**Phase 5 showed alpha=0.75 produced an 83% improvement in joint-beat-base rate against a fixed adversary. Phase 6 shows the same alpha produces a 31% decline in the team setting.**

Average team both-beat-base rate across conditions:
```
0.0 vs 0.0:   0.350  (baseline)
0.50 vs 0.50: 0.306  (-12%)
0.75 vs 0.75: 0.243  (-31%)
0.75 vs 0.0:  0.291  (-17%)
```

Every form of cooperation tested in Phase 6 produces WORSE joint performance against the Base adversary than the purely competitive baseline. This is the opposite of Phase 5.

The Base agent's average finishing position tells the same story from the other direction:
```
0.0 vs 0.0:   3.315  (Base finishes near last)
0.50 vs 0.50: 3.161
0.75 vs 0.75: 2.956  (Base finishes near THIRD — ahead of most DQN agents)
0.75 vs 0.0:  3.136
```

As teams cooperate more, the Base agent climbs the field. At alpha=0.75 vs 0.75, the Base agent finishes in position 2.96 on average (in a 5-driver race). It is beating multiple DQN agents per race.

---

## 4. The asymmetric condition: the cooperative team loses

In the headline experiment (Team A alpha=0.75 vs Team B alpha=0.0):

- **Team A win rate: 0.475** (wins only 2 of 9 trials, ties 2)
- **Team A avg position: 3.116** (behind Team B's 2.816)
- **Team A both-beat-base: 0.245** (30.6% worse than baseline)
- **Team B both-beat-base: 0.338** (2.4% worse than baseline)

The cooperative team does not merely fail to gain an advantage. It actively performs worse than the competitive team, and worse than either team in the fully competitive baseline. The cooperative incentive is a liability in the multi-team setting.

Per-stochasticity breakdown for the asymmetric condition:

| Stoch | TA win rate | TA beat Base | TB beat Base | Base pos |
|-------|-----------|-------------|-------------|---------|
| s0 | 0.516 | 0.271 | 0.329 | 3.171 |
| s1 | 0.478 | 0.247 | 0.367 | 3.164 |
| s2 | 0.431 | 0.218 | 0.318 | 3.073 |

Team A's disadvantage grows with stochasticity. At s2, Team A wins only 43.1% of team-vs-team comparisons and beats Base only 21.8% of the time.

---

## 5. Why does cooperation hurt? Root cause analysis

### 5.1 The credit assignment problem

In Phase 5 (2 DQN + 1 Base), the reward formula at alpha=0.75 was:
```
reward_i = 0.25 * own_delta + 0.75 * teammate_delta
```

The agent received 75% of its reward from its single teammate's outcome. In a 3-driver race, the agent's zone-level decisions DIRECTLY affected the teammate's finishing position because there were only 3 competitors. If Agent 1 overtook the Base agent, that created space for Agent 2 to also improve. The shared reward signal had a clean causal pathway: your decisions → teammate's outcome → your reward.

In Phase 6 (4 DQN + 1 Base), the same formula applies within each team. But the teammate's outcome is now determined by the teammate's decisions, the two opposing agents' decisions, and the Base agent's behaviour. The agent's own decisions have a much weaker causal influence on the teammate's finishing position. At alpha=0.75, 75% of the reward signal comes from a source the agent can only weakly influence.

This is the multi-agent credit assignment problem. The shared reward signal has become noise rather than a coordination mechanism. Q-value updates from the teammate's outcome provide no useful gradient for zone-level decision-making because the agent cannot attribute changes in teammate outcome to its own actions.

### 5.2 The public goods dilemma

From a game theory perspective, the intra-team reward sharing creates a public goods problem. Each agent's cooperative effort (positioning itself to help the teammate) benefits the team but is diluted by external factors. The Nash equilibrium in a public goods game with imperfect influence is to under-contribute. This is precisely what we observe: cooperative teams develop less aggressive overtaking strategies (lower attempt rates at difficult zones in deterministic conditions), producing fewer successful overtakes and worse finishing positions.

Evidence from the asymmetric s0 trials: Team A (alpha=0.75) Agent 1 attempts Raidillon at 27.0% vs the baseline of 42.3%. The cooperative signal teaches the agent that aggressive attempts at difficult zones are risky for the team (if you fail and lose position, your teammate's reward is diluted). The agent learns caution. But caution against 4 other drivers means you lose ground.

### 5.3 Cooperative effort is directed at the wrong target

In Phase 5, the only way to improve both DQN agents' outcomes was to push the Base agent down. The cooperative incentive aligned perfectly with the anti-Base objective.

In Phase 6, the cooperative incentive within a team can be satisfied by either:
(a) Pushing the Base agent down (helps both teams)
(b) Pushing the opposing team down (helps only your team)

The reward signal does not distinguish between these. When your teammate overtakes an opposing agent, your blended reward increases. This teaches you that your teammate doing well is good, but it provides no learning signal about whether to help the teammate against Base or against the opponents. The alpha signal is agnostic about the source of the teammate's improvement.

The result: cooperative teams spend their coordination effort on inter-team positioning rather than anti-Base performance. But inter-team positioning is a zero-sum game between teams, so the cooperative effort produces no net benefit. Meanwhile, the attention diverted from anti-Base performance causes Base to climb the field.

### 5.4 The training budget hypothesis

With 4 DQN agents each learning simultaneously, the non-stationarity of the environment is substantially worse than in Phase 5 (2 DQN + 1 fixed Base). Each agent's environment changes as 3 other agents adapt. At 500 training episodes, the cooperative signal may not have had sufficient time to produce stable coordination. The Q-values may be undertrained, and the cooperative gradient (noisy to begin with) may need more episodes to emerge as a useful signal.

However, the competitive baseline (alpha=0.0 vs 0.0) at the same 500-episode budget produces coherent, interpretable strategies (both teams near position parity, reasonable beat-base rates). This suggests that 500 episodes is sufficient for the competitive case. The cooperative case may genuinely require more training, but the burden of proof is on the cooperative signal: if it requires substantially more training to produce ANY benefit, that is itself a practical limitation of the approach.

---

## 6. Intra-team cooperation never emerges

The intra-team cooperation metric (Pearson correlation between teammates' position deltas) is negative or near-zero across ALL 36 trials:

| Condition | Mean Intra-A | Mean Intra-B |
|-----------|-------------|-------------|
| 0.0 vs 0.0 | -0.133 | -0.116 |
| 0.50 vs 0.50 | -0.116 | -0.127 |
| 0.75 vs 0.75 | -0.122 | -0.109 |
| 0.75 vs 0.0 | -0.152 | -0.168 |

At alpha=0.0, the negative correlation is expected: teammates compete for the same positions, so when one gains, the other tends to lose. But at alpha=0.75, the same anti-correlation persists. The reward sharing signal does not produce correlated improvements between teammates. Not once across 36 trials does the intra-team cooperation metric turn positive for either team.

This is the strongest evidence that the reward sharing mechanism fails in the 4-agent setting. Even at its strongest (alpha=0.75), the shared incentive cannot overcome the structural anti-correlation between teammates competing for the same finite set of positions against three other agents.

---

## 7. What DOES change with cooperation

### 7.1 Zone differentiation increases

| Condition | Mean ZD |
|-----------|---------|
| 0.0 vs 0.0 | 0.122 |
| 0.50 vs 0.50 | 0.163 |
| 0.75 vs 0.75 | 0.195 |
| 0.75 vs 0.0 | 0.126 |

Consistent with Phases 4 and 5: higher alpha produces more zone specialisation. But this specialisation does not translate into improved performance. The agents differentiate zones but this differentiation is not productive because the 5-agent game provides insufficient positional feedback for zone-level coordination to help.

### 7.2 Zero degenerate collapse

All 36 trials produce competitive outcomes (WR between 0.38 and 0.63). No collapse. The 5-agent non-zero-sum structure completely prevents the passive Nash equilibrium, consistent with Phase 5. The Base adversary is a reliable collapse preventor regardless of team structure.

### 7.3 Team-aware observations produce no visible team-conditional behaviour

Agents received `is_teammate_ahead` and `is_teammate_behind` features. If these features were useful, we would expect to see differentiated attempt rates when a teammate is ahead vs when an opponent is ahead. The data shows no such pattern. Team A agents in the asymmetric condition (alpha=0.75) attempt at roughly the same rates regardless of whether the car ahead is a teammate or opponent. The team-aware observations are present in the state vector but the agents do not appear to have learned to use them.

This is consistent with the credit assignment diagnosis: if the shared reward signal is too noisy to produce useful Q-value gradients, additional observational features about teammates do not help because the problem is in the reward structure, not the observation space.

---

## 8. Research question implications

### RQ1: What patterns of strategic behaviour emerge when independent RL agents train concurrently?

**Phase 6 extension:** In the 5-agent team game, the dominant behavioural pattern is competitive self-interest. Agents develop moderate zone differentiation and symmetric risk profiles across all conditions. The team structure (same-team vs opposing-team) does not produce visibly different behavioural clusters. The 5-agent game is too crowded for the territorial patterns seen in the 2-3 agent games to emerge clearly.

Attempt counts are substantially higher (1500-2200 per agent per trial vs 300-800 in Phase 5), confirming that more zones are active with more competitors. This validates the prediction that the 5-agent game would alleviate the zone utilisation limitation.

### RQ2: Is there a threshold beyond which shared incentive destabilises rather than improves learning?

**Phase 6 extension:** The threshold is lower than expected and depends on game structure. In the 2+1 game (Phase 5), the optimal alpha was 0.75. In the 4+1 game (Phase 6), even alpha=0.50 degrades performance relative to baseline. The cooperative signal never becomes beneficial in the team-vs-team setting — the entire alpha-performance curve is below the competitive baseline.

This means the stability threshold is not a fixed property of the alpha value. It is a function of (alpha) x (number of agents the teammate competes against) x (the agent's influence over its teammate's outcome). As the game grows more complex and the agent's influence over its teammate becomes more diluted, the threshold shifts toward zero.

### RQ3: Does shared incentive produce genuine cooperative advantage?

**Phase 6 answer: No, not in the multi-team IL-MARL setting.**

Phase 5 showed cooperative advantage against a fixed adversary (positive answer to RQ3). Phase 6 shows cooperative disadvantage against adaptive opponents (negative answer for the team extension). The combined finding is that cooperative reward sharing under IL-MARL produces genuine benefit only when:
1. The game is non-zero-sum (Phase 5 vs Phase 4)
2. The agent has strong influence over its teammate's outcome (Phase 5 vs Phase 6)
3. The adversary is not itself a learning agent (Phase 5 vs Phase 6)

When any of these conditions is violated, the shared reward signal becomes noise rather than a coordination mechanism.

---

## 9. Comparison with Phase 5

| Metric | Phase 5 (2 DQN + Base) alpha=0.75 | Phase 6 (4 DQN + Base) alpha=0.75 symmetric |
|--------|-----------------------------------|---------------------------------------------|
| Both-beat-base rate | **0.567** (+83% vs baseline) | **0.243** (-31% vs baseline) |
| Base avg position | 2.409 (pushed toward last) | 2.956 (near middle of field) |
| Collapse rate | 0/9 | 0/9 |
| Intra-team cooperation | N/A (only 2 agents) | -0.116 (anti-correlated) |

The reversal is complete. The same alpha value that produced the strongest cooperative performance in Phase 5 produces the weakest performance in Phase 6. This is the central scientific finding of Phase 6: **the effectiveness of reward sharing is not a property of alpha alone but of the interaction between alpha and game complexity.**

---

## 10. Mechanistic model

```
Phase 5 success:
  alpha=0.75 + 2 DQN agents + 1 Base
  → agent can DIRECTLY influence teammate outcome
  → shared reward has high signal-to-noise ratio
  → Q-values converge to zone differentiation
  → joint performance improves

Phase 6 failure:
  alpha=0.75 + 4 DQN agents + 1 Base
  → agent has WEAK influence over teammate outcome (diluted by 3 other agents)
  → shared reward has low signal-to-noise ratio
  → Q-values receive noisy cooperative gradient
  → agents learn caution (lower attempt rates at difficult zones)
  → caution against 4 opponents = losing ground
  → joint performance degrades
```

The causal variable is the ratio of influence-over-teammate-outcome to reward-weight-on-teammate-outcome. At alpha=0.75 in a 3-agent game, this ratio is high (you are one of two agents affecting positions). At alpha=0.75 in a 5-agent game, this ratio is low (you are one of four agents affecting positions). The cooperative signal becomes pure noise when this ratio falls below a critical level.

---

## 11. Threats to validity

### 11.1 Training budget
500 episodes with 4 concurrent learners produces substantially higher non-stationarity than 500 episodes with 2 learners. The cooperative signal may require more training to produce coordination. A training budget ablation (1000 or 2000 episodes) would determine whether the negative result is a convergence issue or a fundamental limitation.

### 11.2 Team size
Only 2-agent teams were tested. A larger team (3 per team) might change the dynamics. The 2-agent team is the minimum team size and may not generalise to larger teams.

### 11.3 Fixed team composition
All 4 DQN agents are identical rainbow-lite networks. Heterogeneous teams (different architectures, or pre-trained specialist agents) might produce different coordination patterns.

### 11.4 Terminal-only cooperative signal
Alpha sharing is applied only to the terminal outcome component. Denser cooperative feedback (per-step shared reward for zone-level decisions) might produce stronger coordination than the sparse end-of-race signal.

### 11.5 IL-MARL without coordination mechanisms
No centralised critic, no communication channel, no shared gradients. CTDE methods (MADDPG, QMIX) that train with global state information might succeed where IL-MARL fails. The Phase 6 result characterises IL-MARL specifically, not cooperative MARL in general.

### 11.6 Zone utilisation improvement is untested as a confound
The 5-agent game activates more zones (1500-2200 attempts vs 300-800 in Phase 5). This changes the strategic landscape. It is possible that the cooperation failure is partly due to the richer decision space overwhelming the cooperative signal, rather than purely the multi-agent credit assignment problem.

---

## 12. Open questions for future investigation

1. **Training budget ablation.** Does cooperation become beneficial at 1000 or 2000 training episodes in the 4-agent game? This would distinguish convergence issues from fundamental limitations.

2. **Lower alpha in the team setting.** Phase 6 tested alpha=0.50 and 0.75, both of which hurt. Would a much lower alpha (0.10 or 0.15) provide a mild coordination benefit without the noisy gradient problem?

3. **Per-step reward sharing.** Apply alpha not just to the terminal outcome but to the per-step positional component. This would give agents cooperative feedback at every zone decision, not just at race end.

4. **CTDE methods.** Would QMIX or MADDPG with access to all agents' states during training produce genuine team coordination where IL-MARL fails?

5. **Asymmetric team alpha within a team.** Give one teammate alpha=0.75 (the "team player") and the other alpha=0.25 (the "star driver"). Does asymmetric intra-team cooperation produce role differentiation that symmetric cooperation does not?

6. **Remove the Base agent.** Run 2v2 without Base. Does the result change when the game is zero-sum between teams? (Expected: possibly worse, given Phase 4's zero-sum collapse finding.)

---

## 13. Conclusions

**C1. Intra-team reward sharing under IL-MARL does not produce cooperative advantage in the multi-team setting.** All cooperative conditions (alpha > 0) produce worse joint-beat-base performance than the purely competitive baseline. The cooperative team loses to the competitive team in the asymmetric condition (2 wins, 2 ties, 5 losses out of 9 trials). This is the opposite of Phase 5's finding.

**C2. The mechanism of failure is diluted credit assignment.** In a 5-agent race, each agent's influence over its teammate's finishing position is weak relative to the combined influence of three other competitors. At alpha=0.75, 75% of the reward signal comes from a source the agent cannot meaningfully control. The resulting Q-value gradient is noise, producing less aggressive and less effective strategies.

**C3. The Base agent climbs the field as teams cooperate.** Base finishes at position 3.315 (near last) under competition, but at 2.956 (near third) under symmetric alpha=0.75. The cooperative teams direct their effort at inter-team positioning rather than anti-Base performance, allowing the fixed-policy adversary to exploit the gap.

**C4. The effectiveness of reward sharing is a function of game complexity, not alpha alone.** Phase 5 showed alpha=0.75 is optimal in a 3-agent game. Phase 6 shows the same alpha is harmful in a 5-agent game. The critical variable is the signal-to-noise ratio of the shared reward: how much the agent's own actions influence the teammate's outcome relative to how much reward weight is placed on that outcome. When influence is diluted by additional agents, the cooperative signal becomes noise.

**C5. Genuine intra-team cooperation did not emerge under any condition.** The intra-team correlation metric remained negative (anti-correlated) across all 36 trials. Teammates' fortunes remain antagonistic even at alpha=0.75. Team-aware observations (is_teammate_ahead, is_teammate_behind) did not produce observable team-conditional behaviour. The IL-MARL reward sharing mechanism is insufficient for multi-team coordination in this domain.

**C6. This is a contribution, not a failure.** The negative result is scientifically as valuable as Phase 5's positive result. Together, they define the boundary conditions for when reward sharing produces cooperative benefit under IL-MARL: it works when agents have strong influence over teammates in small non-zero-sum games with fixed adversaries. It fails when that influence is diluted by additional competing agents. This boundary condition is the central empirical finding of the dissertation's multi-agent investigation.
