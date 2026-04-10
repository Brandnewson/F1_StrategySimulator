# Phase 10: CTDE Literature Survey

**Purpose:** Structured summary of Centralised Training with Decentralised Execution (CTDE) methods for multi-agent reinforcement learning, annotated for relevance to the F1 Strategy Simulator's credit assignment problem at N≥4 agents.

---

## The Problem

Phases 6-8 established that independent learner MARL with reward sharing (alpha) fails at N≥4 agents. The root cause is **credit assignment dilution**: each agent's Q-value update includes reward from partners' actions it cannot influence. As N grows, signal-to-noise in Q-updates drops, and agents converge to passivity (the caution pathology). CTDE methods address this by using global information during training to assign credit to individual agents while preserving decentralised execution.

---

## Value Decomposition Methods (DQN-Compatible)

These methods factorise a joint team Q-value into per-agent Q-values. Compatible with our existing DQN/Rainbow-lite architecture.

### VDN — Value Decomposition Networks
- **Authors:** Sunehag, Lever, Gruslys, Lanctot, et al.
- **Venue:** AAMAS 2018
- **Key:** `sunehag2018vdn` (already in refs.bib)
- **Mechanism:** `Q_tot = Q_1 + Q_2 + ... + Q_n`. Simplest decomposition — joint value is the sum of individual values.
- **Credit assignment:** Each agent's gradient depends only on its own Q-value contribution. No cross-agent noise.
- **Limitation:** Assumes agent contributions are additive and independent. Cannot represent complementary strategies where one agent's action changes the value of another's.
- **Relevance:** Baseline CTDE method. Quick to implement (~30 lines). May be too restrictive for racing where zone access is competitive even among teammates.

### QMIX — Monotonic Value Function Factorisation
- **Authors:** Rashid, Samvelyan, Schroeder de Witt, Farquhar, Foerster, Whiteson
- **Venue:** ICML 2018
- **Key:** `rashid2018qmix` (already in refs.bib)
- **Mechanism:** `Q_tot = f(Q_1, Q_2, ..., Q_n; s)` where f is a mixing network with non-negative weights, conditioned on global state s via a hypernetwork.
- **Monotonicity constraint:** `∂Q_tot/∂Q_i ≥ 0` — if an agent's individual Q-value increases, joint Q-value also increases. Preserves individual incentive.
- **Credit assignment:** State-dependent. The mixing weights change based on the global race state, so credit attribution adapts to the situation (e.g., an agent near La Source gets more credit when it has a close gap).
- **Scaling:** Validated on StarCraft SMAC benchmark with 3-27 agents.
- **Relevance:** **Primary candidate for Phase 10.** DQN-native, state-dependent credit, canonical CTDE baseline, already cited in Chapter 1.

### QTRAN — Learning to Factorise with Transformation
- **Authors:** Son, Kim, Kang, Hostallero, Yi
- **Venue:** ICML 2019
- **Proposed key:** `son2019qtran`
- **Mechanism:** Removes QMIX's monotonicity constraint. Uses a transformation network that maps individual Q-values to joint Q-values with soft constraints.
- **Tradeoff:** More expressive than QMIX but less stable in practice. Training can diverge.
- **Relevance:** Reference for why QMIX's monotonicity constraint is acceptable — QTRAN's instability shows the expressiveness-stability tradeoff.

### QPLEX — Duplex Dueling Multi-Agent Q-Learning
- **Authors:** Wang, Ren, Liu, Yu, Zhang
- **Venue:** ICLR 2021
- **Proposed key:** `wang2021qplex`
- **Mechanism:** Decomposes Q_tot using dueling-style advantage decomposition. Each agent contributes a value and advantage term; mixing preserves the advantage structure.
- **Relevance:** Interesting because our DQN family already includes a Dueling variant. However, added complexity over QMIX is not justified for our initial CTDE experiment.

### Weighted QMIX
- **Authors:** Rashid, Farquhar, Peng, Whiteson
- **Venue:** ICML 2020
- **Proposed key:** `rashid2020weighted`
- **Mechanism:** Addresses QMIX's monotonicity limitation by weighting the projection from true Q_tot to factored Q_tot. Allows better approximation of non-monotonic value functions.
- **Relevance:** Upgrade path if QMIX's monotonicity proves too restrictive in our racing domain.

---

## Policy Gradient CTDE Methods (Not DQN-Compatible)

These require a shift from value-based to policy-gradient architectures. Listed for completeness and literature context.

### COMA — Counterfactual Multi-Agent Policy Gradients
- **Authors:** Foerster, Farquhar, Afouras, Nardelli, Whiteson
- **Venue:** AAAI 2018
- **Key:** `foerster2018coma` (already in refs.bib)
- **Mechanism:** Centralised critic computes a counterfactual baseline: "what would the team reward have been if agent i had taken the default action, holding everyone else fixed?" The difference is agent i's marginal contribution.
- **Credit assignment:** Theoretically optimal — each agent's gradient reflects exactly its causal contribution.
- **Relevance:** Gold standard for credit assignment, but requires actor-critic architecture incompatible with our DQN setup. Cited in Chapter 1 as future work.

### MADDPG — Multi-Agent Deep Deterministic Policy Gradient
- **Authors:** Lowe, Wu, Tamar, Harb, Abbeel, Mordatch
- **Venue:** NeurIPS 2017
- **Key:** `lowe2017maddpg` (already in refs.bib)
- **Mechanism:** Each agent has its own actor (decentralised) and critic (centralised, sees all agents' observations and actions). Designed for mixed cooperative-competitive settings.
- **Relevance:** Closest to our mixed-incentive setting but requires continuous action spaces and DDPG. Config stub exists in codebase (`states.py` lines 452-465) but unimplemented.

### MAPPO — Multi-Agent PPO
- **Authors:** Yu, Velu, Vinitsky, Gao, Wang, Baez, Fishi
- **Venue:** NeurIPS 2022
- **Proposed key:** `yu2022mappo`
- **Mechanism:** Surprisingly simple: standard PPO with a centralised value function that sees global state. Each agent's policy is decentralised.
- **Relevance:** Shows that simple centralised-value methods can match QMIX. PPO reference if we ever move beyond DQN.

---

## Difference Rewards (Reward Shaping, No Architecture Change)

### Optimal Payoff Functions for Members of Collectives
- **Authors:** Wolpert, Tumer
- **Venue:** Advances in Complex Systems, 2002
- **Key:** `wolpert2002optimal` (already in refs.bib)
- **Mechanism:** Each agent receives `D_i = G(z) - G(z_{-i})` where G is the global reward, z is the joint action, and z_{-i} is the joint action with agent i's action replaced by a default. Measures agent i's marginal contribution.
- **Credit assignment:** Per-agent, requires computing a counterfactual "what if agent i did nothing?"
- **Relevance:** Simplest possible upgrade from alpha-blending. No new networks — changes only the reward computation in `simulator.py`. Phase 10A-i baseline.

### Potential-Based Difference Rewards
- **Authors:** Devlin, Yliniemi, Kudenko, Tumer
- **Venue:** AAMAS 2014
- **Key:** `devlin2014potential` (already in refs.bib)
- **Mechanism:** Extends difference rewards with potential-based shaping to improve learning speed. Maintains the theoretical guarantees of difference rewards while adding a shaping term.
- **Relevance:** Possible refinement if basic difference rewards show promise but learn slowly.

---

## Benchmarks and Applications

### SMAC — StarCraft Multi-Agent Challenge
- **Authors:** Samvelyan, Rashid, de Witt, Farquhar, Nardelli, Rudner, Hung, Torr, Foerster, Whiteson
- **Venue:** CoRR 2019
- **Proposed key:** `samvelyan2019smac`
- **Relevance:** Standard benchmark for QMIX and CTDE methods. Validated at 3-27 agents in cooperative combat scenarios. Provides scaling reference for our N=4 experiment.

### Gran Turismo Sophy
- **Authors:** Wurman, Barrett, Kawamoto, et al.
- **Venue:** Nature 2022
- **Relevance:** State-of-the-art RL for racing. Used population-based SAC (not CTDE) but demonstrates that RL can achieve superhuman racing strategy. Different scale (full physics sim) but same domain.

---

## Summary: Method Selection Rationale

| Method | Credit Quality | DQN Compatible | Implementation | Selected |
|--------|:---:|:---:|:---:|:---:|
| Alpha-blending | None | Yes | Done (Phases 4-8) | Baseline |
| Difference rewards | Per-agent marginal | Yes | Minimal (~30 lines) | **10A-i** |
| VDN | Additive | Yes | Low | Ablation |
| **QMIX** | **State-dependent** | **Yes** | **Medium** | **10A-ii** |
| QTRAN | Full (unstable) | Yes | High | Not selected |
| COMA | Counterfactual | No | High | Future work |
| MADDPG | Centralised critic | No | High | Future work |

The progression from alpha → difference rewards → QMIX represents increasing sophistication of credit assignment, all testable within our existing DQN architecture.
