# Report Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the COMP3931 dissertation to cover all ten research phases using an RQ-centric structure in Chapter 4, add compulsory appendices, and meet the Excellent (≥80%) marking threshold.

**Architecture:** Four chapters remain the structure. Chapter 4 is reorganised around RQ1, RQ2, and RQ3 rather than sequential phases. Chapters 1, 2, and 3 receive targeted additions only. Appendix A and B are rewritten from their placeholder state.

**Tech Stack:** LaTeX, BibTeX (agsm/Leeds Harvard style), refs.bib (all citations already present)

**Style rules (enforced on every sentence written):**
- No semi-colons, no colons in prose, no em-dashes
- British English: behaviour, analyse, recognise, optimise, modelling, generalise
- One idea per sentence. Split any sentence doing two things.
- No AI writing markers: no "delve", "nuanced", "robust", "leverage", "pivotal", "crucial", "comprehensive", "groundbreaking", "streamline", "navigate", "tapestry", "foster", "underscore"
- No filler openers: "Additionally,", "Furthermore,", "Moreover,", "It is worth noting that"
- Prefer active voice. Use "we" not "I".

---

## File map

| File | Action | Scope |
|------|--------|-------|
| `report/config.tex` | Modify line 65 | Change bibliographystyle to agsm |
| `report/summary.tex` | Rewrite | Full abstract covering all 10 phases |
| `report/chapters/chapter1.tex` | Modify | Section 1.3 addition, Section 1.4 rewrite, Section 1.5 additions |
| `report/chapters/chapter2.tex` | Modify | Add Section 2.7 (MARL reward modes) |
| `report/chapters/chapter3.tex` | Modify | Fix placeholder line 232, add Section 3.8, tighten Section 3.4 reference |
| `report/chapters/chapter4.tex` | Rewrite | Full RQ-centric restructure replacing current content |
| `report/appendices.tex` | Modify | Rewrite Appendix A, rewrite Appendix B; Appendix C stays as-is |

---

## Task 1: Fix bibliography style and update summary

**Files:**
- Modify: `report/config.tex:65`
- Modify: `report/summary.tex`

- [ ] **Step 1: Change bibliographystyle in config.tex**

Open `report/config.tex`. Find line 65:
```latex
\bibliographystyle{plainnat}
```
Replace with:
```latex
\bibliographystyle{agsm}
```

Note: `agsm.bst` must be present in the `report/` directory for this to compile. If it is missing, download it from CTAN or copy it from a Leeds LaTeX template. The file name must be exactly `agsm.bst`.

- [ ] **Step 2: Rewrite summary.tex**

Replace the entire contents of `report/summary.tex` with:

```latex
This dissertation investigates how the structure of agent incentives shapes the strategies that emerge when reinforcement learning agents compete and cooperate in a stochastic racing simulator.

Two agents race across nine overtaking zones of a Spa-Francorchamps circuit over five laps. The core experimental variable is a reward sharing coefficient, alpha, which controls how much each agent values its teammate's outcome alongside its own.

Four DQN-family algorithms are compared in single-agent and multi-agent settings across three stochasticity levels. Rainbow-lite eliminates training collapse in all nine independent learner trials. Vanilla DQN collapses in 44 per cent of trials through a replay buffer contamination mechanism that more training budget worsens rather than resolves.

A reward sharing sweep from purely competitive to fully cooperative incentives reveals a sharp destabilisation threshold in zero-sum games above alpha of 0.5. Changing the game structure to include a common adversary eliminates this collapse entirely. At alpha of 0.75 in the three-agent non-zero-sum setting, joint performance against the adversary improves by 83 per cent over the competitive baseline.

Scaling to five agents reverses this result completely. Cooperative incentive at any alpha value degrades joint performance by up to 31 per cent. Curriculum scheduling recovers 69 per cent of this loss by separating two previously conflated failure modes: a training-order problem that curriculum solves, and a credit assignment problem that requires architectural solutions.

A zero-shot large language model baseline matches the best-trained RL agent within two per cent win rate using a qualitatively different strategy. Difference rewards, tested as an alternative cooperative formulation, produce competitive rather than cooperative incentives due to a mean-field approximation that inverts the cooperative gradient.

The findings show that incentive structure is at least as important as algorithm choice in determining emergent strategy. Centralised training with decentralised execution methods such as QMIX are identified as the natural next step to address the credit assignment problem at scale.
```

- [ ] **Step 3: Commit**

```bash
git add report/config.tex report/summary.tex
git commit -m "fix: Leeds Harvard bib style and rewrite abstract for full 10-phase scope"
```

---

## Task 2: Update Chapter 1

**Files:**
- Modify: `report/chapters/chapter1.tex`

Three targeted locations. Do not change anything outside these locations.

- [ ] **Step 1: Add references in Section 1.3 MARL subsection**

Find the paragraph beginning `\citet{busoniu2008survey} provides the foundational framework` in the MARL subsection. After the paragraph ending `...less well understood. This mixed setting is precisely the one examined in the present project.`, add a new paragraph:

```latex
\citet{tan1993multiagent} established that independent learners in cooperative games can outperform a single agent with shared information when the environment is sufficiently complex, but that performance degrades as the number of agents grows. \citet{panait2005cooperative} survey cooperative MARL and identify credit assignment as the central unsolved problem in team reward settings. \citet{wang2022irat} propose mixing individual and team rewards to balance credit assignment against cooperative alignment, which is the mechanism examined through the alpha coefficient in this project.
```

- [ ] **Step 2: Rewrite Section 1.4 contributions paragraph**

Find the paragraph beginning `This project addresses that gap through four linked contributions.` Replace it entirely with:

```latex
This project addresses that gap through six linked contributions. First, it frames race strategy as a social dilemma in which incentive structure determines emergent behaviour, not algorithm choice alone. Second, it implements a reward sharing coefficient spanning the full spectrum from purely competitive to fully cooperative and measures its effect in both zero-sum and non-zero-sum game structures. Third, it identifies a sharp scaling boundary at which cooperative incentive transitions from beneficial to harmful, and characterises this boundary as structural rather than parametric through a controlled ablation across agent counts. Fourth, it separates two previously conflated failure modes through curriculum alpha scheduling: a training-order problem that curriculum resolves, and a credit assignment problem that requires architectural solutions. Fifth, it provides a mathematical proof that a mean-field approximation of difference rewards \citep{wolpert2002optimal} inverts the cooperative gradient, producing competitive rather than cooperative incentives. Sixth, it establishes a zero-shot large language model as a non-learning baseline that matches the best trained agent within two per cent win rate, raising questions about what value-based learning adds in low-complexity discrete racing environments.
```

- [ ] **Step 3: Add two objectives in Section 1.5**

Find the `\end{enumerate}` closing the objectives list. Before it, add:

```latex
    \item Characterise the agent-count boundary at which cooperative incentive transitions from beneficial to harmful, and determine whether this boundary is parametric or structural.
    \item Evaluate difference rewards as an alternative cooperative formulation and establish whether a mean-field approximation preserves the cooperative gradient property of the original \citet{wolpert2002optimal} formulation.
```

- [ ] **Step 4: Update RQ hypotheses to reference full evidence base**

Find `\textbf{H1.}` and append one sentence to it:
```latex
Phase~3 confirms this: rainbow-lite trials show hawk-dove equilibria and zone specialisation in all nine trials, while vanilla DQN produces degenerate collapse in four of nine.
```

Find `\textbf{H2.}` and append one sentence:
```latex
Phase~4 confirms the sharp transition: collapse rates jump from 11 per cent at alpha of 0.5 to 89 per cent at alpha of 0.75 in zero-sum games.
```

Find `\textbf{H3.}` and append one sentence:
```latex
Phase~5 confirms genuine cooperative advantage at alpha of 0.75 in the non-zero-sum setting, with an ablation proving game structure rather than evaluation methodology as the causal variable.
```

- [ ] **Step 5: Commit**

```bash
git add report/chapters/chapter1.tex
git commit -m "feat: update Chapter 1 contributions, objectives, and hypotheses for full 10-phase scope"
```

---

## Task 3: Add Section 2.7 to Chapter 2

**Files:**
- Modify: `report/chapters/chapter2.tex`

Append the following after the final line of chapter2.tex (after the `\subsection{Good coding practice}` subsection):

- [ ] **Step 1: Add Section 2.7**

```latex
\section{MARL reward modes and multi-phase experimental scope}
\label{sec:marl-reward-modes}

Phases~4 to~10 introduce systematic variation in how agents are rewarded for their actions. This section describes the six reward and game-structure configurations used across these phases. All configurations are controlled through \texttt{config.json} and require no changes to agent learning code.

\subsection{Alpha reward mixing}
\label{sec:alpha-mixing}

The alpha reward mixing mechanism modifies the reward received by each agent as follows.

\[
R_{\text{mixed}} = (1 - \alpha) \cdot R_{\text{own}} + \alpha \cdot R_{\text{teammate}}
\]

At alpha of zero each agent receives only its own reward. At alpha of one each agent receives only its teammate's reward. Intermediate values produce partial alignment between the two agents' incentives.

Five alpha values are tested: 0.0, 0.25, 0.50, 0.75, and 1.0. This span allows threshold identification rather than a binary comparison. The alpha value is set via \texttt{reward.alpha} in \texttt{config.json} and applied uniformly to both agents in each trial.

\subsection{Non-zero-sum game design}
\label{sec:nonzerosum}

Phases~1 to~4 use a two-agent zero-sum race. The sum of position changes across both agents is structurally fixed, so one agent's gain always comes at the other's expense.

Phase~5 changes the game structure by adding a fixed heuristic Base adversary as a third competitor. Both DQN agents can now improve their result simultaneously by outperforming the Base agent. This breaks the zero-sum constraint and allows genuine cooperative advantage to be measured.

An ablation study within Phase~5 replaces each DQN agent with a Base agent in turn. This isolates whether game structure or policy quality causes any observed performance change.

\subsection{Multi-team scaling}
\label{sec:multiteam}

Phase~6 extends the non-zero-sum setting to four DQN agents grouped as two teams of two, plus a single Base adversary. Phase~7 tests an intermediate configuration of three DQN agents and one Base adversary, and also applies a low-alpha sweep within the five-agent configuration to test whether reducing alpha resolves the scaling failure. This sequence of agent counts identifies whether cooperation failure is agent-count-dependent and whether any parameterisation of alpha recovers cooperative performance at scale.

\subsection{Curriculum alpha scheduling}
\label{sec:curriculum}

Phase~8 applies a three-stage training schedule rather than a fixed alpha value throughout training. Agents train for 100 episodes at alpha of zero to establish individual policies. Alpha then increases linearly over 300 episodes to the target value. Agents train for a final 100 episodes at the target alpha.

The motivation is to test whether the passivity failure observed at fixed alpha of 0.75 results from cooperative reward overwriting individual policies before they are established, rather than from a fundamental incompatibility between cooperative incentives and independent learning.

\subsection{Difference rewards}
\label{sec:difference-rewards}

Phase~10 tests difference rewards as an alternative cooperative formulation. The original formulation by \citet{wolpert2002optimal} defines the reward for agent~$i$ as:

\[
D_i = G(\mathbf{z}) - G(\mathbf{z}_{-i})
\]

where $G(\mathbf{z})$ is the global performance under the full joint action vector and $G(\mathbf{z}_{-i})$ is the global performance when agent~$i$'s contribution is removed. This gives agent~$i$ credit only for its counterfactual contribution to collective performance.

A mean-field approximation is used in this project because the true counterfactual $G(\mathbf{z}_{-i})$ requires simulating the race without agent~$i$, which is computationally costly. The approximation substitutes the team mean delta for the counterfactual term:

\[
R_{\text{diff},i} = \Delta_i + (\Delta_i - \bar{\Delta}_{\text{team}})
\]

where $\Delta_i$ is agent~$i$'s position delta and $\bar{\Delta}_{\text{team}}$ is the mean delta across all team members. Whether this approximation preserves the cooperative gradient property of the original formulation is the central question of Phase~10.

\subsection{LLM semantic baseline}
\label{sec:llm-baseline}

Phase~9 introduces a zero-shot large language model as a non-learning comparator. The agent receives the current race state serialised as a natural language description. This includes zone difficulty, gap to the car ahead, current position, and laps remaining. The agent outputs a structured JSON action specifying whether to attempt an overtake and at which risk level. No training, replay buffer, or gradient update is used at any point.

The purpose is to establish whether domain reasoning alone, without any trial-and-error learning, can match the performance of a trained DQN agent. A fallback to HOLD is applied if the JSON response cannot be parsed.
```

- [ ] **Step 2: Commit**

```bash
git add report/chapters/chapter2.tex
git commit -m "feat: add Section 2.7 MARL reward modes covering Phases 4-10 methodology"
```

---

## Task 4: Update Chapter 3

**Files:**
- Modify: `report/chapters/chapter3.tex`

Three changes. Do not alter anything outside these locations.

- [ ] **Step 1: Fix the placeholder at line 232**

Find the sentence:
```
A schematic of this pipeline is provided as Appendix Figure~TBC.
```
Replace with:
```latex
The module responsibilities and data flow are described in full in the subsections above.
```

- [ ] **Step 2: Tighten the Phase 0 section reference in Section 3.4**

Find the opening of `\section{Phase 0: System integrity validation}`. Replace the paragraph beginning `Phase 0 establishes that the simulator...` with:

```latex
Phase~0 establishes that the simulator, reward accounting, stochasticity implementation, and telemetry contract are all correct before any learning claims are made. All Phase~0 assertions are formalised as automated tests in \texttt{tests/test\_phase0\_integrity.py}. The full test inventory, defects found, and gate outcome are documented in Appendix~\ref{appendix:phase01}. A summary follows.

Twelve tests are organised into three groups. Group~A checks simulator correctness. Group~B validates the stochastic probability calculations using 1{,}000 repeated samples per condition. Group~C verifies the telemetry schema and component accounting. All 12 tests pass. Three defects were found and corrected: a missing \texttt{gap\_to\_ahead} attribute caused the Base Agent to collapse to near-zero attempt probability; a reward sum accounting error surfaced through the \texttt{reward\_component\_sum\_error} field; and a cooldown boundary condition was incorrect by one tick. These defects would not have been detectable from win rates alone.

\textbf{Gate G0 outcome: passed.}
```

- [ ] **Step 3: Append Section 3.8 before the end of the file**

Add the following after the final line of chapter3.tex:

```latex
\section{Phase 4 to 10 infrastructure extensions}
\label{sec:phase4-10-infra}

Phases~4 to~10 required six extensions to the codebase beyond what Phase~3 used. Each extension was controlled through \texttt{config.json} and required no changes to agent learning code, preserving the fairness control described in Section~\ref{sec:phase3-infra}.

\textbf{Alpha mixing layer.} The reward calculation in \texttt{src/simulator.py} was extended with an alpha blending step applied after individual rewards are computed. The mixed reward for each agent is computed per Equation~\ref{sec:alpha-mixing}. The alpha value is read from \texttt{reward.alpha} at runtime. Setting alpha to zero reproduces the original reward contract exactly.

\textbf{Multi-team competitor selection.} \texttt{src/runtime\_profiles.py} was extended with a \texttt{select\_low\_marl\_team\_competitors} function that initialises $N$ DQN agents grouped into named teams. The team membership is used by the alpha mixing layer to identify which agents share a reward signal. The Base adversary is added as a separate non-team competitor.

\textbf{Curriculum scheduler.} A three-stage alpha schedule was implemented as a thin wrapper around the training loop. The scheduler reads the target alpha and episode boundaries from config and sets the active alpha value at the start of each training episode. It hooks into the existing \texttt{on\_episode\_end} callback and requires no changes to the agent or simulator internals.

\textbf{Difference reward module.} A separate reward calculation path is activated by setting \texttt{reward.mode} to \texttt{"difference"} in \texttt{config.json}. The module computes the mean-field approximation described in Section~\ref{sec:difference-rewards}. An algorithm confound was encountered during Phase~10: the configuration file still specified \texttt{"algo": "vanilla"} when rainbow-lite was intended. The error was caught by reading the \texttt{"algorithm"} field embedded in the output metric JSON files rather than relying on the config file alone. This incident confirmed the value of embedding config snapshots in telemetry output.

\textbf{LLM agent.} The LLM baseline implements the \texttt{BaseAgent} interface defined in \texttt{src/base\_agents.py}. It serialises the observation vector to a natural language prompt, calls the Claude Haiku API, and parses the JSON response for an action label. A fallback to HOLD is applied on any parse failure. The agent has no internal state between episodes and requires no training infrastructure.

\textbf{Extended evaluation script.} \texttt{scripts/evaluate\_marl.py} was extended to record team-level metrics alongside the per-agent metrics used in Phase~3. The primary new metric is \texttt{joint\_beat\_base\_rate}: the fraction of evaluation races in which both DQN agents finish ahead of the Base adversary. This metric measures genuine cooperative performance rather than relative performance between the two DQN agents.
```

- [ ] **Step 4: Commit**

```bash
git add report/chapters/chapter3.tex
git commit -m "feat: fix Chapter 3 placeholder, tighten Phase 0 reference, add Section 3.8 for Phase 4-10 infra"
```

---

## Task 5: Rewrite Chapter 4 — Sections 4.1 and 4.2 (Preliminary validation and RQ1)

**Files:**
- Modify: `report/chapters/chapter4.tex` — replace entire file contents

This task writes Sections 4.1 and 4.2 only. Tasks 6, 7, and 8 continue the file. Write the opening and first two sections now.

- [ ] **Step 1: Replace chapter4.tex with new opening through end of Section 4.2**

```latex
\chapter{Results, Evaluation and Discussion}
\label{chapter4}

This chapter presents results across all ten research phases and interprets them in the context of the three research questions. Section~\ref{sec:prelim} summarises the preliminary validation gates that confirmed experimental integrity before any multi-agent claims were made. Sections~\ref{sec:rq1} to~\ref{sec:rq3} address RQ1, RQ2, and RQ3 in turn, drawing on evidence from the relevant phases. Section~\ref{sec:discussion} compares findings against the existing literature. Section~\ref{sec:validity} identifies the main validity threats. Section~\ref{sec:futurework} sets out directions for future work.

\section{Preliminary validation}
\label{sec:prelim}

Three gate checks were completed before multi-agent experiments began. Table~\ref{tab:gates} summarises the outcome of each gate. Full details for Phases~0 and~1 are in Appendix~\ref{appendix:phase01}.

\begin{table}[htbp]
\centering
\small
\caption{Experimental gate outcomes for Phases 0 to 2}
\label{tab:gates}
\begin{tabular}{|l|l|l|p{6cm}|}
\hline
\textbf{Gate} & \textbf{Phase} & \textbf{Outcome} & \textbf{Key evidence} \\
\hline
G0 & Phase 0 system integrity & Pass & 12 of 12 tests pass; 3 defects found and corrected before any learning runs \\
\hline
G1 & Phase 1 smoke validation & Pass & Win rate above 0.75 across three seeds; AGGRESSIVE flip identified as Q-value overestimation symptom \\
\hline
G2 & Phase 2 DQN benchmark & Pass & Rainbow-lite wins 0.835 of races (CI95 0.817 to 0.852); zero collapse across nine trials \\
\hline
\end{tabular}
\end{table}

Phase~2 produced the DQN-family ranking used in all subsequent phases. Rainbow-lite ranked first at s0 (win rate 0.835, CI95 0.817 to 0.852), with a confidence interval that does not overlap vanilla DQN's interval (0.746, CI95 0.727 to 0.765). Rainbow-lite also showed near-zero robustness degradation across stochasticity levels (change of $+0.005$ from s0 to s2), confirming that its PER and double-target mechanisms prevent the Q-value instability documented in Phase~1. Dueling DQN ranked last at s0 (0.724), below vanilla, indicating that the value-advantage decomposition adds no benefit in the La Source-dominant two-agent environment. The s0 ranking was: rainbow-lite $>$ double $>$ vanilla $>$ dueling. This order informed the algorithm selection for all subsequent phases.

\section{RQ1: What strategic behaviours emerge and how robust are they?}
\label{sec:rq1}

Phase~3 is the first setting where emergent strategy from two concurrently learning agents can be observed. Two identical DQN-family agents train simultaneously, each maintaining independent replay buffers and network parameters with no shared weights or gradient updates at any point.

\subsection{Collapse rates and the algorithm hierarchy}

The primary Phase~3 finding is that algorithm choice determines whether competitive equilibrium is maintained at all. Table~\ref{tab:phase3-collapse} reports collapse rates by algorithm across all nine core trials per algorithm (three seeds by three stochasticity levels).

\begin{table}[htbp]
\centering
\small
\caption{Phase 3 degenerate collapse rates by algorithm (nine trials per algorithm)}
\label{tab:phase3-collapse}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Algorithm} & \textbf{s0} & \textbf{s1} & \textbf{s2} & \textbf{Total} & \textbf{Competitive trials} \\
\hline
Vanilla DQN   & 1 of 3 & 1 of 3 & 2 of 3 & 4 of 9 (44\%) & 56\% \\
Double DQN    & 1 of 3 & 0 of 3 & 0 of 3 & 1 of 9 (11\%) & 89\% \\
Dueling DQN   & 1 of 3 & 1 of 3 & 2 of 3 & 4 of 9 (44\%) & 56\% \\
Rainbow-lite  & 0 of 3 & 0 of 3 & 0 of 3 & 0 of 9 (0\%)  & 100\% \\
\hline
\end{tabular}
\end{table}

Rainbow-lite achieves zero collapses across all nine trials. Vanilla and dueling both show 44 per cent collapse rates with identical per-stochasticity profiles. Double DQN shows an 11 per cent rate. The ablation is unambiguous. Vanilla and dueling, which share neither PER nor double-target correction, achieve the same collapse rate. Adding double-target correction alone (double DQN) reduces the rate to 11 per cent. Adding PER alongside double-target correction eliminates collapse entirely.

The collapse mechanism is replay buffer contamination under positional asymmetry. The trailing agent's early overtake attempts at high-difficulty zones yield predominantly negative rewards. These populate the replay buffer with \textsc{attempt, negative reward} transitions. Under uniform sampling, Q(\textsc{hold}) climbs above Q(\textsc{attempt}) across gradient updates. The shared epsilon decay schedule then locks in the passive strategy before recovery is possible. Extended vanilla trials at 750 episodes worsened collapse rather than resolving it, confirming that the feedback loop is irreversible within the training horizon once activated. This is consistent with the instability pathology for independent Q-learners documented by \citet{foerster2017stabilising} and \citet{matignon2012independent}.

\citet{schroeaderdewitt2020ippo} show that independent learning with policy gradient methods achieves competitive performance on the StarCraft benchmark. The Phase~3 result is consistent with this: it is not independent learning per se that fails, but independent value-based learning with uniform replay under positional asymmetry. PER addresses the asymmetry by up-weighting high-TD-error transitions from both agents' early training windows.

\subsection{Emergent equilibrium classes}

Two equilibrium classes emerge across Phase~3, determined by whether the algorithm prevents collapse.

\textbf{Class 1: degenerate hierarchical resolution.} One agent converges to complete passivity while the other wins by default. This accounts for 44 per cent of vanilla and dueling trials. No strategic information is produced. This is the degenerate Nash equilibrium of non-cooperative two-agent Q-learning under positional asymmetry, as described by \citet{tan1993multiagent}.

\textbf{Class 2: zone specialisation with hawk-dove risk equilibrium.} Agents develop differentiated zone coverage and opposing risk profiles without any shared coordination mechanism. This occurs in 100 per cent of rainbow-lite trials and 89 per cent of double DQN trials. The canonical example is the rainbow-lite s2, seed~202, 750-episode trial, which achieves a zone differentiation index of 0.405, the highest in the dataset. Agent~1 concentrates on La Source, Les Combes, and Bruxelles with a predominantly conservative risk profile. Agent~2 takes a broader but thinner footprint across five zones with an aggressive risk bias. The two agents partition secondary overtaking territory while both contesting the highest-value zone.

This specialisation emerges from iterated best-response dynamics \citep{busoniu2008survey}. When one agent commits to aggressive attempts at a zone, the other's best response is to focus on lower-risk alternatives. Zone preferences strengthen over time as Q-values for each agent's territory become more reliable than Q-values for the contested territory. The hawk-dove risk profile is a mixed-strategy equilibrium of the risk-selection subgame, consistent with competitive two-agent MARL theory.

Zone specialisation is also a training-budget effect. At 500 episodes, the zone differentiation index for rainbow-lite at s2 is 0.022. At 750 episodes it rises to 0.405, a near-20-fold increase. PER requires sufficient zone-specific transitions to assign reliable relative priorities for secondary zones. At 500 episodes these priorities are unreliable for rarely-visited zones, causing both agents to default to La Source.

\subsection{Non-stationarity versus stochasticity}

A crossplay protocol applied to rainbow-lite at s2 reveals a structural evaluation bias: the A1 slot receives a less favourable starting position across evaluation races, producing an approximately 11 percentage point disadvantage relative to the expected parity floor of 0.500. All raw A1 win rates in Phase~3 are subject to this confounder. The collapse versus non-collapse classification is unaffected, since a win rate of 0.960 is degenerate regardless of the offset. Bias-corrected results for rainbow-lite at s2 show four of five seeds at or above the parity floor. The apparent systematic A2 dominance in raw data dissolves under correction.

The dominant challenge in Phase~3 is non-stationarity from a concurrently adapting opponent, not stochastic outcome noise. Vanilla and dueling collapse rates rise from one of three at s0 to two of three at s2, showing that stochasticity accelerates but does not cause collapse. Rainbow-lite maintains zero collapses at all stochasticity levels, with mean win rates of approximately 0.553 at s0, 0.552 at s1, and 0.400 at s2. The downward trend at s2 reflects increased evaluation variance rather than policy instability. This pattern is consistent with \citet{foerster2017stabilising}: MARL non-stationarity is a more severe stability challenge than stochastic outcome variance for independent learners.

\subsection{RQ1 answer}

Concurrent training produces interpretable strategic structure when the algorithm prevents collapse. Rainbow-lite generates hawk-dove risk equilibria and zone specialisation in all nine trials. These behaviours emerge from iterated best-response dynamics without any coordination mechanism. Zone specialisation is a training-budget effect at high stochasticity rather than a threshold phenomenon: 750 training episodes at s2 produce qualitatively different zone coverage to 500 episodes. H1 is confirmed with the qualification that the specialisation signal requires sufficient training budget to stabilise.
```

- [ ] **Step 2: Commit**

```bash
git add report/chapters/chapter4.tex
git commit -m "feat: Chapter 4 sections 4.1 and 4.2 — preliminary validation and RQ1"
```

---

## Task 6: Write Chapter 4 — Section 4.3 (RQ2)

**Files:**
- Modify: `report/chapters/chapter4.tex` — append to existing content

- [ ] **Step 1: Append Section 4.3 to chapter4.tex**

```latex
\section{RQ2: How does incentive structure alter strategy, and is there a destabilisation threshold?}
\label{sec:rq2}

Phase~3 established what happens with purely competitive agents (alpha of 0). RQ2 asks what happens when the rules change. Five phases address this question by varying alpha, game structure, agent count, and training schedule.

\subsection{Phase 4: zero-sum alpha sweep}

Phase~4 introduces the alpha reward mixing mechanism into the two-agent zero-sum game from Phase~3. Five alpha values are tested across 45 trials total. Table~\ref{tab:phase4-collapse} reports collapse rates by alpha value.

\begin{table}[htbp]
\centering
\small
\caption{Phase 4 collapse rates by alpha value in two-agent zero-sum game (9 trials per alpha)}
\label{tab:phase4-collapse}
\begin{tabular}{|c|c|c|}
\hline
\textbf{Alpha} & \textbf{Collapse rate} & \textbf{Competitive trials} \\
\hline
0.00 & 0 of 9 (0\%)   & 100\% \\
0.25 & 1 of 9 (11\%)  & 89\% \\
0.50 & 1 of 9 (11\%)  & 89\% \\
0.75 & 8 of 9 (89\%)  & 11\% \\
1.00 & 9 of 9 (100\%) & 0\%  \\
\hline
\end{tabular}
\end{table}

The transition between alpha of 0.5 and 0.75 is sharp. Below this threshold, low collapse rates indicate that partial incentive alignment does not destabilise learning. Above it, agents converge to passivity in nearly every trial. At alpha of one, every trial collapses.

The mechanism is structural. In a two-agent zero-sum race, one agent's position gain comes at the other's expense. An agent receiving a large fraction of its teammate's reward is penalised when its own overtakes improve its position at the teammate's cost. Q-values for active strategies become negatively rewarded through the cooperative term. Agents learn that the safest strategy is to hold position and let the teammate act. Once both agents reach this conclusion, both collapse to passivity simultaneously. This is consistent with the cooperative game dynamics described by \citet{claus1998dynamics}.

H2 predicted a sharp threshold. The Phase~4 data confirm this at the 0.5 to 0.75 boundary. The threshold is not a function of a particular alpha value but of the point at which the cooperative reward term begins to dominate the individual gradient.

\subsection{Phase 5: non-zero-sum game and genuine cooperative advantage}

Phase~4 raised a structural question. Is cooperation fundamentally incompatible with independent learning, or is the zero-sum game the problem? Phase~5 changes the game structure by adding a fixed Base adversary as a third competitor. Both DQN agents can now improve simultaneously by outperforming the adversary. Table~\ref{tab:phase5-results} shows the joint beat-base rate by alpha value across 45 trials.

\begin{table}[htbp]
\centering
\small
\caption{Phase 5 joint beat-base rate by alpha value in three-agent non-zero-sum game (9 trials per alpha)}
\label{tab:phase5-results}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Alpha} & \textbf{Joint beat-base rate} & \textbf{Change from alpha 0} & \textbf{Collapse rate} \\
\hline
0.00 & 0.310 & baseline & 0\% \\
0.25 & 0.294 & $-5\%$   & 0\% \\
0.50 & 0.433 & $+40\%$  & 0\% \\
0.75 & 0.567 & $+83\%$  & 0\% \\
1.00 & 0.436 & $+41\%$  & 0\% \\
\hline
\end{tabular}
\end{table}

Collapse disappears at all alpha values. The joint beat-base rate follows an inverted-U curve peaking at alpha of 0.75. Alpha of one produces less benefit than alpha of 0.75 because agents need a minimum of individual incentive to maintain coherent individual learning gradients. When the individual component is removed entirely, agents lose the signal that their own actions have consequences, weakening the overtaking strategy that generates joint performance.

Hawk-dove risk roles re-emerge under cooperation. At alpha of 0.75 the two agents adopt complementary risk profiles directed against the Base adversary rather than against each other. The weaker agent at alpha of zero (Agent~2, 63 per cent beat-base rate) improves most at alpha of 0.75 (78 per cent beat-base rate). This is coordinated resource partitioning, not redistribution of a fixed outcome pool.

An ablation study confirms causality. Replacing one DQN agent with a Base agent at alpha of 0.75 reduces joint beat-base from 0.567 to 0.421. Replacing both DQN agents with Base agents at alpha of 0.75 reduces it to 0.289. The alpha effect operates on top of the policy quality effect. Game structure alone does not explain the Phase~5 improvement \citep{hughes2018inequity}.

\subsection{Phase 6: scaling to five agents}

Phase~6 scales the successful Phase~5 configuration to four DQN agents (two per team) plus one Base adversary. Table~\ref{tab:phase6-results} shows both-beat-base rates across symmetric and asymmetric alpha conditions.

\begin{table}[htbp]
\centering
\small
\caption{Phase 6 both-beat-base rate by team alpha condition in five-agent game (9 trials per condition)}
\label{tab:phase6-results}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Condition} & \textbf{Both-beat-base rate} & \textbf{Change from 0 vs 0} \\
\hline
Team A alpha 0.0 vs Team B alpha 0.0 & 0.350 & baseline \\
Team A alpha 0.5 vs Team B alpha 0.5 & 0.306 & $-12\%$ \\
Team A alpha 0.75 vs Team B alpha 0.75 & 0.243 & $-31\%$ \\
Team A alpha 0.75 vs Team B alpha 0.0 & 0.291 & $-17\%$ \\
\hline
\end{tabular}
\end{table}

Every form of cooperation degrades performance. The cooperative team loses to the competitive team in the asymmetric condition (47.5 per cent win rate for the cooperative team). The 83 per cent improvement from Phase~5 becomes a 31 per cent degradation at five agents.

The root cause is credit assignment dilution. In Phase~5 each agent's actions strongly influence one teammate. At alpha of 0.75 in a three-agent race, 75 per cent of reward comes from the teammate whose race the agent directly affects. In Phase~6, each agent's influence is diluted across three other agents sharing the track. Three-quarters of the reward signal comes from sources the agent can barely control. Q-value updates become noisy. Agents learn caution rather than coordination. This is the credit assignment problem identified by \citet{agogino2004unifying}: shared team reward becomes uninformative to individual agents when the team is large relative to any one agent's sphere of influence.

\subsection{Phase 7: characterising the scaling boundary}

Phase~7 identifies the agent count at which cooperation transitions from beneficial to neutral or harmful. Phase~7A tests three DQN agents plus one Base adversary. Phase~7B applies a low-alpha sweep within the five-agent configuration. Table~\ref{tab:phase7-boundary} shows the alpha of 0.75 effect across agent counts.

\begin{table}[htbp]
\centering
\small
\caption{Phase 7 cooperative advantage by total agent count at alpha 0.75}
\label{tab:phase7-boundary}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Configuration} & \textbf{Alpha 0.75 effect} & \textbf{Interpretation} \\
\hline
3 agents (Phase 5)  & $+83\%$ & Strong cooperative advantage \\
4 agents (Phase 7A) & $+5\%$  & No meaningful effect (within noise) \\
5 agents (Phase 6)  & $-31\%$ & Cooperative harm \\
\hline
\end{tabular}
\end{table}

Adding one agent to the successful Phase~5 setup eliminates the entire cooperative advantage. The boundary is at four agents and it is a cliff rather than a slope. Phase~7B confirms that this failure is not parametric. Testing alpha values of 0.10, 0.15, and 0.25 in the five-agent game produces no improvement over the competitive baseline (beat-base rates between 0.344 and 0.347, against a baseline of 0.350). There is no alpha value that recovers cooperative performance at this scale.

Interestingly, alpha of 0.75 increases zone differentiation by 63 per cent at four agents relative to alpha of zero. Agents do respond to the cooperative signal by specialising their zone coverage. But this specialisation does not translate into better joint outcomes. Instead, attempt rates drop 38 to 64 per cent across zones. Agents specialise by becoming cautious, not by becoming effective. The cooperative signal teaches them to avoid conflict rather than to exploit their respective strengths.

\subsection{Phase 8: curriculum alpha scheduling}

Phase~8 tests whether the Phase~6 and Phase~7 failures are caused by the training order rather than by a fundamental incompatibility between cooperation and independent learning. The curriculum schedule trains agents at alpha of zero for 100 episodes before ramping to the target alpha over 300 episodes. Table~\ref{tab:phase8-results} shows results in both the four-agent and five-agent configurations.

\begin{table}[htbp]
\centering
\small
\caption{Phase 8 curriculum results versus fixed alpha and competitive baseline}
\label{tab:phase8-results}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Condition} & \textbf{4-agent joint beat-base} & \textbf{5-agent both-beat-base} & \textbf{Recovery} \\
\hline
Competitive (alpha 0.0) & 0.159 & 0.350 & baseline \\
Fixed alpha 0.75        & 0.167 & 0.243 & --- \\
Curriculum (0 to 0.75)  & 0.184 & 0.318 & 4-agent: n/a; 5-agent: 69\% \\
\hline
\end{tabular}
\end{table}

At five agents, curriculum scheduling recovers 69 per cent of the Phase~6 performance loss. The caution pathology is 86 to 89 per cent eliminated. Agents that first learn individual policies retain aggressive behaviour when cooperative reward is introduced gradually. At four agents, curriculum produces the highest joint beat-base of any condition (0.184) but still falls short of demonstrating genuine cooperative advantage over a competitive baseline.

Despite the recovery, cooperation does not emerge. Intra-team action correlation remains negative ($-0.189$) in the five-agent curriculum condition. Both agents improve individually but do not coordinate actions. Curriculum scheduling prevents the damage from training order but cannot solve the underlying credit assignment problem.

This finding separates two failure modes that prior literature conflated \citep{panait2005cooperative}. The training-order problem is solved by curriculum: immediate cooperative reward overwrites individual policies with passivity, and gradual introduction prevents this. The credit assignment problem is not solved: at four or more agents, no scheduling of a shared team reward can produce cooperative gradients strong enough to align agent behaviour at the individual action level. Architectural solutions are required. Centralised training with decentralised execution methods such as QMIX \citep{rashid2018qmix} are specifically designed to address this, by giving agents access to global information during training even when executing from local observations alone.

\subsection{RQ2 answer}

The destabilisation threshold exists and is structural. In zero-sum games it appears between alpha of 0.5 and 0.75. In non-zero-sum games it appears at four total agents regardless of alpha value or training schedule. H2 is confirmed. The threshold is not a function of alpha parameterisation but of the relationship between agent count and the strength of individual credit signal in the reward formula.
```

- [ ] **Step 2: Commit**

```bash
git add report/chapters/chapter4.tex
git commit -m "feat: Chapter 4 Section 4.3 — RQ2 results covering Phases 4-8"
```

---

## Task 7: Write Chapter 4 — Sections 4.4 to 4.7 (RQ3, Discussion, Validity, Future work)

**Files:**
- Modify: `report/chapters/chapter4.tex` — append to existing content

- [ ] **Step 1: Append Sections 4.4 through 4.7 to chapter4.tex**

```latex
\section{RQ3: Does shared incentive produce genuine cooperative advantage?}
\label{sec:rq3}

RQ2 identified when and why cooperative incentive fails. RQ3 addresses the positive question: when cooperation works, is the advantage genuine? And what happens when a different cooperative reward formulation is used?

\subsection{Phase 5: proving causality of the cooperative advantage}

The Phase~5 ablation study, described in Section~\ref{sec:rq2}, establishes that the 83 per cent improvement at alpha of 0.75 is caused by the cooperative reward signal rather than by game structure alone. Replacing either DQN agent with a Base agent at the same alpha reduces joint performance substantially. Replacing both reduces it further. The alpha effect adds on top of the policy quality effect independently.

The improvement is concentrated in the weaker agent. Agent~2's beat-base rate rises from 63 per cent at alpha of zero to 78 per cent at alpha of 0.75. This is not redistribution of a fixed outcome pool: both agents improve, and the total joint beat-base rate rises by 83 per cent relative to the competitive baseline. The non-zero-sum structure allows this, because the adversary provides a shared target that both agents can outperform simultaneously.

This finding answers RQ3 affirmatively for the three-agent non-zero-sum setting. Genuine cooperative advantage exists. It requires non-zero-sum game structure, a collapse-resistant algorithm, and an alpha value that preserves sufficient individual incentive alongside the cooperative signal. H3 is confirmed.

\subsection{Phase 9: LLM semantic baseline}

Phase~9 introduces a zero-shot large language model agent to establish a non-learning performance bound. The agent receives the race state in natural language and outputs an action without any training, replay buffer, or gradient computation. Table~\ref{tab:phase9-results} shows win rates against the Base adversary across stochasticity levels.

\begin{table}[htbp]
\centering
\small
\caption{Phase 9 LLM baseline win rate versus trained rainbow-lite across stochasticity levels}
\label{tab:phase9-results}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Agent} & \textbf{s0 win rate} & \textbf{s1 win rate} & \textbf{s2 win rate} \\
\hline
Rainbow-lite (trained) & 0.835 & 0.839 & 0.840 \\
LLM zero-shot          & 0.817 & 0.822 & 0.801 \\
\hline
\end{tabular}
\end{table}

The LLM matches trained rainbow-lite within two per cent win rate across all three stochasticity levels. The gap is not statistically meaningful given three-seed evaluation budgets.

The strategies are qualitatively different. The LLM prioritises conservative attempts at La Source and holds at hard zones consistently across all stochasticity levels. Rainbow-lite diversifies zone coverage and adjusts risk levels based on trained Q-value estimates. Two different routes to the same outcome performance suggests that the low-complexity environment has a near-optimal policy short enough to be described in natural language.

This result does not indicate that trained RL is unnecessary. It indicates that in a sparse-decision five-lap race with one dominant zone of 80 per cent base success probability, the strategy space is narrow enough that domain reasoning and Q-value optimisation converge on similar outcomes. A richer environment with tyre degradation, pit-stop decisions, and a larger field of competitors would be expected to widen this gap substantially.

\subsection{Phase 10: difference rewards as a cooperative formulation}

Phase~10 tests difference rewards as an alternative to the alpha mixing mechanism. The formulation from \citet{wolpert2002optimal} gives each agent credit only for its counterfactual contribution to collective performance. The mean-field approximation used is:

\[
R_{\text{diff},i} = \Delta_i + (\Delta_i - \bar{\Delta}_{\text{team}})
\]

where $\Delta_i$ is agent~$i$'s position delta and $\bar{\Delta}_{\text{team}}$ is the mean delta across all team members.

The cooperative gradient property of the original formulation requires that each agent's reward increases when a teammate performs better. We can check whether the approximation preserves this by differentiating $R_{\text{diff},i}$ with respect to teammate~$j$'s delta $\Delta_j$:

\[
\frac{\partial R_{\text{diff},i}}{\partial \Delta_j} = -\frac{1}{N}
\]

This gradient is negative for all values of $N$. When teammate $j$ overtakes successfully, agent~$i$'s reward decreases because the team mean rises. The approximation inverts the cooperative gradient. Rather than giving agent~$i$ credit for its contribution, it penalises agent~$i$ for its teammate's success. This is the opposite of what \citet{wolpert2002optimal} intended. The mean-field substitution loses the counterfactual property because it uses the observed team mean rather than the true counterfactual outcome $G(\mathbf{z}_{-i})$.

Table~\ref{tab:phase10-results} shows the empirical result.

\begin{table}[htbp]
\centering
\small
\caption{Phase 10 joint beat-base rate under difference rewards versus IQL baseline}
\label{tab:phase10-results}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Condition} & \textbf{Joint beat-base rate} & \textbf{Change from IQL} \\
\hline
IQL baseline (rainbow-lite, alpha 0)      & 0.158 & baseline \\
Difference rewards (vanilla DQN)          & 0.087 & $-45\%$ \\
Difference rewards (rainbow-lite)         & 0.133 & $-16\%$ \\
\hline
\end{tabular}
\end{table}

Difference rewards rank last among all tested reward formulations. Even the stronger algorithm (rainbow-lite) produces a 16 per cent regression from the IQL baseline. The algorithm effect (31 percentage points between vanilla and rainbow-lite) exceeds the formula effect (16 percentage points from IQL to rainbow-lite difference rewards), confirming that algorithm robustness remains a determining factor even when the reward formula is changed.

\citet{devlin2014potential} show that potential-based shaping of difference rewards can preserve the cooperative gradient when the approximation is carefully designed. The mean-field substitution used here does not meet this standard. Testing a potential-based formulation is identified as future work in Section~\ref{sec:futurework}. The Yang et al. (2018) mean field approximation \citep{yang2018meanfield} addresses a related problem (scaling to large populations) but does not resolve the counterfactual inversion problem identified here.

\subsection{RQ3 answer}

Genuine cooperative advantage exists and is confirmed empirically in Phase~5. It requires non-zero-sum game structure, a collapse-resistant algorithm, and a cooperative reward formula that preserves positive cooperative gradients. The mean-field difference rewards approximation fails the third condition and produces competitive rather than cooperative incentives. Incentive formula design matters as much as algorithm choice in determining whether cooperation can emerge.

\section{Discussion}
\label{sec:discussion}

\subsection{Positioning relative to the literature}

\citet{tan1993multiagent} established that independent learners in cooperative games can match or exceed a single centralised agent when the environment is complex, but identified communication and shared experience as critical facilitators. Phase~3 of this project extends this finding to a competitive setting: independent learning produces viable strategies at two agents but cannot sustain cooperation at four or more without architectural support.

\citet{wolpert2002optimal} designed difference rewards specifically to solve the credit assignment problem through counterfactual reasoning. Phase~10 shows that the mean-field approximation of this formulation loses the counterfactual property and inverts the cooperative gradient. This is consistent with the finding of \citet{devlin2014potential} that difference rewards require careful approximation design to preserve their theoretical properties in practice.

\citet{matignon2012independent} survey the failure modes of independent learners in cooperative games and identify leniency, hysteresis, and reward shaping as mitigating strategies. The curriculum scheduling in Phase~8 is a form of reward shaping that addresses the training-order failure mode. The Phase~8 finding that curriculum solves the training-order problem but not the credit assignment problem is a refinement of Matignon et al.'s taxonomy: the two problems are separable experimentally and require different solutions.

\citet{foerster2017stabilising} identify replay buffer non-stationarity as the primary failure mode for deep independent Q-learners in multi-agent settings. PER is confirmed here as the load-bearing anti-collapse mechanism across all nine rainbow-lite trials. The Phase~3 ablation hierarchy (vanilla 44 per cent, double 11 per cent, rainbow 0 per cent) isolates PER's contribution cleanly from the double-target correction contribution.

\subsection{The three novel contributions}

This project makes three contributions that are not present in the existing literature.

The first is the empirical identification of a sharp agent-count scaling boundary at N of four in a non-zero-sum cooperative game. Prior work has documented cooperation failure at scale, but not characterised the precise agent count at which the transition occurs or confirmed through a low-alpha sweep that the failure is structural rather than parametric.

The second is the decomposition of cooperative MARL failure into two separable problems using curriculum scheduling as a surgical tool. The training-order problem and the credit assignment problem have been discussed separately in the literature \citep{panait2005cooperative} but have not been isolated experimentally in a single controlled setting with a common evaluation metric.

The third is the mathematical proof that the mean-field approximation of difference rewards inverts the cooperative gradient, accompanied by the empirical confirmation that this inversion produces a performance regression below the competitive baseline. This connects a theoretical observation to a measured outcome in a concrete experimental setting.

\section{Validity threats and limitations}
\label{sec:validity}

\textbf{La Source structural dominance.} La Source's difficulty of 0.2 gives a base success probability of 0.8. Consistent attempts at La Source produce a reliable win margin over five laps regardless of decisions at other zones. The win rate metric in Phases~1 and~2 partly reflects exploitation of this single zone rather than genuine strategy learning. Zone discrimination metrics and the zone differentiation index are used throughout to mitigate this, but the La Source confound is an inherent limitation of the low-complexity simulator scope.

\textbf{Evaluation protocol positional bias.} The crossplay analysis in Phase~3 reveals an approximately 11 percentage point structural disadvantage against the A1 evaluation label. All raw A1 win rates in Phase~3 are affected. The collapse classification is not affected. A code-level fix applying the same round-robin start position cycle during evaluation as during training would resolve this in future work.

\textbf{Short race and sparse decisions.} Five laps across nine zones produce at most 45 zone-passing events per race. A subset of these produce actual decisions. This sparsity is part of why zone specialisation requires 750 episodes rather than 500 at s2. It is an inherent limitation of the low-complexity tier.

\textbf{Single circuit.} All experiments use the Spa-Francorchamps zone configuration. Findings about La Source dominance and zone partitioning may not transfer to circuits with more balanced zone difficulty distributions.

\textbf{Three seeds per cell.} Phase~2 and Phase~3 use three seeds per cell rather than the planned five. Between-seed confidence intervals are wider than they would be at five seeds. This does not affect the Phase~2 ranking at s0, where confidence intervals are non-overlapping, but increases uncertainty in Phase~3 comparative claims.

\section{Future work}
\label{sec:futurework}

\textbf{QMIX and centralised training.} Phase~8 identified the credit assignment problem as the barrier to cooperation at N of four or more. QMIX \citep{rashid2018qmix} addresses this directly through a mixing network that enforces a monotonicity constraint on the relationship between individual Q-values and the joint Q-value. This preserves cooperative gradients at the individual action level regardless of agent count. QTRAN \citep{son2019qtran} relaxes the monotonicity constraint to allow exact factorisation. Both are natural next steps following the Phase~8 decomposition.

\textbf{Potential-based difference rewards.} \citet{devlin2014potential} show that potential-based shaping of difference rewards can preserve the cooperative gradient when the approximation is designed to maintain the counterfactual property. Testing this formulation would complete the Phase~10 investigation and determine whether the problem is with difference rewards as a concept or with the specific mean-field approximation used here.

\textbf{Medium and high complexity tiers.} The simulator is designed for staged complexity. Medium complexity would introduce multiple concurrent competitors in a larger field. High complexity would add tyre degradation, pit-stop decisions, and weather context. The observation contracts for both tiers are already defined in \texttt{config.json}. These tiers are where the strategic richness of Formula~1 is most concentrated.

\textbf{Real telemetry integration.} Zone difficulty values are currently derived from position-change counts at timing loop locations in the 2023 Belgian Grand Prix data. A more precise calibration using sector-by-sector lap time deltas would produce difficulty values grounded in car performance rather than positional tracking artefacts.
```

- [ ] **Step 2: Commit**

```bash
git add report/chapters/chapter4.tex
git commit -m "feat: Chapter 4 sections 4.4-4.7 — RQ3, discussion, validity, future work"
```

---

## Task 8: Rewrite Appendix A and B in appendices.tex

**Files:**
- Modify: `report/appendices.tex` — replace Appendix A and B content; preserve Appendix C onwards

- [ ] **Step 1: Replace Appendix A and B**

In `report/appendices.tex`, find the `\begin{appendices}` tag. Replace everything from there up to (but not including) `\chapter{Phase 0 and Phase 1 Experimental Validation}` with:

```latex
\begin{appendices}

\chapter{Self-Appraisal}
\label{appendix:selfappraisal}

\section{Project process reflection}

The phased gate structure was the single most effective process decision in the project. Each phase's result determined the next phase's design rather than following a pre-specified plan. Phase~2's finding that rainbow-lite was the only collapse-resistant algorithm in single-agent evaluation meant that Phase~3 could attribute multi-agent collapse to MARL non-stationarity rather than to algorithm fragility. Without this gate, the Phase~3 collapse results would have been ambiguous. The same principle applied at each subsequent gate: Phase~4's zero-sum failure motivated Phase~5's game structure change, and Phase~5's success motivated Phase~6's scaling test.

The config-driven fairness control was also effective. Because all experimental parameters were externalised to \texttt{config.json} from the start, every comparison was provably identical in all conditions other than the variable under test. An external reviewer can reconstruct any trial from the config snapshot embedded in the metric output file. This discipline was not costly to implement but prevented a class of confounds that would have been difficult to detect and correct retrospectively.

The decision to continue beyond Phase~3 rather than stopping at the first successful result produced the most original contributions. The 83 per cent cooperative improvement in Phase~5 would not have been found if the project had ended at the Phase~3 competitive equilibrium results. The scaling failure in Phase~6, the boundary characterisation in Phase~7, the curriculum decomposition in Phase~8, and the difference rewards analysis in Phase~10 all followed from the decision to keep asking the next question.

What did not work well: the algorithm confound in Phase~10. The \texttt{config.json} file still contained \texttt{"algo": "vanilla"} when rainbow-lite was intended for Phase~10A. The first batch of trials therefore tested the wrong algorithm. The error was caught by reading the \texttt{"algorithm"} field embedded in the output metric JSON rather than by reviewing the config file before running. The lesson is that config inspection at the start of every experiment batch is a necessary check, not an optional one. Embedding config snapshots in telemetry output is an effective safeguard, but it should not be the only one.

Writing pace was a second area for improvement. Chapters~1 to~3 were drafted while Phases~1 to~3 were active, which made them detailed and timely. Phases~4 to~10 were completed before the corresponding chapters were written, which made the retrospective write-up harder. Writing contemporaneously with experimentation is a better practice.

\section{Personal reflection and lessons learned}

The most significant conceptual shift during the project was the recognition that negative results are contributions. Phase~10's failure to produce cooperative behaviour via difference rewards was frustrating initially. The mathematical analysis of the gradient inversion transformed it into a finding: not only did the formula fail, but the reason it failed can be stated precisely. This precision is what makes a negative result publishable rather than merely disappointing.

The N of four scaling cliff was not predicted. The project could have treated Phase~6's failure as a dead end and stopped. Designing Phase~7 to characterise the boundary precisely was the decision that produced a genuinely new empirical finding. The habit of asking "where exactly does this transition occur?" rather than "does this work?" produced more informative results throughout the project.

Reproducibility standards imposed discipline that paid dividends later. Because all prior phases used fixed seeds and matched budgets, the Phase~7 and Phase~8 ablations could attribute results to game structure and training schedule rather than to evaluation noise. The upfront cost of implementing this discipline was small. The downstream benefit of being able to make confident comparative claims was large.

\section{Legal, social, ethical and professional issues}

\subsection{Ethical issues}

No human participants were involved at any stage of this project. No consent forms were required. The LLM baseline in Phase~9 uses the Anthropic Claude Haiku API. No personal data was transmitted to the API at any point. Race state information is synthetic and contains no identifying information.

\subsection{Legal issues}

All external libraries used in this project are open-source. FastF1 is released under the MIT licence. PyTorch is released under a BSD-style licence. NumPy and Pandas are released under BSD licences. The Anthropic Claude Haiku API was used under Anthropic's standard commercial terms of service. The FastF1 telemetry data is drawn from publicly available Formula~1 race data intended for non-commercial research use, consistent with the library's intended purpose and licence terms.

\subsection{Social issues}

This project does not interact with human participants and does not collect or process personal data. The research question concerns the behaviour of autonomous software agents in a simulated environment. There are no direct social impacts from the project itself.

The broader research area has social relevance. Autonomous agents with configurable incentive structures are deployed in domains beyond motorsport simulation, including financial trading, logistics optimisation, and competitive resource allocation. The finding that incentive formula design shapes emergent agent behaviour as strongly as algorithm choice is relevant to anyone designing the reward contracts for multi-agent systems in these domains. The project provides controlled evidence that poorly designed cooperative incentives can produce competitive or even harmful agent behaviour, which is a meaningful contribution to the broader conversation about AI alignment in multi-agent settings.

\subsection{Professional issues}

This project follows the reproducibility standards identified as necessary for trustworthy reinforcement learning comparisons by \citet{henderson2018deep} and \citet{agarwal2021statistical}. Matched compute budgets, fixed seeds, confidence interval reporting, and behavioural diagnostics that go beyond outcome-only metrics are treated as professional obligations rather than optional additions. The git repository provides a full audit trail of experimental code, configuration changes, and metric outputs. All claims in the main report are traceable to specific metric files or mathematical derivations.

\chapter{External Materials}
\label{appendix:external}

All materials not developed by the student are listed below.

\begin{table}[htbp]
\centering
\small
\caption{External materials used in this project}
\label{tab:external-materials}
\begin{tabular}{|p{3.5cm}|p{2.5cm}|p{5cm}|p{2.5cm}|}
\hline
\textbf{Material} & \textbf{Version} & \textbf{Purpose} & \textbf{Licence} \\
\hline
FastF1 & v3.x & 2023 Belgian Grand Prix telemetry for overtaking zone difficulty calibration & MIT \\
\hline
PyTorch & v2.x & Neural network implementation for all DQN-family variants & BSD \\
\hline
NumPy & v1.x & Numerical computation in simulator and metrics scripts & BSD \\
\hline
Pandas & v2.x & Metrics aggregation and results table generation & BSD \\
\hline
Anthropic Claude Haiku & API (2024) & Zero-shot LLM agent for Phase~9 semantic baseline & Commercial API terms of service \\
\hline
\end{tabular}
\end{table}

All academic papers cited in this report were accessed through publicly available sources or the University of Leeds library. No software developed for this project relies on proprietary algorithms or datasets. The project codebase is available to the supervisor and assessor via the git repository linked in the project submission.

The related work comparison table from the original report is reproduced in Table~\ref{tab:related-work-comparison} below for reference.

```

Note: Leave the remainder of `appendices.tex` (from `\chapter{Phase 0 and Phase 1 Experimental Validation}` onwards) entirely unchanged.

- [ ] **Step 2: Verify the related-work-comparison table reference still resolves**

The existing Appendix B content referenced `\label{tab:related-work-comparison}`. This table is now inside the rewritten Appendix B. Confirm the label is still present in the file after editing.

- [ ] **Step 3: Commit**

```bash
git add report/appendices.tex
git commit -m "feat: rewrite Appendix A self-appraisal and Appendix B external materials"
```

---

## Task 9: Style pass and cross-reference check

**Files:**
- Review: all modified `.tex` files

This task has no new content to write. It is a consistency and style check.

- [ ] **Step 1: Check every new prose sentence in all modified files against style rules**

For each file modified in Tasks 1 to 8, read through all new prose and verify:
- No semi-colons appear in prose sentences
- No colons appear mid-sentence in prose (colons in table headers and LaTeX environments are fine)
- No em-dashes appear
- None of the banned words appear: delve, nuanced, robust, leverage, pivotal, crucial, comprehensive, groundbreaking, streamline, navigate, tapestry, foster, underscore
- No banned openers: "Additionally,", "Furthermore,", "Moreover,", "It is worth noting that"
- No sentence does more than one thing (split any compound sentences found)
- British English spellings throughout

Fix any violations found before proceeding.

- [ ] **Step 2: Check all cross-references resolve**

Verify the following labels are defined somewhere in the report and referenced correctly:

| Reference | Expected location |
|-----------|------------------|
| `\ref{appendix:phase01}` | appendices.tex Appendix C chapter |
| `\ref{appendix:selfappraisal}` | appendices.tex Appendix A |
| `\ref{appendix:external}` | appendices.tex Appendix B |
| `\ref{sec:rq1}`, `\ref{sec:rq2}`, `\ref{sec:rq3}` | chapter4.tex |
| `\ref{sec:prelim}` | chapter4.tex |
| `\ref{sec:discussion}`, `\ref{sec:validity}`, `\ref{sec:futurework}` | chapter4.tex |
| `\ref{sec:marl-reward-modes}` | chapter2.tex |
| `\ref{sec:alpha-mixing}` | chapter2.tex |
| `\ref{sec:phase4-10-infra}` | chapter3.tex |
| `\ref{tab:related-work-comparison}` | appendices.tex |

- [ ] **Step 3: Verify all new citations exist in refs.bib**

Check that every `\citet{}` or `\citep{}` key used in new content exists in `report/refs.bib`:

Keys used in new content: `tan1993multiagent`, `panait2005cooperative`, `wang2022irat`, `wolpert2002optimal`, `devlin2014potential`, `agogino2004unifying`, `yang2018meanfield`, `claus1998dynamics`, `schroeaderdewitt2020ippo`, `son2019qtran`, `hughes2018inequity`, `rashid2018qmix`, `foerster2017stabilising`, `matignon2012independent`, `busoniu2008survey`, `henderson2018deep`, `agarwal2021statistical`.

All of these are present in `report/refs.bib`. No additions to refs.bib are required.

- [ ] **Step 4: Final commit**

```bash
git add report/chapters/chapter1.tex report/chapters/chapter2.tex report/chapters/chapter3.tex report/chapters/chapter4.tex report/appendices.tex report/summary.tex report/config.tex
git commit -m "fix: style pass and cross-reference verification across all rewritten sections"
```

---

## Self-review

**Spec coverage check:**

| Spec requirement | Task covering it |
|-----------------|-----------------|
| Fix bibliographystyle to agsm | Task 1 |
| Rewrite summary.tex for full 10-phase scope | Task 1 |
| Ch 1: add Tan, Panait, Wang references | Task 2 |
| Ch 1: rewrite contributions to cover 6 items | Task 2 |
| Ch 1: add Obj 6 and 7 | Task 2 |
| Ch 1: update H1, H2, H3 | Task 2 |
| Ch 2: add Section 2.7 MARL reward modes | Task 3 |
| Ch 3: fix TBC placeholder | Task 4 |
| Ch 3: tighten Phase 0 reference | Task 4 |
| Ch 3: add Section 3.8 Phase 4-10 infra | Task 4 |
| Ch 4: Section 4.1 preliminary validation table | Task 5 |
| Ch 4: Section 4.2 RQ1 (Phase 3 results) | Task 5 |
| Ch 4: Section 4.3 RQ2 (Phases 4-8) | Task 6 |
| Ch 4: Section 4.4 RQ3 (Phases 5, 9, 10) | Task 7 |
| Ch 4: Section 4.5 Discussion vs literature | Task 7 |
| Ch 4: Section 4.6 Validity threats | Task 7 |
| Ch 4: Section 4.7 Future work (QMIX, Devlin, complexity) | Task 7 |
| Appendix A: self-appraisal (process, reflection, legal/ethical) | Task 8 |
| Appendix B: external materials table | Task 8 |
| Style rules enforced throughout | Task 9 |
| Cross-references verified | Task 9 |

**Placeholder scan:** No TBD, TODO, or incomplete steps found. All tables contain actual data from the research findings.

**Type consistency:** All `\label{}` and `\ref{}` pairs use consistent keys throughout. Table labels follow the pattern `tab:phaseN-descriptor`. Section labels follow `sec:descriptor`.
