# Phase 3 — Independent Learner MARL: Run Protocol and Scientific Design
**Date:** 2026-03-26
**Gate prerequisite:** Gate G2 passed (Phase 2 complete, 36/36 trials present).
**Status:** Infrastructure implemented. Awaiting execution.

---

## 1. Scientific Rationale

Phase 2 established which DQN-family algorithm is most reliable against a fixed opponent. Phase 3 introduces the minimal step that makes the MARL framing of RQ1 valid: the opponent is now also a learning agent, adapting simultaneously. This is called **independent learner MARL (IL-MARL)** — the standard baseline in the MARL literature (Foerster et al., 2017).

**What changes from Phase 2:**
- The Base Agent opponent is replaced by a second concurrent DQN agent
- Both agents train from their own independent replay buffers after every episode
- Neither agent observes the other's Q-values, network weights, or training signal — they only experience the race outcome
- Non-stationarity is introduced: from Agent 1's perspective, the opponent's policy is changing throughout training, violating the stationarity assumption that Q-learning depends on

**What this tests that Phase 2 could not:**
- Whether win-rate stability erodes when the opponent is also adapting (RQ1 MARL component)
- Whether two agents with the same algorithm spontaneously develop differentiated strategies (RQ2 first signal)
- Whether rainbow-lite's stability advantage from Phase 2 transfers to the non-stationary setting

---

## 2. Controlled Variables

Everything held constant from Phase 2 to Phase 3, to isolate the effect of opponent type:

| Variable | Value |
|:---------|:------|
| Track | Spa-Francorchamps, 5 laps, 9 overtaking zones |
| Training budget | 500 episodes per trial |
| Evaluation budget | 150 races per seed |
| Seeds | 101, 202, 303 |
| Stochasticity levels | s0, s1, s2 |
| Network architecture | hidden_size=512, lr=7×10⁻⁴, γ=0.99 |
| Reward weights | outcome=2.0, tactical=0.05, penalty=2.0 |
| Complexity profile | `low_marl` (two DQN agents, no Base Agent) |

**Independent variable introduced:** opponent type (fixed heuristic Base Agent → concurrent DQN learner)

---

## 3. Algorithm Pairings

Two algorithm pairings are run. Both are same-algorithm matchups (both agents use the same DQN variant) to ensure the policy difference between agents is driven by non-stationarity dynamics, not algorithm asymmetry.

| Pairing | Agent 1 | Agent 2 | Purpose |
|:--------|:--------|:--------|:--------|
| Vanilla vs Vanilla | DQN_A1_vanilla | DQN_A2_vanilla | IL-MARL control — simplest non-stationary baseline |
| Rainbow vs Rainbow | DQN_A1_rainbow_lite | DQN_A2_rainbow_lite | IL-MARL primary — tests whether rainbow's stability survives non-stationarity |

Double DQN and dueling are excluded: double was brittle under noise (H3a disconfirmed), dueling had bimodal collapse at s0. Both were dropped after Phase 2.

---

## 4. Primary and Secondary Metrics

### Primary metric: Relative positional advantage

With two agents and positions always {1, 2}, the win rate of Agent 1 vs Agent 2 captures the full relative performance. This is reported as `win_rate_a1_vs_a2` with Wilson CI95. By symmetry, `win_rate_a2 = 1 - win_rate_a1`.

**Interpretation:** At initialisation, both agents have identical architectures and random weights. Any sustained departure from 0.50 win rate indicates that one agent has converged to a consistently better policy — a finding in itself. If win rates stay at 0.50 across all seeds, the two agents have reached a competitive equilibrium with no dominant strategy.

### Secondary metrics

**Non-stationarity signal:** Win rate of Agent 1 in the first third of evaluation races vs the last third. A shift > 0.05 in either direction indicates that the policies are still co-adapting during evaluation — evidence of non-stationarity that has not converged.

**Strategy differentiation index:** For each overtaking zone, the absolute difference in attempt rates between agents is averaged across all zones. A high index (> 0.25) indicates that the two agents have spontaneously specialised to different zones — the first emergent behaviour signal relevant to RQ2.

**Risk differentiation index:** Absolute difference in CONSERVATIVE / NORMAL / AGGRESSIVE proportions between agents. Compared against Phase 2 single-agent baselines to detect whether competition amplifies or suppresses risk differences.

**Per-agent diagnostics:** Reported separately for each agent — risk distribution, zone behaviour, overtake success rate, DNF rate. This allows direct comparison of each agent's profile against its Phase 2 single-agent equivalent.

---

## 5. Hypotheses Under Test

**H_ns1 — Non-stationarity erodes positional stability:**
If the non-stationarity signal drift is > 0.05 magnitude across multiple seeds, the agents have not converged by the end of training. This would indicate that 500 episodes is insufficient for IL-MARL convergence, requiring a longer training budget in Phase 4.

**H_ns2 — Rainbow-lite's stability advantage transfers to MARL:**
Rainbow-lite showed the flattest win-rate profile across s0/s1/s2 in Phase 2 (range 0.005). If rainbow vs rainbow shows lower across-seed variance in relative advantage than vanilla vs vanilla, the PER + n-step stability mechanism transfers to the non-stationary setting.

**H_diff — Same-algorithm agents develop differentiated strategies:**
Even with identical initialisation conditions (same algorithm, same stochasticity), different random seeds should cause minor early differences that compound under non-stationary training. A zone differentiation index > 0.10 in at least one pairing would confirm that IL-MARL introduces genuine behavioural diversity — the precursor to Phase 4's emergent strategy analysis.

**Falsification condition for H_ns1:** If drift < 0.02 across all seeds and pairings, the 500-episode budget is sufficient for convergence and Phase 4 can proceed with the same budget.

---

## 6. Run Commands

All commands use the `evaluate_marl.py` script. Run each shell independently in parallel. Each command produces one output JSON file.

### Vanilla vs Vanilla — s0 (3 seeds)

```bash
# Shell 1
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 500 --eval-runs 150 --train-seed 101 --eval-seeds 101 --stochasticity-level s0 --out metrics/phase3/vanilla_marl_s0_s101.json

# Shell 2
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 500 --eval-runs 150 --train-seed 202 --eval-seeds 202 --stochasticity-level s0 --out metrics/phase3/vanilla_marl_s0_s202.json

# Shell 3
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 500 --eval-runs 150 --train-seed 303 --eval-seeds 303 --stochasticity-level s0 --out metrics/phase3/vanilla_marl_s0_s303.json
```

### Rainbow vs Rainbow — s0 (3 seeds)

```bash
# Shell 4
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 101 --eval-seeds 101 --stochasticity-level s0 --out metrics/phase3/rainbow_marl_s0_s101.json

# Shell 5
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 202 --eval-seeds 202 --stochasticity-level s0 --out metrics/phase3/rainbow_marl_s0_s202.json

# Shell 6
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 303 --eval-seeds 303 --stochasticity-level s0 --out metrics/phase3/rainbow_marl_s0_s303.json
```

### Robustness tracks — s1 and s2 (run after s0 Gate G3 passes)

```bash
# Vanilla s1
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 500 --eval-runs 150 --train-seed 101 --eval-seeds 101 --stochasticity-level s1 --out metrics/phase3/vanilla_marl_s1_s101.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 500 --eval-runs 150 --train-seed 202 --eval-seeds 202 --stochasticity-level s1 --out metrics/phase3/vanilla_marl_s1_s202.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 500 --eval-runs 150 --train-seed 303 --eval-seeds 303 --stochasticity-level s1 --out metrics/phase3/vanilla_marl_s1_s303.json

# Vanilla s2
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 500 --eval-runs 150 --train-seed 101 --eval-seeds 101 --stochasticity-level s2 --out metrics/phase3/vanilla_marl_s2_s101.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 500 --eval-runs 150 --train-seed 202 --eval-seeds 202 --stochasticity-level s2 --out metrics/phase3/vanilla_marl_s2_s202.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_vanilla_marl.json --train-runs 500 --eval-runs 150 --train-seed 303 --eval-seeds 303 --stochasticity-level s2 --out metrics/phase3/vanilla_marl_s2_s303.json

# Rainbow s1
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 101 --eval-seeds 101 --stochasticity-level s1 --out metrics/phase3/rainbow_marl_s1_s101.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 202 --eval-seeds 202 --stochasticity-level s1 --out metrics/phase3/rainbow_marl_s1_s202.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 303 --eval-seeds 303 --stochasticity-level s1 --out metrics/phase3/rainbow_marl_s1_s303.json

# Rainbow s2
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 101 --eval-seeds 101 --stochasticity-level s2 --out metrics/phase3/rainbow_marl_s2_s101.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 202 --eval-seeds 202 --stochasticity-level s2 --out metrics/phase3/rainbow_marl_s2_s202.json
conda run -n f1StrategySim python scripts/evaluate_marl.py --config metrics/phase3/config_rainbow_marl.json --train-runs 500 --eval-runs 150 --train-seed 303 --eval-seeds 303 --stochasticity-level s2 --out metrics/phase3/rainbow_marl_s2_s303.json
```

**Total: 18 trials** (2 pairings × 3 seeds × 3 stochasticity levels). Run s0 first; proceed to s1/s2 only after Gate G3 passes.

---

## 7. What to Look for in Outputs

Each output JSON contains `metrics.non_stationarity_signal.drift`, `metrics.strategy_differentiation`, `metrics.win_rate_a1_vs_a2`, and per-agent diagnostics.

**Immediate red flags (investigate before proceeding to s1/s2):**
- `win_rate_a1_vs_a2.mean` consistently at exactly 0.500 across all 3 seeds with near-zero variance → degenerate equilibrium; agents may not be training properly
- `strategy_differentiation.zone_differentiation_index` = 0.000 across all seeds → both agents have identical policies; inspect model checkpoints
- `metrics.agent1.dnf_rate` or `metrics.agent2.dnf_rate` > 0 → simulator contract violation; halt and diagnose

**Healthy signals:**
- Win rate drifts to 0.55–0.65 for one agent across most seeds → one random initialisation produces a consistently dominant early policy; normal in IL-MARL
- Zone differentiation index > 0.10 → agents specialising to different zones spontaneously; RQ2 first evidence
- Non-stationarity drift magnitude < 0.05 → agents have co-converged within the training budget; stable enough for Phase 4

---

## 8. Gate G3 Pass Criteria

| Criterion | Required |
|:----------|:---------|
| All 6 s0 trials complete without training collapse | DNF rate = 0 for both agents |
| At least one pairing shows non-zero positional advantage at s0 | win_rate_a1 ≠ 0.500 with CI95 excluding 0.500 in at least 2/3 seeds |
| Strategy differentiation index interpretable | At least one pairing shows index > 0.05 in at least one seed |
| Non-stationarity drift interpretable | Drift reported as a number, not NaN/error, in all 6 s0 trials |
| Phase 2 risk distribution patterns recognisable per-agent | Each agent's individual risk profile broadly consistent with its algorithm's Phase 2 profile |

**If Gate G3 passes:** Proceed to s1/s2 robustness runs, then Phase 4 implementation.

**If Gate G3 fails on win rate (all 0.500):** Increase training budget to 750 episodes and re-run s0. If still degenerate at 750, this is a genuine finding — the two agents reach a symmetric Nash equilibrium and the non-stationarity effect is absent at low complexity. Report as such and proceed directly to Phase 4 with the pace asymmetry as the symmetry-breaking mechanism.

---

## 9. What Phase 3 Produces for the Dissertation

**Chapter 3:** Phase 3 implementation section is now complete — `runtime_profiles.select_low_marl_competitors()`, the `low_marl` complexity profile, and `evaluate_marl.py` with its per-agent diagnostic machinery.

**Chapter 4 Phase 3 section** will report:
1. Whether non-stationarity erodes relative positional stability (H_ns1)
2. Whether rainbow-lite's stability advantage survives the MARL setting (H_ns2)
3. Whether same-algorithm agents spontaneously differentiate their zone strategies (H_diff)
4. The stochasticity robustness ordering — does the Phase 2 finding (rainbow > vanilla > double) hold when both agents face a learning opponent?

**Direct RQ coverage:**
- RQ1 (robustness under stochasticity): Phase 3 provides the MARL-framed answer — single-agent robustness from Phase 2 is now tested under the additional pressure of a co-adapting opponent
- RQ2 (emergent strategic behaviours): Phase 3 provides the first empirical window — if zone differentiation index > 0.10, two agents learning independently have begun to specialise
