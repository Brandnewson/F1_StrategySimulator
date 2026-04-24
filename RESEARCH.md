# RESEARCH.md

This document defines the current research workflow and experiment interface.

Last updated: 2026-04-07

## 1) Research Scope

The research investigates how shared incentive shapes emergent strategy in multi-agent reinforcement learning, using F1 racing as the domain. Ten experimental phases (Phases 1-10) are complete, spanning single-agent benchmarking, cooperative MARL, scaling analysis, curriculum learning, LLM baselines, and reward shaping for credit assignment.

See `research_findings/full_research_journey.md` for the complete narrative and results.

### Complexity

- Implemented and used: `low` (all phases)
- Future-ready only: `medium`, `high` (runtime falls back to `low`)

### Agent configurations used across phases

| Phases | Setup |
|--------|-------|
| 1-2 | 1 DQN vs 1 Base (single-agent benchmark) |
| 3-4 | 2 DQN head-to-head (zero-sum MARL) |
| 5 | 2 DQN + 1 Base (non-zero-sum, common adversary) |
| 6 | 4 DQN (2 teams) + 1 Base (multi-team scaling) |
| 7-8, 10 | 3 DQN + 1 Base (boundary characterisation, curriculum, difference rewards) |
| 9 | 1 LLM vs 1 Base (semantic baseline) |

The objective is finish-first race performance. Tactical overtakes are helper behaviour, not the terminal objective.

## 2) Config Contracts (Single Source of Truth)

### `complexity`

- `complexity.active_profile`: `low | medium | high`
- today only `low` executes

### `feedback`

Config-driven state schema:

- `feedback.schema_version`
- `feedback.active_profile`
- `feedback.features_by_complexity.{low,medium,high}`

Low default features:

1. `zone_distance_norm`
2. `gap_to_ahead_norm`
3. `zone_difficulty`
4. `has_car_ahead`
5. `current_position_norm`
6. `laps_remaining_norm`

### `reward`

Invariant reward schema across tiers:

- `outcome`
- `persistent_position`
- `tactical`
- `pace`
- `tyre_pit`
- `penalty`

Config keys:

- `reward.schema_version`
- `reward.weights`
- `reward.normalization.{low,medium,high}`
- `reward.component_activation_by_complexity.{low,medium,high}`
- `reward.tactical`

Low complexity activation should keep:

- active: `outcome`, `persistent_position`, `tactical`, `penalty`
- inactive: `pace`, `tyre_pit`

### `stochasticity`

- `stochasticity.active_level`: `s0 | s1 | s2`
- `stochasticity.levels.*` controls probability scales/noise and probability clipping bounds

Stochasticity is designed as an axis independent of complexity.

### `protocol`

Experiment governance:

- `protocol.stage_order`
- `protocol.stochasticity_order`
- `protocol.seed_sets` (`smoke`, `candidate`, `benchmark`)
- `protocol.train_runs`
- `protocol.eval_runs`
- `protocol.comparison_matrix.algorithms`
- `protocol.benchmark_contract`

`benchmark_contract` is the Stage 1-2 lock:

- `tracks.low_primary` = Track A (low+s0, primary ranking)
- `tracks.low_robustness` = Track B (low+s1/s2, robustness only)
- lock flags for immutable algorithm list / seeds / budgets
- reproducibility rerun count
- promotion gate requirements for medium-stage go/no-go
- fail-fast validation before run start (invalid contract combinations stop execution)

## 3) Algorithm-Neutral Comparison Rules

To keep comparisons methodological:

1. Reward code must not branch on DQN algorithm identity.
2. Feedback contract must be identical across compared algorithms in the same run.
3. Train/eval budgets and seeds must be shared across compared algorithms.
4. Complexity and stochasticity settings must be shared within each comparison cell.

Only learning internals differ between `vanilla`, `double`, `dueling`, and `rainbow_lite`.

## 4) How To Run Research Pipelines

### Single Candidate Evaluation

Protocol-driven defaults from `config.json`:

```bash
python scripts/evaluate_candidate.py --config config.json
```

Typical override pattern:

```bash
python scripts/evaluate_candidate.py --config config.json --complexity-profile low --stochasticity-level s0 --eval-seeds "101,202,303,404,505"
```

### Fair Benchmark Matrix

```bash
python scripts/run_benchmark_matrix.py --config config.json --stage smoke
python scripts/run_benchmark_matrix.py --config config.json --stage candidate
python scripts/run_benchmark_matrix.py --config config.json --stage benchmark
```

Notes:

- Legacy `--stage A/B` aliases remain supported.
- With benchmark lock enabled, run budgets/seeds/algorithms/tracks are governed by `config.json` and CLI overrides are blocked.
- Use `--track low_primary` or `--track low_robustness` to execute one track only.

## 5) Suggested Execution Ladder

1. Track A: low complexity at `s0` for primary algorithm ranking.
2. Track B: same low complexity at `s1`, then `s2` for robustness evidence.
3. Promote to medium only if gate passes: fairness audit, CI-stable Track A ranking, reproducibility consistency, and Track B robustness evidence.
4. After medium implementation exists, repeat the same two-track logic before any high-tier claims.

Primary endpoint remains finish-first outcomes (win rate / finish-position delta). Tactical statistics and reward-component diagnostics are secondary.

## 6) Output Artifacts To Track

- `logs/<run>/metadata.json`: active complexity, reward contract, feedback config, protocol, stochasticity level
- `logs/<run>/race_results.jsonl`: per-run/per-driver outcomes and decision summaries
- `metrics/latest_candidate_metrics.json` (or chosen `--out`): evaluation summary including reward diagnostics
- `metrics/benchmarks/.../summary.json` and `summary.md`: cross-algorithm comparison with fairness audit, reproducibility checks, robustness trends, and promotion gate result

## 7) MARL Reward Modes

The `marl.reward_mode` config key controls credit assignment:

| Mode | Formula | Phases | Outcome |
|------|---------|:------:|---------|
| `alpha` | `(1-a)*own + a*teammate` | 3-8 | Works at N=3 (+83%), cliff at N=4, fails at N=5 |
| `difference` | `own + (own - team_mean)` | 10 | Worse than IQL (-16%) — competitive incentive |
| `qmix` | Monotonic mixing network | Future | Architectural CTDE — not yet implemented |

The MARL evaluation script supports all modes:

```bash
python scripts/evaluate_marl.py \
  --complexity-profile low_marl_3dqn_vs_base \
  --reward-mode difference \
  --alpha 0.0 \
  --stochasticity-level s0 \
  --train-runs 500 --eval-runs 150 \
  --eval-seeds "101" \
  --out metrics/phase10a/example.json
```

## 8) Newcomer Checklist

1. Confirm dependencies installed (`pip install -r requirements.txt`).
2. Confirm `config.json` has `complexity.active_profile = low`.
3. Run one candidate evaluation.
4. Inspect reward diagnostics and finish-first metrics.
5. Run one benchmark stage and check fairness audit is true.
6. Read `research_findings/full_research_journey.md` for the complete experimental arc.
