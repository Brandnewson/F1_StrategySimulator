# RESEARCH.md

This document defines the current research workflow and experiment interface.

Last updated: 2026-03-23

## 1) Research Scope

Current dissertation execution target is low-complexity experiments first, with future-ready config scaffolding for medium/high complexity and higher stochasticity.

- Implemented now: `low`
- Future-ready only: `medium`, `high` (runtime currently falls back to `low`)

Low complexity setup:

- one `dqn` competitor
- one `base` competitor

The objective is finish-first race performance. Tactical overtakes are helper behavior, not the terminal objective.

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
- With protocol present, run budgets and seeds come from `config.json` unless explicitly overridden via CLI.

## 5) Suggested Execution Ladder

1. Low complexity at `s0` until stable finish-first learning.
2. Low complexity at `s1`, then `s2` to quantify robustness drop.
3. Enable medium complexity profile when implemented, repeat `s0 -> s1 -> s2`.
4. Enable high complexity profile when implemented, repeat `s0 -> s1 -> s2`.

Primary endpoint remains finish-first outcomes (win rate / finish-position delta). Tactical statistics and reward-component diagnostics are secondary.

## 6) Output Artifacts To Track

- `logs/<run>/metadata.json`: active complexity, reward contract, feedback config, protocol, stochasticity level
- `logs/<run>/race_results.jsonl`: per-run/per-driver outcomes and decision summaries
- `metrics/latest_candidate_metrics.json` (or chosen `--out`): evaluation summary including reward diagnostics
- `metrics/benchmarks/.../summary.json` and `summary.md`: cross-algorithm comparison and fairness audit

## 7) Newcomer Checklist

1. Confirm dependencies installed (`pip install -r requirements.txt`).
2. Confirm `config.json` has `complexity.active_profile = low`.
3. Run one candidate evaluation.
4. Inspect reward diagnostics and finish-first metrics.
5. Run one benchmark stage and check fairness audit is true.
