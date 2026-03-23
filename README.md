# F1 Strategy Simulator

An F1-inspired reinforcement learning simulator focused on strategy learning under controlled complexity and stochasticity.

Last updated: 2026-03-23

## Current Implementation Status

- Implemented: `low` complexity
- Not yet implemented (future-ready profiles): `medium`, `high`
- If `medium` or `high` is requested today, runtime falls back to `low`

Low complexity currently means one `dqn` driver versus one hard-coded `base` driver. The key goal is still finish-first race outcome, not overtake count maximization.

## Supported Agents

- `dqn`
- `base`
- `random`

`maddpg` is currently gated and falls back to `base`.

## DQN Variants

- `vanilla`
- `double`
- `dueling`
- `rainbow_lite`

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

Run the simulator using `config.json`:

```bash
python src/main.py
```

## Config-Driven Contracts

The experiment interface is now externalized in `config.json`:

- `feedback`: observation schema and active features by complexity
- `reward`: invariant finish-first reward contract, component weights, normalization, activation
- `protocol`: stage order, seed sets, train/eval budgets, algorithm comparison matrix
- `stochasticity`: named presets (`s0`, `s1`, `s2`) for overtake randomness/noise

This keeps environment/reward/feedback/protocol identical across DQN variants, so comparisons remain methodological.

## How-To: Evaluate One Candidate

Uses protocol defaults from `config.json` when run counts/seeds are not passed:

```bash
python scripts/evaluate_candidate.py --config config.json
```

Common overrides:

```bash
python scripts/evaluate_candidate.py --config config.json --complexity-profile low --stochasticity-level s0
python scripts/evaluate_candidate.py --config config.json --train-runs 200 --eval-runs 200 --eval-seeds "101,202,303,404,505"
```

## How-To: Run Fair DQN Benchmark Matrix

Use protocol-driven stages:

```bash
python scripts/run_benchmark_matrix.py --config config.json --stage smoke
python scripts/run_benchmark_matrix.py --config config.json --stage candidate
python scripts/run_benchmark_matrix.py --config config.json --stage benchmark
```

Legacy aliases are still supported:

- `A` -> `smoke`
- `B` -> `benchmark`

Optional overrides:

```bash
python scripts/run_benchmark_matrix.py --config config.json --stage candidate --complexity-profile low --stochasticity-level s1
```

## How-To: Tune Reward/Feedback Without Code Changes

1. Edit `config.json`:
- `reward.weights` to rebalance component dominance
- `reward.component_activation_by_complexity.low` to control active components in low complexity
- `feedback.features_by_complexity.low` to add/remove low-tier state features
- `stochasticity.active_level` to select stress level (`s0/s1/s2`)
2. Re-run:
- `python scripts/evaluate_candidate.py --config config.json`
3. Review reward diagnostics in output metrics:
- per-component totals/means and variance

## Outputs

- `logs/` per-run simulator logs (`race_results.jsonl`, `metadata.json`)
- `metrics/` evaluation and benchmark summaries
- `models/` saved DQN checkpoints

## Project Layout

- `src/main.py` entry point
- `src/simulator.py` race loop, reward contract, stochasticity integration, telemetry logging
- `src/feedback.py` config-driven observation builder
- `src/states.py` race state + agent wiring (state dimension from active feedback features)
- `src/runtime_profiles.py` complexity profile resolution
- `src/agents/DQN.py` DQN family algorithms
- `scripts/evaluate_candidate.py` train/evaluate pipeline + reward diagnostics
- `scripts/run_benchmark_matrix.py` fair comparison runner
- `RESEARCH.md` research workflow and methodology notes
