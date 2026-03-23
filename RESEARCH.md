# RESEARCH.md

This guide documents the current experiment interface for the simulator.

## 1) Current Research Scope

Current work targets low complexity experiments first.

Low complexity means one DQN agent against one Base agent.

This setup is used to confirm that DQN variants learn in the simulator before adding more complexity.

Medium and high complexity are placeholders.

## 2) Top Level Config

### `complexity`

- `complexity.active_profile`: `low`, `medium`, or `high`
- only `low` is implemented today
- `medium` and `high` fall back to `low`

### `competitors`

Low complexity automatically selects:

- one competitor with `agent="dqn"`
- one competitor with `agent="base"`

If either is missing the run fails with a clear error.

### `simulator`

Important keys:

- `runs`
- `method`
- `agent_mode`
- `tick_rate`
- `tick_duration`

### `dqn_params`

Supported algorithms:

- `vanilla`
- `double`
- `dueling`
- `rainbow_lite`

## 3) Evaluation Discipline

Use fixed seed sets.

Use equal train and evaluation budgets across compared algorithms.

Report confidence intervals. Do not rely on single runs.

## 4) Useful Commands

Single candidate evaluation:

```bash
python scripts/evaluate_candidate.py --config config.json --train-runs 200 --eval-runs 200 --eval-seeds "101,202,303" --complexity-profile low
```

Benchmark matrix:

```bash
python scripts/run_benchmark_matrix.py --stage A --complexity-profile low
python scripts/run_benchmark_matrix.py --stage B --complexity-profile low
```

## 5) Planned Incremental Roadmap

1. Confirm DQN learning at low complexity.
2. Add medium complexity with more competitors.
3. Add high complexity with tyre dynamics and pit stops.
4. Add explicit stochasticity controls after complexity tiers are stable.

