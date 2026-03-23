# F1 Strategy Simulator

This repository contains an F1 inspired simulator for reinforcement learning research.

## Current Focus

The current milestone is low complexity evaluation.

Low complexity means one DQN driver races one hard coded Base driver.

Medium complexity and high complexity are planned but not implemented yet.

- `medium`: add more competitors
- `high`: add tyre dynamics and pit stops

## Supported Agents

- `dqn`
- `base`
- `random`

`maddpg` is a gated stretch option. It currently falls back to `base`.

## DQN Variants

- `vanilla`
- `double`
- `dueling`
- `rainbow_lite`

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Configure `config.json`

Set `complexity.active_profile` to `low`.

3. Run simulator

```bash
python src/main.py
```

## Evaluation Script

Run train and evaluate with fixed seeds.

```bash
python scripts/evaluate_candidate.py --config config.json --train-runs 200 --eval-runs 200 --eval-seeds "101,202,303"
```

Optional complexity override

```bash
python scripts/evaluate_candidate.py --config config.json --complexity-profile low
```

## Benchmark Script

```bash
python scripts/run_benchmark_matrix.py --stage A --complexity-profile low
python scripts/run_benchmark_matrix.py --stage B --complexity-profile low
```

## Project Layout

- `src/main.py` entry point
- `src/simulator.py` race loop and logging
- `src/states.py` race state and agent wiring
- `src/runtime_profiles.py` complexity profile resolution
- `src/agents/DQN.py` DQN variants
- `scripts/evaluate_candidate.py` train and evaluate pipeline
- `scripts/run_benchmark_matrix.py` fair benchmark runner
- `RESEARCH.md` detailed config and experiment notes

