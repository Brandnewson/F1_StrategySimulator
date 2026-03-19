# F1 Strategy Simulator

This repository is a research-oriented racing simulator for studying reinforcement learning in a competitive, stochastic environment inspired by Formula 1 race dynamics.

The project currently supports:
- Rule-based baseline agents (`base`)
- Random agents (`random`)
- DQN-family learners (`dqn`) with config-driven variants:
  - `vanilla`
  - `double`
  - `dueling`
  - `rainbow_lite` (Double + Dueling + Prioritized Replay + n-step)

## Research Goal

The core goal is to compare how different RL approaches perform under uncertainty, interaction, and non-trivial race strategy tradeoffs.

For practical benchmarking today, the strongest supported workflow is DQN-family comparison against fixed baselines with consistent seed/budget control.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure experiment settings in `config.json`.

3. Run the simulator:
```bash
python src/main.py
```

## Common Workflows

### 1) Batch Train/Evaluate with Visual Results
Set in `config.json`:
- `simulator.method = "batch"`
- `simulator.runs = <N>`
- `simulator.agent_mode = "training"` or `"evaluation"`

Then run:
```bash
python src/main.py
```

### 2) Replay Existing Results Without New Runs
Set:
- `simulator.runs = 0`
- `simulator.visualise_from_run_name = "<existing_log_folder_name>"`

Then run:
```bash
python src/main.py
```

### 3) Programmatic Evaluation for Research Metrics
Use:
```bash
python scripts/evaluate_candidate.py --config config.json --train-runs 200 --eval-runs 200 --eval-seeds "101,202,303"
```

This produces machine-readable metrics JSON including:
- `win_rate_vs_baseline` (primary objective)
- confidence intervals
- race-quality and tactical metrics
- algorithm metadata (DQN variant + options)

### 4) Fair Multi-Algorithm Benchmark Runs
Use:
```bash
python scripts/run_benchmark_matrix.py --stage A
python scripts/run_benchmark_matrix.py --stage B
```

Default comparison set is:
- `vanilla,double,dueling,rainbow_lite`

The script generates per-trial configs/results plus a summary table and fairness audit.

## How to Choose Between DQN Variants

Choosing an algorithm should be hypothesis-driven, not just leaderboard-driven. A practical decision process:

1. Start with `vanilla` as the control arm.
- You need this to make causal claims about improvements.

2. Try `double` when you suspect value overestimation.
- Useful when Q-values become over-optimistic and unstable.

3. Try `dueling` when many states have similar action value.
- Helps when state-value estimation matters more than action-advantage in many situations.

4. Try `rainbow_lite` when you want stronger sample efficiency and robustness.
- PER can focus learning on informative transitions.
- n-step can propagate sparse/lagged reward faster.
- Combined with double + dueling, this is usually the strongest default challenger.

Potential ways to choose between them in practice:

- Statistical superiority:
  - Prefer the algorithm whose confidence interval is meaningfully above `vanilla`.
- Stability:
  - Lower variance across training seeds and evaluation seeds is often better for research reliability.
- Behavioral quality:
  - Compare overtake success rate, DNF rate, and position dynamics, not only win rate.
- Compute budget:
  - If compute is tight, `double` or `dueling` can be lower-complexity alternatives to `rainbow_lite`.

Recommended minimum for fair claims:
- Use common train/eval seed sets across all arms.
- Keep race settings, opponents, and budget identical across arms.
- Report confidence intervals, not only means.

## Project Layout

- `src/main.py`: entrypoint
- `src/simulator.py`: race loop, logging, visualization, batch execution
- `src/states.py`: race/driver state initialization and agent wiring
- `src/agents/DQN.py`: DQN variants and training logic
- `scripts/evaluate_candidate.py`: train/eval pipeline with research metrics
- `scripts/run_benchmark_matrix.py`: staged fair benchmark runner
- `config.json`: experiment configuration
- `RESEARCH.md`: detailed `config.json` parameter guide

## Notes for New Researchers

- Keep one algorithm change at a time when comparing methods.
- Keep one baseline opponent policy fixed while benchmarking.
- Treat single-seed conclusions as exploratory only.
- Prefer reproducible sweeps (named runs, fixed seeds, versioned configs).

## Additional Documentation

- `RESEARCH.md`: parameter-by-parameter configuration guide and tuning intuition
- `scripts/AUTORESEARCH_README.md`: autonomous iteration workflow
- `AUTORESEARCH_QUICKSTART.md`: quick setup for autoresearch tooling
