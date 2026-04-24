# F1 Strategy Simulator

An F1-inspired reinforcement learning simulator for studying overtaking strategy, incentive design, and multi-agent coordination under controlled complexity and stochasticity.

Last updated: 2026-04-21

## What Is Implemented

Implemented runtime profiles:

- `low`: 1 `dqn` vs 1 `base`
- `low_marl`: 2 `dqn` agents in zero-sum self-play
- `low_marl_vs_base`: 2 `dqn` agents + 1 `base`
- `low_marl_3dqn_vs_base`: 3 `dqn` agents + 1 `base`
- `low_marl_teams`: 4 `dqn` agents in two teams + 1 `base`
- `low_llm_vs_base`: 1 `llm` vs 1 `base`

Not yet implemented:

Simulator complexity levels of:
- `medium`
- `high`

If `medium` or `high` is requested, runtime falls back to `low`.

## Supported Agents

- `dqn`
- `base`
- `llm`
- `random`

Supported DQN variants:

- `vanilla`
- `double`
- `dueling`
- `rainbow_lite`

## Install

```bash
pip install -r requirements.txt
```

Optional test tooling:

```bash
python -m pytest tests/test_phase0_integrity.py tests/test_phase45_integrity.py
```

## Real-Time Vs Batch Simulation

The top-level simulator entry point reads `config.json` and honours:

- `simulator.method = "real-time"`: visualised step-through simulation
- `simulator.method = "batch"`: headless batch execution

Run it with:

```bash
python src/main.py
```

Important distinction:

- `src/main.py` uses whatever `simulator.method` is set in `config.json`
- the evaluator scripts force `batch` mode internally so experimental runs are headless and reproducible

So:

- use `src/main.py` for manual inspection or visual demos
- use the `scripts/` evaluators for dissertation-style experiments

## Config-Driven Contracts

The experiment contract lives in `config.json`:

- `feedback`: observation schema and active features by complexity
- `reward`: reward weights, normalization, and per-complexity activation
- `protocol`: stage budgets, seed sets, benchmark matrix
- `protocol.benchmark_contract`: primary vs robustness tracks, reruns, promotion-gate settings
- `stochasticity`: named presets `s0`, `s1`, `s2`
- `marl`: reward-sharing mode, alpha values, curriculum, team alphas

For benchmark-generated single-agent comparisons, per-trial configs are derived from a common base config and vary only in the selected algorithm and algorithm-specific options.

## Quick Start For Examiners

These are the fastest useful reproductions from a clean checkout.

### 1. Reproduce A Fair Vanilla Vs Double Smoke Comparison

This is the quickest way to verify the benchmark runner, config generation, and artifact structure:

```bash
python scripts/run_benchmark_matrix.py --config config.json --stage smoke --track low_primary --algos vanilla,double --train-runs 2 --eval-runs 2 --train-seeds 101 --eval-seeds 101 --repro-reruns 1 --out-dir metrics/examiner_smoke
```

What this writes:

- `metrics/examiner_smoke/.../configs/`: generated per-trial configs
- `metrics/examiner_smoke/.../raw/`: per-trial metrics JSON
- `metrics/examiner_smoke/.../summary.json`
- `metrics/examiner_smoke/.../summary.md`

The raw metrics JSON now includes:

- `config_snapshot`: the resolved experimental contract
- `runtime_config_snapshot`: the exact runtime config after telemetry naming adjustments
- `protocol`, `reward_contract`, `feedback_contract`, `stochasticity_contract`
- `phases`: executed train/eval seeds and budgets

### 2. Reproduce One Single-Agent Evaluation

This runs the current `dqn_params.algo` from `config.json` against the Base agent:

```bash
python scripts/evaluate_candidate.py --config config.json --complexity-profile low --stochasticity-level s0 --train-runs 20 --eval-runs 20 --train-seed 101 --eval-seeds 101 --out metrics/examiner_candidate.json
```

To compare algorithms fairly, use the benchmark runner rather than editing settings by hand between runs.

### 3. Reproduce The Main MARL Profiles

Zero-sum MARL:

```bash
python scripts/evaluate_marl.py --config config.json --complexity-profile low_marl --train-runs 20 --eval-runs 20 --train-seed 101 --eval-seeds 101 --stochasticity-level s0 --out metrics/examiner_low_marl.json
```

Non-zero-sum MARL with a Base adversary:

```bash
python scripts/evaluate_marl.py --config config.json --complexity-profile low_marl_vs_base --alpha 0.30 --train-runs 20 --eval-runs 20 --train-seed 101 --eval-seeds 101 --stochasticity-level s0 --out metrics/examiner_low_marl_vs_base.json
```

Three-DQN scaling profile:

```bash
python scripts/evaluate_marl.py --config config.json --complexity-profile low_marl_3dqn_vs_base --alpha 0.15 --train-runs 20 --eval-runs 20 --train-seed 101 --eval-seeds 101 --stochasticity-level s0 --out metrics/examiner_low_marl_3dqn_vs_base.json
```

Team-based five-agent profile:

```bash
python scripts/evaluate_marl.py --config config.json --complexity-profile low_marl_teams --team-a-alpha 0.15 --team-b-alpha 0.15 --train-runs 20 --eval-runs 20 --train-seed 101 --eval-seeds 101 --stochasticity-level s0 --out metrics/examiner_low_marl_teams.json
```

### 4. Reproduce The Full Single-Agent Benchmark Matrix

Use protocol defaults from `config.json`:

```bash
python scripts/run_benchmark_matrix.py --config config.json --stage benchmark
```

Target just the clean ranking track:

```bash
python scripts/run_benchmark_matrix.py --config config.json --stage benchmark --track low_primary
```

Target the robustness tracks:

```bash
python scripts/run_benchmark_matrix.py --config config.json --stage benchmark --track low_robustness
```

## LLM Reproduction

Phase 9 requires an Anthropic API key:

```bash
$env:ANTHROPIC_API_KEY="..."
```

Then run:

```bash
python scripts/evaluate_llm_agent.py --config config.json --eval-runs 20 --eval-seeds 101 --stochasticity-level s0 --alpha competitive --out metrics/examiner_llm.json
```

## Notes On Interpreting Smoke Runs

The short commands above are for reproducibility and pipeline verification, not for dissertation-grade statistical claims.

Use smoke runs to confirm:

- the simulator runs end to end
- the correct profile is selected
- the artifact structure is complete
- config snapshots are embedded in the metrics JSON

Use the protocol defaults or larger budgets for substantive result replication.

## Outputs

- `logs/`: run logs, including `race_results.jsonl` and metadata
- `metrics/`: evaluator outputs, benchmark summaries, and phase artifacts
- `models/`: saved DQN checkpoints

## Project Layout

- `config.json`: master experiment contract
- `src/main.py`: top-level simulator entry point
- `src/simulator.py`: race loop, stochasticity, reward logic, batch and real-time execution
- `src/runtime_profiles.py`: complexity-profile resolution
- `src/feedback.py`: config-driven observation builder
- `src/states.py`: race state and agent wiring
- `src/agents/DQN.py`: DQN-family implementations
- `src/agents/LLM.py`: LLM planner integration
- `scripts/evaluate_candidate.py`: single-agent train/evaluate pipeline
- `scripts/evaluate_marl.py`: MARL train/evaluate pipeline
- `scripts/evaluate_llm_agent.py`: Phase 9 LLM evaluation
- `scripts/run_benchmark_matrix.py`: fair single-agent benchmark runner
- `tests/test_phase0_integrity.py`: simulator and telemetry integrity tests
- `tests/test_phase45_integrity.py`: MARL mechanism integrity tests
