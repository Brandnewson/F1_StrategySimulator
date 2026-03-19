# RESEARCH.md

This guide explains what you can change in `config.json`, what each parameter does, and what effect it may have on training/evaluation outcomes.

It is intended for researchers running controlled experiments.

## 1) Recommended Experiment Discipline

Before changing parameters:
- Keep a `vanilla` DQN control arm.
- Change one factor at a time when testing hypotheses.
- Keep seed sets and budgets identical across compared algorithms.
- Report means and confidence intervals, not single-run outcomes.

Good default benchmark script:
```bash
python scripts/run_benchmark_matrix.py --stage A
python scripts/run_benchmark_matrix.py --stage B
```

## 2) Top-Level `config.json` Keys

### `debugMode` (bool)
- Purpose: enables extra debug visuals/logging.
- Likely effect:
  - `true`: more plots/log output, slower iteration.
  - `false`: cleaner/faster experiment runs.

### `agent_review_mode` (bool)
- Purpose: toggles agent learning plots in simulator batch runs.
- Likely effect:
  - `true`: additional visual outputs from simulator.
  - `false`: cleaner unattended runs.

### `competitors` (list of objects)
- Purpose: defines drivers and their agent types.
- Each item fields:
  - `name` (string): driver identifier. Also influences model filename stem for DQN checkpoints.
  - `agent` (string): currently `dqn`, `base`, `random` are supported in practice. `ppo` is currently a placeholder that falls back to `base`.
  - `colour` / `color` (string): plotting color.
- Likely effect:
  - Agent assignment determines who learns and who acts as fixed baseline.
  - Name changes can change which checkpoint file gets loaded.

## 3) `simulator` Section

### `simulator.runs` (int)
- Purpose: number of race episodes to execute in `batch` mode.
- Likely effect:
  - Higher values improve estimate stability but increase runtime.
  - Set to `0` with `visualise_from_run_name` for replay-only visualization.

### `simulator.run_name` (string)
- Purpose: log folder name in `logs/`.
- Likely effect:
  - Improves traceability and reproducibility.

### `simulator.visualise_from_run_name` (string)
- Purpose: replay data source when `runs = 0`.
- Likely effect:
  - Lets you inspect old experiments without rerunning simulation.

### `simulator.method` (string)
- Values: `batch` or `real-time`.
- Purpose:
  - `batch`: faster non-live simulation loop.
  - `real-time`: animated visualization loop.

### `simulator.tick_rate` (int) and `simulator.tick_duration` (float)
- Purpose: simulation temporal resolution.
- Likely effect:
  - Smaller `tick_duration` can increase fidelity and compute cost.
  - Keep fixed across algorithm comparisons.

### `simulator.agent_mode` (string)
- Values: `training` or `evaluation`.
- Purpose:
  - `training`: DQN explores and updates.
  - `evaluation`: DQN behaves greedily and loads checkpoint if present.
- Likely effect:
  - Critical mode switch for train/eval separation.

### Optional: `simulator.checkpoint_tag` (string)
- Purpose: adds tagged checkpoint outputs during training saves.
- Likely effect:
  - Helps archive model snapshots per experiment.

## 4) `race_settings` Section

### `race_settings.total_laps` (int)
- Purpose: episode horizon length.
- Likely effect:
  - Longer races increase decision opportunities and variance.
  - Keep fixed when comparing algorithms.

### `race_settings.weather` (string)
- Purpose: currently informational in this code path.
- Likely effect:
  - No direct simulation effect yet unless future weather dynamics are implemented.

## 5) `track` Section

### `track.name` (string)
- Purpose: track identifier used in labels/log naming.

### `track.coordinate_file` (string path)
- Purpose: CSV path used for plotting and distance mapping.

### `track.distance` (float, km)
- Purpose: track length reference for lap and speed profile mapping.
- Likely effect:
  - Changing this without consistent mini-loop distances may distort dynamics.

### `track.num_corners` / `track.num_overtaking_zones` (int)
- Purpose: mostly metadata/legacy loaders.
- Likely effect:
  - Limited direct impact in current normalized runtime path.

### `track.mini_loops` (object of segments)
- Keys per loop:
  - `start_distance` (km)
  - `end_distance` (km)
  - `base_lap_time` (seconds contribution)
  - `name` (string label)
- Purpose:
  - Defines segment-level pace characteristics.
- Likely effect:
  - Strong influence on base speed profile and race timing.

### `track.overtakingZones` (object of zones)
- Keys per zone:
  - `difficulty` (0 to 1)
  - `name` (string label)
  - `distance_from_start` (km)
- Purpose:
  - Defines where and how difficult overtakes are.
- Likely effect:
  - Strongly shapes tactical risk-reward and overtaking outcomes.

## 6) `dqn_params` Section

### `dqn_params.algo` (string)
- Supported values:
  - `vanilla`
  - `double`
  - `dueling`
  - `rainbow_lite`
- Purpose: selects DQN variant implementation.
- Likely effect:
  - `vanilla`: baseline control.
  - `double`: reduces overestimation bias.
  - `dueling`: separates state-value and action-advantage estimation.
  - `rainbow_lite`: combines double + dueling + prioritized replay + n-step.

### `dqn_params.algo_options` (object)
- Purpose: variant-specific knobs.
- Currently relevant mainly for `rainbow_lite`:
  - `n_step` (int): return horizon for n-step targets.
  - `per_alpha` (float): prioritization strength in replay sampling.
  - `per_beta_start` (float): initial importance-sampling correction.
  - `per_beta_frames` (int): schedule length for beta anneal toward 1.0.
- Likely effect:
  - Larger `n_step`: faster reward propagation, potentially noisier targets.
  - Larger `per_alpha`: more focus on high-error samples.
  - Larger `per_beta_*`: stronger bias correction as training progresses.

### `dqn_params.hidden_size` (int)
- Purpose: network width.
- Likely effect:
  - Larger size can model richer policies but may increase variance/compute.

### `dqn_params.learning_rate` (float)
- Purpose: optimizer step size.
- Likely effect:
  - Too high: unstable learning.
  - Too low: slow convergence.

### `dqn_params.gamma` (float)
- Purpose: discount factor.
- Likely effect:
  - Higher values emphasize long-term outcomes.
  - Lower values bias short-term gains.

### `dqn_params.epsilon_start` (float)
- Purpose: initial exploration probability.

### `dqn_params.epsilon_min` (float)
- Purpose: lower bound for exploration.

### `dqn_params.epsilon_decay` (float)
- Purpose: multiplicative episode-level epsilon decay.
- Likely effect:
  - Slower decay (closer to 1): more exploration longer.
  - Faster decay: quicker exploitation, higher risk of premature convergence.

### `dqn_params.buffer_capacity` (int)
- Purpose: replay memory size.
- Likely effect:
  - Larger buffers improve diversity but can include stale transitions.

### `dqn_params.target_update_freq` (int)
- Purpose: update interval for target network in train steps.
- Likely effect:
  - Frequent updates: faster adaptation, possible instability.
  - Less frequent updates: more stable targets, slower adaptation.

## 7) Practical Tuning Patterns

### If learning is unstable
- Try:
  - lower `learning_rate`
  - increase `target_update_freq`
  - switch `algo` from `vanilla` to `double`

### If learning is too slow
- Try:
  - moderate increase in `learning_rate`
  - `dueling` or `rainbow_lite`
  - adjust `epsilon_decay` to maintain exploration longer

### If overtake behavior is poor
- Check:
  - `track.overtakingZones` difficulty placement
  - `algo = rainbow_lite` with tuned `n_step` and PER settings
  - overtake metrics in evaluator outputs, not just win rate

## 8) Reproducibility Checklist

- Version-control every config used for published results.
- Record:
  - algorithm name/options
  - train seeds and eval seeds
  - train/eval runs and laps
  - commit hash
- Keep control arm (`vanilla`) in every benchmark batch.
- Interpret overlap in confidence intervals conservatively.
