import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from copy import deepcopy

from helpers.simulatorHelpers import (
    build_mini_loops,
    build_speed_profile,
    plot_speed_profile,
    get_speed_at_distance,
)

from helpers.simulatorVisualisers import run_visualisation
from agents.DQN import DQNAgent
from base_agents import RiskLevel
from runtime_profiles import resolve_complexity_profile

REWARD_COMPONENTS = [
    "outcome",
    "persistent_position",
    "tactical",
    "pace",
    "tyre_pit",
    "penalty",
]

DEFAULT_REWARD_CONTRACT: Dict[str, Any] = {
    "schema_version": "v1_finish_first",
    "weights": {
        "outcome": 2.0,
        "persistent_position": 0.1,
        "tactical": 0.05,
        "pace": 0.05,
        "tyre_pit": 0.05,
        "penalty": 2.0,
    },
    "normalization": {
        "low": {
            "outcome": 1.0,
            "persistent_position": 1.0,
            "tactical": 2.0,
            "pace": 1.0,
            "tyre_pit": 1.0,
            "penalty": 1.0,
        },
        "medium": {
            "outcome": 1.0,
            "persistent_position": 1.0,
            "tactical": 2.0,
            "pace": 1.0,
            "tyre_pit": 1.0,
            "penalty": 1.0,
        },
        "high": {
            "outcome": 1.0,
            "persistent_position": 1.0,
            "tactical": 2.0,
            "pace": 1.0,
            "tyre_pit": 1.0,
            "penalty": 1.0,
        },
    },
    "component_activation_by_complexity": {
        "low": {
            "outcome": True,
            "persistent_position": True,
            "tactical": True,
            "pace": False,
            "tyre_pit": False,
            "penalty": True,
        },
        "medium": {
            "outcome": True,
            "persistent_position": True,
            "tactical": True,
            "pace": True,
            "tyre_pit": False,
            "penalty": True,
        },
        "high": {
            "outcome": True,
            "persistent_position": True,
            "tactical": True,
            "pace": True,
            "tyre_pit": True,
            "penalty": True,
        },
    },
    "tactical": {
        "success": 1.0,
        "failure_by_risk": {
            "CONSERVATIVE": -0.5,
            "NORMAL": -1.0,
            "AGGRESSIVE": -1.5,
        },
    },
}

DEFAULT_STOCHASTICITY_CONTRACT: Dict[str, Any] = {
    "active_level": "s0",
    "levels": {
        "s0": {
            "base_probability_scale": 1.0,
            "risk_modifier_scale": 1.0,
            "gap_modifier_scale": 1.0,
            "probability_noise_std": 0.0,
            "min_success_probability": 0.02,
            "max_success_probability": 0.95,
        },
        "s1": {
            "base_probability_scale": 1.0,
            "risk_modifier_scale": 1.0,
            "gap_modifier_scale": 1.0,
            "probability_noise_std": 0.02,
            "min_success_probability": 0.02,
            "max_success_probability": 0.95,
        },
        "s2": {
            "base_probability_scale": 1.0,
            "risk_modifier_scale": 1.0,
            "gap_modifier_scale": 1.0,
            "probability_noise_std": 0.05,
            "min_success_probability": 0.02,
            "max_success_probability": 0.95,
        },
    },
}

DEFAULT_PROTOCOL_CONTRACT: Dict[str, Any] = {
    "enable_curriculum": False,
    "stage_order": ["low", "medium", "high"],
    "stochasticity_order": ["s0", "s1", "s2"],
    "seed_sets": {
        "smoke": [101, 202, 303],
        "candidate": [101, 202, 303, 404, 505],
        "benchmark": [101, 202, 303, 404, 505],
    },
    "train_runs": {"low": 200, "medium": 200, "high": 200},
    "eval_runs": {"low": 200, "medium": 200, "high": 200},
    "comparison_matrix": {
        "algorithms": ["vanilla", "double", "dueling", "rainbow_lite"],
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged

@dataclass
class LapRecord:
    """Record of a completed lap."""
    driver_name: str
    lap_number: int
    lap_time: float
    position: int
    gap_to_leader: float


class RaceSimulator:

    def race_reset(self):
        """Reset all race state for a new run (for batch mode), restoring all driver fields to their initial values."""
        self.race_finished = False
        self.winner = None
        self.lap_records = []
        self.driver_overtake_cooldowns = {}
        self.driver_finish_times = {}
        self.race_state.current_tick = 0
        self.race_state.elapsed_time = 0.0
        # On first call, store initial driver state for full reset
        if not hasattr(self, '_initial_driver_states'):
            self._initial_driver_states = []
            for driver in self.race_state.drivers:
                # Store a shallow copy of all __dict__ fields
                self._initial_driver_states.append(driver.__dict__.copy())
        # Preserve career stats - these must survive the initial-state restore
        career_stats = {
            d.name: {
                "cumulative_positions_gained": getattr(d, "cumulative_positions_gained", 0.0),
                "runs_completed": getattr(d, "runs_completed", 0),
            }
            for d in self.race_state.drivers
        }
        # Restore all driver fields from initial state
        for driver, init_state in zip(self.race_state.drivers, self._initial_driver_states):
            driver.__dict__.clear()
            driver.__dict__.update(init_state)
            # Shallow copy shares list references - give each run fresh histories
            driver.position_history = []
            driver.gap_to_ahead_history = []
            driver.gap_to_behind_history = []
            driver.lap_progress_history = []
            driver.pending_overtake_decisions = {}
            driver.resolved_overtake_decision_keys = set()
            driver.decision_events = []
            driver.terminal_bonus_awarded = False
            self._reset_position_change_counters(driver)
            self._reset_reward_tracking(driver)
            # Re-apply career stats that were accumulated before this reset
            stats = career_stats[driver.name]
            driver.cumulative_positions_gained = stats["cumulative_positions_gained"]
            driver.runs_completed = stats["runs_completed"]
        # Re-assign starting grid positions with balanced coverage for DQN drivers
        import random
        num_drivers = len(self.race_state.drivers)
        grid_gap_meters = 8

        if not hasattr(self, "_start_pos_cycle_idx"):
            self._start_pos_cycle_idx = int(
                self.config.get("marl", {}).get("eval_start_pos_offset", 0)
            )

        all_positions = list(range(1, num_drivers + 1))
        used_positions = set()

        dqn_drivers = [d for d in self.race_state.drivers if isinstance(d.agent, DQNAgent)]

        # Assign DQN drivers positions in a round-robin cycle to ensure even coverage
        for dqn_driver in dqn_drivers:
            pos = all_positions[self._start_pos_cycle_idx % num_drivers]
            self._start_pos_cycle_idx += 1
            while pos in used_positions:
                pos = all_positions[self._start_pos_cycle_idx % num_drivers]
                self._start_pos_cycle_idx += 1
            dqn_driver.starting_position = pos
            dqn_driver.position = pos
            dqn_driver.current_distance = ((num_drivers - pos) * grid_gap_meters) / 1000.0
            used_positions.add(pos)

        # Randomize remaining positions for non-DQN drivers
        remaining_positions = [p for p in all_positions if p not in used_positions]
        if self.active_complexity == "medium":
            random.shuffle(remaining_positions)
        for driver in self.race_state.drivers:
            if driver in dqn_drivers:
                continue
            pos = remaining_positions.pop(0)
            driver.starting_position = pos
            driver.position = pos
            driver.current_distance = ((num_drivers - pos) * grid_gap_meters) / 1000.0
        for driver in self.race_state.drivers:
            driver._last_recorded_position = int(driver.position)
        # Optionally reset other race_state fields if needed
        if hasattr(self.race_state, 'overtaking_zones') and isinstance(self.race_state.overtaking_zones, list):
            for zone in self.race_state.overtaking_zones:
                if hasattr(zone, 'reset') and callable(zone.reset):
                    zone.reset()
    """Main race simulator with live visualization."""
    
    def __init__(self, race_state, config: Dict, track: Dict):
        self.race_state = race_state
        self.config = config
        self.track = track
        
        # Race settings
        race_settings = config.get("race_settings", {})
        self.total_laps = race_settings.get("total_laps")
        
        # Track settings
        track_config = config.get("track", {})
        self.track_distance = track_config.get("distance")  # km

        # Runtime complexity profile
        self.active_complexity, self.active_complexity_profile, self.available_complexity_profiles = (
            resolve_complexity_profile(config)
        )
        self.reward_contract = self._resolve_reward_contract(config)
        self.active_reward_activation = self._resolve_reward_activation(self.active_complexity)
        self.active_reward_normalization = self._resolve_reward_normalization(self.active_complexity)
        self.stochasticity_contract = self._resolve_stochasticity_contract(config)
        self.active_stochasticity_level, self.active_stochasticity_profile = self._resolve_stochasticity_level(
            self.stochasticity_contract
        )
        self.protocol_contract = self._resolve_protocol_contract(config)
        
        # Build mini-loop lookup for speed calculation
        self.mini_loops = build_mini_loops(track_config)
        self.speed_profile = build_speed_profile(self.mini_loops, self.track_distance)
        
        # Simulation settings
        sim_config = config.get("simulator", {})
        telemetry_cfg = sim_config.setdefault("telemetry", {})
        telemetry_cfg.setdefault("include_legacy_tick_histories", False)
        telemetry_cfg.setdefault("log_decision_events", True)
        self.tick_duration = sim_config.get("tick_duration", 0.01)  # seconds
        self.tick_rate = sim_config.get("tick_rate", 100)  # Hz
        self.telemetry_include_legacy_tick_histories = bool(
            telemetry_cfg.get("include_legacy_tick_histories", False)
        )
        self.telemetry_log_decision_events = bool(telemetry_cfg.get("log_decision_events", True))
        
        # Overtake cooldown to prevent rapid re-attempts (in ticks)
        self.overtake_cooldown_ticks = int(5.0 / self.tick_duration)  # 5 second cooldown
        self.driver_overtake_cooldowns: Dict[str, int] = {}  # driver_name -> tick when can attempt again
        
        # Metrics storage
        self.lap_records: List[LapRecord] = []
        self.race_finished = False
        self.winner = None
        self.driver_finish_times: Dict[str, float] = {}  # driver_name -> time when they finished
        
        # Reward sharing coefficient (Phase 4 incentive-regime sweep).
        # Only active when complexity_profile == "low_marl", exactly 2 DQN agents are present,
        # and alpha > 0.  At alpha=0.0 the terminal bonus is numerically identical to Phase 3.
        self.reward_sharing_alpha = float(
            self.config.get("marl", {}).get("reward_sharing_alpha", 0.0)
        )

        # Track coordinates for visualization
        self.track_coords = track.get("coordinates") if isinstance(track, dict) else None
        
        # Animation settings
        self.animation_interval = 50  # ms between frames (20 FPS for smooth viz)
        self.ticks_per_frame = max(1, int(self.animation_interval / 1000 / self.tick_duration))
        
        # Colors for drivers: prefer explicit driver `colour`/`color` attribute when present
        # otherwise fall back to a default colormap.
        default_colors = plt.cm.tab10(np.linspace(0, 1, len(race_state.drivers)))
        self.driver_colors = []
        for i, driver in enumerate(race_state.drivers):
            drv_color = getattr(driver, "colour", None) or getattr(driver, "color", None)
            if drv_color:
                try:
                    # validate/convert color (accepts CSS names, hex, rgb tuples, etc.)
                    rgba = mcolors.to_rgba(drv_color)
                    # keep original string where possible (matplotlib accepts either)
                    # but use rgba tuple to guarantee validity
                    self.driver_colors.append(rgba)
                except Exception:
                    # invalid colour string -> fallback to colormap entry
                    if self.config.get("debugMode", False):
                        print(f"Invalid colour '{drv_color}' for {driver.name}; using default colormap")
                    self.driver_colors.append(tuple(default_colors[i]))
            else:
                self.driver_colors.append(tuple(default_colors[i]))

        # plot speed profile for debugging
        if self.config.get("debugMode"):
            plot_speed_profile(self.speed_profile, self.mini_loops, show_loops=True)

        # set agents to training mode if config is set so
        for driver in self.race_state.drivers:
            self._reset_position_change_counters(driver)
            self._reset_reward_tracking(driver)
            if isinstance(driver.agent, DQNAgent) and self.config.get('simulator').get('agent_mode') == "training":
                driver.agent.set_training_mode(True)
            if isinstance(driver.agent, DQNAgent) and self.config.get('simulator').get('agent_mode') == "evaluation":
                driver.agent.set_training_mode(False)
        
        # create dicts to store different visualisations
        self.agent_learning_visualisations = defaultdict(dict)
        self.race_results = defaultdict(dict)
        self._initialise_run_logging()

    @staticmethod
    def _resolve_complexity_bucket_name(raw_name: str) -> str:
        name = str(raw_name or "low").strip().lower()
        if name not in {"low", "medium", "high"}:
            return "low"
        return name

    def _resolve_reward_contract(self, config: Dict[str, Any]) -> Dict[str, Any]:
        reward_cfg = config.get("reward", {})
        if reward_cfg is not None and not isinstance(reward_cfg, dict):
            raise TypeError("config.reward must be an object")
        contract = _deep_merge(DEFAULT_REWARD_CONTRACT, reward_cfg if isinstance(reward_cfg, dict) else {})

        weights = contract.get("weights", {})
        normalization = contract.get("normalization", {})
        activation = contract.get("component_activation_by_complexity", {})
        if not isinstance(weights, dict):
            raise TypeError("config.reward.weights must be an object")
        if not isinstance(normalization, dict):
            raise TypeError("config.reward.normalization must be an object")
        if not isinstance(activation, dict):
            raise TypeError("config.reward.component_activation_by_complexity must be an object")

        for component in REWARD_COMPONENTS:
            if component not in weights:
                raise ValueError(f"Missing config.reward.weights.{component}")
            weights[component] = float(weights.get(component, 0.0))

        for complexity_name in ("low", "medium", "high"):
            norms = normalization.get(complexity_name)
            if not isinstance(norms, dict):
                raise ValueError(f"Missing config.reward.normalization.{complexity_name}")
            actives = activation.get(complexity_name)
            if not isinstance(actives, dict):
                raise ValueError(f"Missing config.reward.component_activation_by_complexity.{complexity_name}")
            for component in REWARD_COMPONENTS:
                if component not in norms:
                    raise ValueError(
                        f"Missing config.reward.normalization.{complexity_name}.{component}"
                    )
                norm_scale = float(norms.get(component, 1.0))
                if norm_scale <= 0.0:
                    raise ValueError(
                        f"Normalization scale must be > 0 for "
                        f"config.reward.normalization.{complexity_name}.{component}"
                    )
                norms[component] = norm_scale
                actives[component] = bool(actives.get(component, False))

        tactical_cfg = contract.get("tactical", {})
        if not isinstance(tactical_cfg, dict):
            raise TypeError("config.reward.tactical must be an object")
        failure_map = tactical_cfg.get("failure_by_risk", {})
        if not isinstance(failure_map, dict):
            raise TypeError("config.reward.tactical.failure_by_risk must be an object")
        tactical_cfg["success"] = float(tactical_cfg.get("success", 1.0))
        for key in ("CONSERVATIVE", "NORMAL", "AGGRESSIVE"):
            failure_map[key] = float(failure_map.get(key, -1.0))

        return contract

    def _resolve_reward_activation(self, complexity_name: str) -> Dict[str, bool]:
        bucket = self._resolve_complexity_bucket_name(complexity_name)
        by_complexity = self.reward_contract.get("component_activation_by_complexity", {})
        active = dict(by_complexity.get(bucket, by_complexity.get("low", {})))
        return {component: bool(active.get(component, False)) for component in REWARD_COMPONENTS}

    def _resolve_reward_normalization(self, complexity_name: str) -> Dict[str, float]:
        bucket = self._resolve_complexity_bucket_name(complexity_name)
        normalization = self.reward_contract.get("normalization", {})
        scales = dict(normalization.get(bucket, normalization.get("low", {})))
        return {component: float(scales.get(component, 1.0)) for component in REWARD_COMPONENTS}

    def _resolve_stochasticity_contract(self, config: Dict[str, Any]) -> Dict[str, Any]:
        stochastic_cfg = config.get("stochasticity", {})
        if stochastic_cfg is not None and not isinstance(stochastic_cfg, dict):
            raise TypeError("config.stochasticity must be an object")
        contract = _deep_merge(
            DEFAULT_STOCHASTICITY_CONTRACT,
            stochastic_cfg if isinstance(stochastic_cfg, dict) else {},
        )
        levels = contract.get("levels", {})
        if not isinstance(levels, dict) or not levels:
            raise ValueError("config.stochasticity.levels must define at least one level")
        for level_name, level_data in levels.items():
            if not isinstance(level_data, dict):
                raise TypeError(f"config.stochasticity.levels.{level_name} must be an object")
            for key in (
                "base_probability_scale",
                "risk_modifier_scale",
                "gap_modifier_scale",
                "probability_noise_std",
                "min_success_probability",
                "max_success_probability",
            ):
                if key not in level_data:
                    raise ValueError(f"Missing config.stochasticity.levels.{level_name}.{key}")
                level_data[key] = float(level_data.get(key, 0.0))
            min_p = float(level_data["min_success_probability"])
            max_p = float(level_data["max_success_probability"])
            if min_p < 0.0 or max_p > 1.0 or min_p >= max_p:
                raise ValueError(
                    f"Invalid probability bounds for config.stochasticity.levels.{level_name}: "
                    f"min={min_p}, max={max_p}"
                )
        return contract

    def _resolve_stochasticity_level(self, stochasticity_contract: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        levels = stochasticity_contract.get("levels", {})
        active = str(stochasticity_contract.get("active_level", "s0")).strip()
        if active not in levels:
            active = "s0" if "s0" in levels else next(iter(levels.keys()))
        profile = levels.get(active, {})
        return active, {k: float(v) for k, v in profile.items()}

    def _resolve_protocol_contract(self, config: Dict[str, Any]) -> Dict[str, Any]:
        protocol_cfg = config.get("protocol", {})
        if protocol_cfg is not None and not isinstance(protocol_cfg, dict):
            raise TypeError("config.protocol must be an object")
        contract = _deep_merge(
            DEFAULT_PROTOCOL_CONTRACT,
            protocol_cfg if isinstance(protocol_cfg, dict) else {},
        )
        for key in ("stage_order", "stochasticity_order"):
            values = contract.get(key, [])
            if not isinstance(values, list) or not values:
                raise ValueError(f"config.protocol.{key} must be a non-empty array")

        seed_sets = contract.get("seed_sets", {})
        if not isinstance(seed_sets, dict):
            raise TypeError("config.protocol.seed_sets must be an object")
        for seed_set_name, seeds in seed_sets.items():
            if not isinstance(seeds, list) or not seeds:
                raise ValueError(f"config.protocol.seed_sets.{seed_set_name} must be a non-empty array")
            seed_sets[seed_set_name] = [int(s) for s in seeds]

        for key in ("train_runs", "eval_runs"):
            runs_cfg = contract.get(key, {})
            if not isinstance(runs_cfg, dict):
                raise TypeError(f"config.protocol.{key} must be an object")
            for complexity_name in ("low", "medium", "high"):
                runs_cfg[complexity_name] = int(runs_cfg.get(complexity_name, 0))
                if runs_cfg[complexity_name] <= 0:
                    raise ValueError(
                        f"config.protocol.{key}.{complexity_name} must be a positive integer"
                    )

        matrix_cfg = contract.get("comparison_matrix", {})
        if not isinstance(matrix_cfg, dict):
            raise TypeError("config.protocol.comparison_matrix must be an object")
        algos = matrix_cfg.get("algorithms", [])
        if not isinstance(algos, list) or not algos:
            raise ValueError("config.protocol.comparison_matrix.algorithms must be a non-empty array")

        return contract

    def _initialise_run_logging(self):
        """Set up disk-backed logging for race results and visualisation replay."""
        sim_config = self.config.get("simulator", {})
        run_count = int(sim_config.get("runs", 0) or 0)
        visualise_from = str(sim_config.get("visualise_from_run_name", "")).strip()
        logs_root = Path("logs")
        logs_root.mkdir(exist_ok=True)

        # Replay-only mode: do not create a new logs folder when runs == 0.
        if run_count == 0:
            self.run_name = None
            self.run_dir = None
            self.race_results_log_path = None
            self.is_replay_only = True

            if visualise_from:
                candidate = logs_root / visualise_from / "race_results.jsonl"
                if candidate.exists():
                    self.visualisation_log_path = candidate
                else:
                    print(f"Requested visualise_from_run_name '{visualise_from}' not found.")
                    self.visualisation_log_path = None
            else:
                self.visualisation_log_path = None
            return

        self.is_replay_only = False
        configured_name = sim_config.get("run_name", "")

        run_name = str(configured_name).strip() if configured_name is not None else ""
        if not run_name:
            track_name = self.config.get("track", {}).get("name", "track")
            laps = self.config.get("race_settings", {}).get("total_laps", 0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{track_name}_{run_count}runs_{laps}laps_{timestamp}"

        safe_run_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_name)
        if not safe_run_name:
            safe_run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        run_dir = logs_root / safe_run_name
        if run_dir.exists():
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = logs_root / f"{safe_run_name}_{suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_dir.name
        self.run_dir = run_dir
        self.race_results_log_path = run_dir / "race_results.jsonl"
        self.race_results_log_path.write_text("", encoding="utf-8")

        visualise_log_path = None
        if visualise_from:
            candidate = logs_root / visualise_from / "race_results.jsonl"
            if candidate.exists():
                visualise_log_path = candidate
            else:
                print(f"Requested visualise_from_run_name '{visualise_from}' not found; using current run logs.")
        self.visualisation_log_path = visualise_log_path or self.race_results_log_path

        metadata = {
            "run_name": self.run_name,
            "created_at": datetime.now().isoformat(),
            "simulator": {
                "runs": sim_config.get("runs"),
                "method": sim_config.get("method"),
                "agent_mode": sim_config.get("agent_mode"),
                "tick_rate": sim_config.get("tick_rate"),
                "tick_duration": sim_config.get("tick_duration"),
                "telemetry": sim_config.get("telemetry", {}),
            },
            "complexity": {
                "active_profile": self.active_complexity,
                "resolved_profile": self.active_complexity_profile,
            },
            "reward": {
                "schema_version": self.reward_contract.get("schema_version"),
                "active_components": self.active_reward_activation,
                "normalization": self.active_reward_normalization,
                "weights": self.reward_contract.get("weights", {}),
            },
            "feedback": self.config.get("feedback", {}),
            "stochasticity": {
                "active_level": self.active_stochasticity_level,
                "profile": self.active_stochasticity_profile,
            },
            "protocol": self.protocol_contract,
            "race_settings": self.config.get("race_settings", {}),
            "track": {
                "name": self.config.get("track", {}).get("name"),
                "distance": self.config.get("track", {}).get("distance"),
            },
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _append_result_record(self, record: Dict):
        """Append one per-driver, per-run race record to JSONL logs."""
        if not self.race_results_log_path:
            return
        with self.race_results_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _load_race_results_from_logs(self) -> Dict[int, Dict[str, Dict]]:
        """Load race results from disk logs into the legacy dict shape for plotting."""
        loaded = defaultdict(dict)
        if not hasattr(self, "visualisation_log_path") or not self.visualisation_log_path.exists():
            return loaded

        with self.visualisation_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                run_number = int(record.get("run_number"))
                driver_name = record.get("driver_name")
                data = record.get("data", {})
                loaded[run_number][driver_name] = data
        return loaded


    def _update_driver(self, driver, dt: float):
        """Update a single driver's position for one tick."""
        # Skip updating if driver has already finished
        if driver.completed_laps >= self.total_laps:
            driver.speed = 0.0
            return
        
        # Get current speed based on track position
        speed = get_speed_at_distance(driver.current_distance, 
                                      self.speed_profile, 
                                      self.mini_loops, 
                                      self.track_distance)

        # Move driver forward
        distance_delta = speed * dt
        driver.current_distance += distance_delta
        driver.current_lap_time += dt
        driver.total_race_time += dt
        driver.speed = speed * 3600  # Convert to km/h for display
        
        # Check for lap completion
        if driver.current_distance >= self.track_distance:
            driver.current_distance -= self.track_distance
            driver.completed_laps += 1
            
            # Record lap
            gap_to_leader = self._calculate_gap_to_leader(driver)
            lap_record = LapRecord(
                driver_name=driver.name,
                lap_number=driver.completed_laps,
                lap_time=driver.current_lap_time,
                position=driver.position,
                gap_to_leader=gap_to_leader
            )
            self.lap_records.append(lap_record)

            print(f"  Lap {driver.completed_laps}/{self.total_laps}: "
                  f"{driver.name} - {driver.current_lap_time:.3f}s (P{driver.position})")
            
            driver.current_lap_time = 0.0
            
            # Record finish time when driver completes their laps
            if driver.completed_laps >= self.total_laps and driver.name not in self.driver_finish_times:
                self.driver_finish_times[driver.name] = driver.total_race_time
                if self.winner is None:
                    self.winner = driver
                    print(f"  {driver.name} FINISHED! (1st to cross the line)")
                else:
                    print(f"  {driver.name} FINISHED!")
                self._award_terminal_bonus_transition(driver)
    
    def _all_drivers_finished(self) -> bool:
        """Check if all drivers have completed the required laps."""
        return all(driver.completed_laps >= self.total_laps for driver in self.race_state.drivers)
    
    def _calculate_gap_to_leader(self, driver) -> float:
        """Calculate time gap to race leader."""
        leader = min(self.race_state.drivers, key=lambda d: d.position)
        if driver == leader:
            return 0.0
        return driver.total_race_time - leader.total_race_time

    def _reset_position_change_counters(self, driver) -> None:
        """Reset lightweight per-race position change counters for a driver."""
        driver.total_absolute_position_changes = 0
        driver.total_in_race_position_gains = 0
        driver.total_in_race_position_losses = 0
        driver._last_recorded_position = int(getattr(driver, "position", 0))

    def _reset_reward_tracking(self, driver) -> None:
        """Reset per-race reward accounting for one driver."""
        driver.reward_total = 0.0
        driver.reward_component_totals = {component: 0.0 for component in REWARD_COMPONENTS}
        driver.reward_component_totals_raw = {component: 0.0 for component in REWARD_COMPONENTS}

    def _blank_reward_components(self) -> Dict[str, float]:
        return {component: 0.0 for component in REWARD_COMPONENTS}

    def _compose_reward_from_raw_components(self, raw_components: Dict[str, float]) -> Dict[str, Any]:
        """Normalize + weight reward components and return a full reward breakdown."""
        raw = self._blank_reward_components()
        normalized = self._blank_reward_components()
        weighted = self._blank_reward_components()

        total = 0.0
        for component in REWARD_COMPONENTS:
            active = bool(self.active_reward_activation.get(component, False))
            raw_value = float(raw_components.get(component, 0.0)) if active else 0.0
            norm_scale = float(self.active_reward_normalization.get(component, 1.0))
            norm_scale = norm_scale if norm_scale > 0.0 else 1.0
            normalized_value = raw_value / norm_scale
            weight = float(self.reward_contract.get("weights", {}).get(component, 0.0))
            weighted_value = normalized_value * weight

            raw[component] = raw_value
            normalized[component] = normalized_value
            weighted[component] = weighted_value
            total += weighted_value

        component_sum = float(sum(weighted.values()))
        return {
            "total": float(total),
            "raw": raw,
            "normalized": normalized,
            "weighted": weighted,
            "active_components": [
                component for component in REWARD_COMPONENTS
                if bool(self.active_reward_activation.get(component, False))
            ],
            "inactive_components": [
                component for component in REWARD_COMPONENTS
                if not bool(self.active_reward_activation.get(component, False))
            ],
            "component_sum_error": float(total - component_sum),
        }

    @staticmethod
    def _merge_reward_breakdowns(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two reward breakdown objects into one additive breakdown."""
        merged_raw = {
            component: float(first.get("raw", {}).get(component, 0.0))
            + float(second.get("raw", {}).get(component, 0.0))
            for component in REWARD_COMPONENTS
        }
        merged_normalized = {
            component: float(first.get("normalized", {}).get(component, 0.0))
            + float(second.get("normalized", {}).get(component, 0.0))
            for component in REWARD_COMPONENTS
        }
        merged_weighted = {
            component: float(first.get("weighted", {}).get(component, 0.0))
            + float(second.get("weighted", {}).get(component, 0.0))
            for component in REWARD_COMPONENTS
        }
        total = float(first.get("total", 0.0)) + float(second.get("total", 0.0))
        component_sum = float(sum(merged_weighted.values()))
        active_components = list(dict.fromkeys(
            list(first.get("active_components", [])) + list(second.get("active_components", []))
        ))
        inactive_components = list(dict.fromkeys(
            list(first.get("inactive_components", [])) + list(second.get("inactive_components", []))
        ))
        return {
            "total": total,
            "raw": merged_raw,
            "normalized": merged_normalized,
            "weighted": merged_weighted,
            "active_components": active_components,
            "inactive_components": inactive_components,
            "component_sum_error": float(total - component_sum),
        }

    def _accumulate_driver_reward_breakdown(self, driver, breakdown: Dict[str, Any]) -> None:
        """Accumulate one reward breakdown into per-race driver reward totals."""
        driver.reward_total = float(getattr(driver, "reward_total", 0.0)) + float(breakdown.get("total", 0.0))
        totals = getattr(driver, "reward_component_totals", None)
        raw_totals = getattr(driver, "reward_component_totals_raw", None)
        if not isinstance(totals, dict):
            totals = {component: 0.0 for component in REWARD_COMPONENTS}
            driver.reward_component_totals = totals
        if not isinstance(raw_totals, dict):
            raw_totals = {component: 0.0 for component in REWARD_COMPONENTS}
            driver.reward_component_totals_raw = raw_totals

        for component in REWARD_COMPONENTS:
            totals[component] = float(totals.get(component, 0.0)) + float(
                breakdown.get("weighted", {}).get(component, 0.0)
            )
            raw_totals[component] = float(raw_totals.get(component, 0.0)) + float(
                breakdown.get("raw", {}).get(component, 0.0)
            )

    def _record_position_change_metrics(self) -> None:
        """Track every in-race position movement without requiring full tick histories."""
        for driver in self.race_state.drivers:
            current_position = int(driver.position)
            previous_position = getattr(driver, "_last_recorded_position", None)
            if previous_position is None:
                driver._last_recorded_position = current_position
                continue

            delta = int(previous_position) - current_position
            if delta != 0:
                driver.total_absolute_position_changes += abs(delta)
                if delta > 0:
                    driver.total_in_race_position_gains += delta
                else:
                    driver.total_in_race_position_losses += -delta

            driver._last_recorded_position = current_position
    
    def _check_overtakes(self):
        """Resolve one explicit decision per driver per zone-pass."""
        drivers = self.race_state.drivers
        zones = self.race_state.overtaking_zones
        current_tick = self.race_state.current_tick
        
        for driver in drivers:
            if driver.agent is None:
                continue
            
            # Check cooldown
            cooldown_until = self.driver_overtake_cooldowns.get(driver.name, 0)
            if current_tick < cooldown_until:
                continue
            
            # Find driver immediately ahead on track (by distance, accounting for laps)
            driver_ahead, gap_distance = self._find_driver_ahead_on_track(driver)
            if driver_ahead is None:
                continue
            
            # Must be close enough for a meaningful overtake decision.
            if gap_distance > 0.1:  # Must be within 100m to consider overtake
                continue
            
            for zone_idx, zone in enumerate(zones, start=1):
                zone_dist = float(zone.get("distance_from_start", 0.0))
                zone_id = str(zone.get("id") or f"zone_{zone_idx}")
                zone_name = str(zone.get("name") or zone_id)
                zone_difficulty = float(zone.get("difficulty", 0.5))

                approach_distance = zone_dist - driver.current_distance
                if approach_distance < 0:
                    approach_distance += self.track_distance

                pass_lap = self._zone_pass_lap(driver, zone_dist)
                decision_key = f"{zone_id}|lap:{pass_lap}"
                in_approach_window = 0.0 < approach_distance <= 0.2
                at_zone = abs(driver.current_distance - zone_dist) < 0.05

                # Create one decision per zone pass while in the approach window.
                if (
                    in_approach_window
                    and decision_key not in driver.pending_overtake_decisions
                    and decision_key not in driver.resolved_overtake_decision_keys
                ):
                    driver.gap_to_ahead = gap_distance  # expose gap so policy functions can read it
                    action = driver.agent.get_action(driver, self.race_state, upcoming_zone=zone)
                    dqn_context = None
                    if isinstance(driver.agent, DQNAgent):
                        dqn_context = driver.agent.get_last_decision_context()
                    driver.pending_overtake_decisions[decision_key] = {
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "zone_difficulty": zone_difficulty,
                        "decision_tick": current_tick,
                        "decision_lap": driver.completed_laps,
                        "gap_to_ahead_km": float(gap_distance),
                        "action": action,
                        "action_label": self._action_label(action),
                        "attempt": bool(action.attempt_overtake),
                        "risk_level": action.risk_level.name,
                        "dqn_context": dqn_context,
                    }

                # Resolve when the driver reaches the zone.
                if at_zone and decision_key in driver.pending_overtake_decisions:
                    decision = driver.pending_overtake_decisions.pop(decision_key)
                    driver.resolved_overtake_decision_keys.add(decision_key)
                    action = decision["action"]
                    pos_before = driver.position
                    success = False
                    success_probability = 0.0
                    if decision["attempt"]:
                        success, success_probability = self._attempt_overtake(
                            overtaking_driver=driver,
                            target_driver=driver_ahead,
                            zone=zone,
                            risk_level=action.risk_level,
                            gap_km=decision["gap_to_ahead_km"],
                        )
                        self.driver_overtake_cooldowns[driver.name] = (
                            current_tick + self.overtake_cooldown_ticks
                        )

                    done = driver.completed_laps >= self.total_laps
                    reward_breakdown = self._calculate_decision_reward(
                        action=action,
                        overtake_success=success,
                        pos_before=pos_before,
                        pos_after=driver.position,
                    )
                    reward = float(reward_breakdown.get("total", 0.0))
                    if done and not getattr(driver, "terminal_bonus_awarded", False):
                        terminal_breakdown = self._calculate_terminal_bonus(driver, finished=True)
                        reward_breakdown = self._merge_reward_breakdowns(
                            reward_breakdown,
                            terminal_breakdown,
                        )
                        reward = float(reward_breakdown.get("total", 0.0))
                        driver.terminal_bonus_awarded = True

                    self._accumulate_driver_reward_breakdown(driver, reward_breakdown)

                    next_zone = self._find_next_zone_for_driver(driver)
                    if isinstance(driver.agent, DQNAgent):
                        driver.agent.store_transition_from_context(
                            context=decision.get("dqn_context"),
                            reward=reward,
                            next_driver=driver,
                            next_race_state=self.race_state,
                            done=done,
                            next_zone=next_zone,
                        )

                    self._record_decision_event(
                        driver=driver,
                        decision=decision,
                        zone_id=zone_id,
                        zone_name=zone_name,
                        zone_difficulty=zone_difficulty,
                        gap_to_ahead_km=float(decision["gap_to_ahead_km"]),
                        success=success,
                        reward=float(reward),
                        reward_breakdown=reward_breakdown,
                        pos_before=pos_before,
                        pos_after=driver.position,
                        success_probability=float(success_probability),
                    )
                    break
    
    def _find_driver_ahead_on_track(self, driver) -> Tuple[Optional[object], Optional[float]]:
        """Find the closest driver ahead on track and the corresponding gap in km."""
        drivers = self.race_state.drivers
        
        candidates = []
        for other in drivers:
            if other == driver:
                continue
            
            # Calculate effective distance (laps * track_distance + current_distance)
            driver_total = driver.completed_laps * self.track_distance + driver.current_distance
            other_total = other.completed_laps * self.track_distance + other.current_distance
            
            # Other must be ahead
            if other_total > driver_total:
                gap = other_total - driver_total
                candidates.append((other, gap))
        
        if not candidates:
            return None, None
        
        candidates.sort(key=lambda x: x[1])
        nearest = candidates[0]
        return nearest[0], float(nearest[1])
    
    def _calculate_track_gap(self, driver, other):
        """Calculate the actual track gap in km between two drivers."""
        driver_total = driver.completed_laps * self.track_distance + driver.current_distance
        other_total = other.completed_laps * self.track_distance + other.current_distance
        return abs(other_total - driver_total)
    
    def _find_next_zone_for_driver(self, driver):
        """Find the next overtaking zone ahead of the driver."""
        zones = self.race_state.overtaking_zones
        driver_dist = driver.current_distance
        
        # Find zones ahead
        ahead_zones = [z for z in zones if z.get("distance_from_start", 0) > driver_dist]
        
        if ahead_zones:
            # Return closest zone ahead
            return min(ahead_zones, key=lambda z: z.get("distance_from_start", 0))
        elif zones:
            # Wrap around to first zone (on next lap)
            return min(zones, key=lambda z: z.get("distance_from_start", 0))
        else:
            return None

    def _record_tick_metrics(self):
        """Record per-tick position and gap metrics for every driver.

        Drivers are sorted by current race position so we can derive the
        gap to the car immediately ahead and immediately behind using track
        distance converted to an approximate time gap at the driver's current
        speed.  The lap-progress value (completed_laps + fractional lap) is
        stored so that visualisations can use laps as the x-axis.
        """
        if not self.telemetry_include_legacy_tick_histories:
            return

        sorted_drivers = sorted(self.race_state.drivers, key=lambda d: d.position)

        for i, driver in enumerate(sorted_drivers):
            lap_progress = driver.completed_laps + (
                driver.current_distance / self.track_distance
                if self.track_distance > 0 else 0.0
            )
            driver.lap_progress_history.append(lap_progress)
            driver.position_history.append(driver.position)

            driver_total_dist = (
                driver.completed_laps * self.track_distance + driver.current_distance
            )
            speed_kms = driver.speed / 3600.0 if driver.speed > 0 else 1.0

            # Gap to car ahead
            if i > 0:
                ahead = sorted_drivers[i - 1]
                ahead_dist = ahead.completed_laps * self.track_distance + ahead.current_distance
                driver.gap_to_ahead_history.append((ahead_dist - driver_total_dist) / speed_kms)
            else:
                driver.gap_to_ahead_history.append(0.0)

            # Gap to car behind
            if i < len(sorted_drivers) - 1:
                behind = sorted_drivers[i + 1]
                behind_dist = behind.completed_laps * self.track_distance + behind.current_distance
                driver.gap_to_behind_history.append((driver_total_dist - behind_dist) / speed_kms)
            else:
                driver.gap_to_behind_history.append(0.0)

    def _zone_pass_lap(self, driver, zone_dist: float) -> int:
        """Lap number on which the driver will next pass this zone."""
        return int(driver.completed_laps if zone_dist >= driver.current_distance else driver.completed_laps + 1)

    def _action_label(self, action) -> str:
        if not action.attempt_overtake:
            return "HOLD"
        return f"ATTEMPT_{action.risk_level.name}"

    def _calculate_decision_reward(
        self,
        action,
        overtake_success: bool,
        pos_before: int,
        pos_after: int,
    ) -> Dict[str, Any]:
        """Compute config-driven reward breakdown for one zone decision."""
        raw_components = self._blank_reward_components()

        if action.attempt_overtake:
            if overtake_success:
                raw_components["tactical"] = float(
                    self.reward_contract.get("tactical", {}).get("success", 1.0)
                )
            else:
                risk_name = str(getattr(action.risk_level, "name", "NORMAL")).upper()
                failure_map = self.reward_contract.get("tactical", {}).get("failure_by_risk", {})
                raw_components["tactical"] = float(failure_map.get(risk_name, -1.0))

        raw_components["persistent_position"] = float(int(pos_before) - int(pos_after))
        return self._compose_reward_from_raw_components(raw_components)

    def _get_mixed_outcome_raw(self, driver) -> float:
        """Return the outcome raw delta for this driver, applying reward sharing when safe to do so.

        Guard conditions (all must be true):
          1. reward_sharing_alpha > 0.0
          2. active complexity profile is "low_marl" or "low_marl_vs_base"
          3. exactly two DQN agents are present in the current race

        When any condition fails the method returns the driver's own positional delta
        unchanged, preserving numerical identity with all Phase 2 and Phase 3 runs.
        The Base agent in low_marl_vs_base is never included in the sharing formula —
        the DQN count guard (== 2) ensures this automatically.
        """
        own_delta = float(int(driver.starting_position) - int(driver.position))
        alpha = self.reward_sharing_alpha
        if (
            alpha <= 0.0
            or self.active_complexity not in ("low_marl", "low_marl_vs_base")
            or sum(1 for d in self.race_state.drivers if isinstance(d.agent, DQNAgent)) != 2
        ):
            return own_delta
        other_dqn = [
            d for d in self.race_state.drivers
            if d is not driver and isinstance(d.agent, DQNAgent)
        ]
        if len(other_dqn) != 1:
            return own_delta
        teammate = other_dqn[0]
        teammate_delta = float(int(teammate.starting_position) - int(teammate.position))
        return (1.0 - alpha) * own_delta + alpha * teammate_delta

    def _calculate_terminal_bonus(self, driver, finished: bool = True) -> Dict[str, Any]:
        """Compute config-driven terminal reward breakdown."""
        raw_components = self._blank_reward_components()
        if finished:
            raw_components["outcome"] = self._get_mixed_outcome_raw(driver)
        else:
            raw_components["penalty"] = -1.0
        return self._compose_reward_from_raw_components(raw_components)

    def _award_terminal_bonus_transition(self, driver) -> None:
        if getattr(driver, "terminal_bonus_awarded", False):
            return

        terminal_breakdown = self._calculate_terminal_bonus(driver, finished=True)
        self._accumulate_driver_reward_breakdown(driver, terminal_breakdown)

        if not isinstance(driver.agent, DQNAgent):
            driver.terminal_bonus_awarded = True
            return

        next_zone = self._find_next_zone_for_driver(driver)
        driver.agent.store_transition(
            reward=float(terminal_breakdown.get("total", 0.0)),
            next_driver=driver,
            next_race_state=self.race_state,
            done=True,
            next_zone=next_zone,
        )
        driver.terminal_bonus_awarded = True

    def _record_decision_event(
        self,
        driver,
        decision: Dict,
        zone_id: str,
        zone_name: str,
        zone_difficulty: float,
        gap_to_ahead_km: float,
        success: bool,
        reward: float,
        reward_breakdown: Dict[str, Any],
        pos_before: int,
        pos_after: int,
        success_probability: float,
    ) -> None:
        event = {
            "tick": int(self.race_state.current_tick),
            "lap": int(driver.completed_laps),
            "zone_id": zone_id,
            "zone_name": zone_name,
            "zone_difficulty": float(zone_difficulty),
            "gap_to_ahead_km": float(gap_to_ahead_km),
            "action_label": str(decision["action_label"]),
            "risk_level": str(decision["risk_level"]),
            "attempt": bool(decision["attempt"]),
            "success": bool(success),
            "reward": float(reward),
            "reward_components": {
                component: float(reward_breakdown.get("weighted", {}).get(component, 0.0))
                for component in REWARD_COMPONENTS
            },
            "reward_components_raw": {
                component: float(reward_breakdown.get("raw", {}).get(component, 0.0))
                for component in REWARD_COMPONENTS
            },
            "reward_components_normalized": {
                component: float(reward_breakdown.get("normalized", {}).get(component, 0.0))
                for component in REWARD_COMPONENTS
            },
            "reward_component_sum_error": float(reward_breakdown.get("component_sum_error", 0.0)),
            "position_before": int(pos_before),
            "position_after": int(pos_after),
            "success_probability": float(success_probability),
            "decision_tick": int(decision["decision_tick"]),
            "decision_lap": int(decision["decision_lap"]),
        }
        driver.decision_events.append(event)

    def _build_decision_summary(self, events: List[Dict]) -> Dict:
        action_keys = ["HOLD", "ATTEMPT_CONSERVATIVE", "ATTEMPT_NORMAL", "ATTEMPT_AGGRESSIVE"]
        risk_keys = ["CONSERVATIVE", "NORMAL", "AGGRESSIVE"]
        reward_component_samples = {component: [] for component in REWARD_COMPONENTS}
        summary = {
            "total_decisions": len(events),
            "action_counts": {k: 0 for k in action_keys},
            "risk_attempt_counts": {k: 0 for k in risk_keys},
            "reward_component_sums": {component: 0.0 for component in REWARD_COMPONENTS},
            "reward_component_means": {component: 0.0 for component in REWARD_COMPONENTS},
            "reward_component_variance": {component: 0.0 for component in REWARD_COMPONENTS},
            "reward_component_sums_raw": {component: 0.0 for component in REWARD_COMPONENTS},
            "reward_component_means_raw": {component: 0.0 for component in REWARD_COMPONENTS},
            "reward_total_sum": 0.0,
            "reward_total_mean": 0.0,
            "reward_component_sum_error_max_abs": 0.0,
            "zone_stats": {},
        }
        for event in events:
            action_label = str(event.get("action_label", "HOLD"))
            risk_level = str(event.get("risk_level", "NORMAL"))
            zone_id = str(event.get("zone_id", "unknown"))
            zone_name = str(event.get("zone_name", zone_id))
            zone_difficulty = float(event.get("zone_difficulty", 0.5))
            attempt = bool(event.get("attempt", False))
            success = bool(event.get("success", False))
            reward = float(event.get("reward", 0.0))
            gap = float(event.get("gap_to_ahead_km", 0.0))
            success_probability = float(event.get("success_probability", 0.0))
            reward_components = event.get("reward_components", {})
            reward_components_raw = event.get("reward_components_raw", {})
            component_sum_error = float(event.get("reward_component_sum_error", 0.0))

            summary["action_counts"].setdefault(action_label, 0)
            summary["action_counts"][action_label] += 1
            if attempt:
                summary["risk_attempt_counts"].setdefault(risk_level, 0)
                summary["risk_attempt_counts"][risk_level] += 1
            summary["reward_total_sum"] += reward
            summary["reward_component_sum_error_max_abs"] = max(
                float(summary["reward_component_sum_error_max_abs"]),
                abs(component_sum_error),
            )
            for component in REWARD_COMPONENTS:
                weighted_val = float(reward_components.get(component, 0.0))
                raw_val = float(reward_components_raw.get(component, 0.0))
                summary["reward_component_sums"][component] += weighted_val
                summary["reward_component_sums_raw"][component] += raw_val
                reward_component_samples[component].append(weighted_val)

            zone_bucket = summary["zone_stats"].setdefault(
                zone_id,
                {
                    "zone_name": zone_name,
                    "zone_difficulty": zone_difficulty,
                    "decisions": 0,
                    "holds": 0,
                    "attempts": 0,
                    "successes": 0,
                    "action_counts": {k: 0 for k in action_keys},
                    "risk_attempt_counts": {k: 0 for k in risk_keys},
                    "_reward_sum": 0.0,
                    "_gap_sum": 0.0,
                    "_success_prob_sum": 0.0,
                    "_reward_component_sums": {component: 0.0 for component in REWARD_COMPONENTS},
                    "_reward_component_sums_raw": {component: 0.0 for component in REWARD_COMPONENTS},
                },
            )
            zone_bucket["decisions"] += 1
            zone_bucket["action_counts"].setdefault(action_label, 0)
            zone_bucket["action_counts"][action_label] += 1
            zone_bucket["_reward_sum"] += reward
            zone_bucket["_gap_sum"] += gap
            zone_bucket["_success_prob_sum"] += success_probability
            for component in REWARD_COMPONENTS:
                zone_bucket["_reward_component_sums"][component] += float(
                    reward_components.get(component, 0.0)
                )
                zone_bucket["_reward_component_sums_raw"][component] += float(
                    reward_components_raw.get(component, 0.0)
                )

            if attempt:
                zone_bucket["attempts"] += 1
                zone_bucket["risk_attempt_counts"].setdefault(risk_level, 0)
                zone_bucket["risk_attempt_counts"][risk_level] += 1
                if success:
                    zone_bucket["successes"] += 1
            else:
                zone_bucket["holds"] += 1

        for zone_stats in summary["zone_stats"].values():
            decisions = max(1, int(zone_stats["decisions"]))
            attempts = int(zone_stats["attempts"])
            zone_stats["attempt_rate"] = float(attempts / decisions)
            zone_stats["success_rate"] = float(zone_stats["successes"] / attempts) if attempts > 0 else 0.0
            zone_stats["avg_reward"] = float(zone_stats["_reward_sum"] / decisions)
            zone_stats["avg_gap_to_ahead_km"] = float(zone_stats["_gap_sum"] / decisions)
            zone_stats["avg_success_probability"] = float(zone_stats["_success_prob_sum"] / decisions)
            zone_stats["avg_reward_components"] = {
                component: float(zone_stats["_reward_component_sums"][component] / decisions)
                for component in REWARD_COMPONENTS
            }
            zone_stats["avg_reward_components_raw"] = {
                component: float(zone_stats["_reward_component_sums_raw"][component] / decisions)
                for component in REWARD_COMPONENTS
            }
            zone_stats.pop("_reward_sum", None)
            zone_stats.pop("_gap_sum", None)
            zone_stats.pop("_success_prob_sum", None)
            zone_stats.pop("_reward_component_sums", None)
            zone_stats.pop("_reward_component_sums_raw", None)

        num_events = max(1, len(events))
        summary["reward_total_mean"] = float(summary["reward_total_sum"] / num_events)
        for component in REWARD_COMPONENTS:
            summary["reward_component_means"][component] = float(
                summary["reward_component_sums"][component] / num_events
            )
            summary["reward_component_means_raw"][component] = float(
                summary["reward_component_sums_raw"][component] / num_events
            )
            samples = reward_component_samples.get(component, [])
            summary["reward_component_variance"][component] = float(np.var(samples)) if samples else 0.0

        return summary

    def _attempt_overtake(
        self,
        overtaking_driver,
        target_driver,
        zone: Dict,
        risk_level: RiskLevel,
        gap_km: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Attempt overtake using only zone difficulty, risk level, and gap."""
        difficulty = float(zone.get("difficulty", 0.5))
        gap = float(gap_km if gap_km is not None else self._calculate_track_gap(overtaking_driver, target_driver))

        profile = self.active_stochasticity_profile
        base_probability = (1.0 - difficulty) * float(profile.get("base_probability_scale", 1.0))
        risk_prob_modifiers = {
            RiskLevel.CONSERVATIVE: -0.08,
            RiskLevel.NORMAL: 0.0,
            RiskLevel.AGGRESSIVE: 0.12,
        }
        risk_modifier_scale = float(profile.get("risk_modifier_scale", 1.0))
        risk_adjustment = float(risk_prob_modifiers.get(risk_level, 0.0)) * risk_modifier_scale

        gap_norm = float(np.clip(gap / 0.1, 0.0, 1.0))
        gap_modifier_scale = float(profile.get("gap_modifier_scale", 1.0))
        gap_modifier = (1.0 - gap_norm - 0.5) * 0.3 * gap_modifier_scale
        probability_noise_std = float(profile.get("probability_noise_std", 0.0))
        probability_noise = float(np.random.normal(0.0, probability_noise_std)) if probability_noise_std > 0.0 else 0.0

        min_success_probability = float(profile.get("min_success_probability", 0.02))
        max_success_probability = float(profile.get("max_success_probability", 0.95))
        success_probability = float(
            np.clip(
                base_probability + risk_adjustment + gap_modifier + probability_noise,
                min_success_probability,
                max_success_probability,
            )
        )

        overtaking_driver.overtakes_attempted += 1
        success = bool(np.random.random() < success_probability)

        if self.config.get("debugMode", False):
            print(
                f"  Overtake probability for {overtaking_driver.name} ({risk_level.name}) "
                f"at {zone.get('name')}: {success_probability:.2f}"
            )

        if success:
            print(f"  OVERTAKE: {overtaking_driver.name} overtook {target_driver.name} at {zone.get('name')}!")
            overtaking_driver.overtakes_succeeded += 1

            old_distance = overtaking_driver.current_distance
            overtaking_driver.current_distance = target_driver.current_distance + 0.01
            target_driver.current_distance = old_distance
            self.race_state.update_driver_positions()
            self._record_position_change_metrics()
            return True, success_probability

        risk_penalty_km = {
            RiskLevel.CONSERVATIVE: 0.02,
            RiskLevel.NORMAL: 0.05,
            RiskLevel.AGGRESSIVE: 0.08,
        }
        penalty_km = float(risk_penalty_km.get(risk_level, 0.05))
        new_dist = overtaking_driver.current_distance - penalty_km
        if new_dist < 0:
            new_dist = max(0.0, new_dist + self.track_distance)
        overtaking_driver.current_distance = new_dist

        self.race_state.update_driver_positions()
        self._record_position_change_metrics()
        print(
            f"  FAILED: {overtaking_driver.name} ({risk_level.name}) failed to overtake "
            f"{target_driver.name} at {zone.get('name')}"
        )
        return False, success_probability

    def _simulation_step(self):
        """Execute one simulation step (multiple ticks for animation frame)."""
        if self.race_finished:
            return
            
        for _ in range(self.ticks_per_frame):
            if self.race_finished:
                break
                
            # Update all drivers
            for driver in self.race_state.drivers:
                self._update_driver(driver, self.tick_duration)
            
            # Update positions
            self.race_state.update_driver_positions()
            self._record_position_change_metrics()

            # Check overtakes
            self._check_overtakes()

            # Record per-tick metrics
            self._record_tick_metrics()

            # Update race state
            self.race_state.current_tick += 1
            self.race_state.elapsed_time += self.tick_duration
    
    def run_with_visualization(self):
        """Run the simulation with live matplotlib visualization."""
        
        run_visualisation(self)
    
    def run_batch(self):
        """Run the simulation without visualization (fast mode)."""
        print(f"\n{'='*60}")
        print(f"Starting Race (Batch Mode): {self.total_laps} laps")
        print(f"Active complexity profile: {self.active_complexity}")
        print(f"Active stochasticity level: {self.active_stochasticity_level}")
        if self.run_dir:
            print(f"Logging run data to: {self.run_dir}")
        elif self.is_replay_only:
            print("Replay mode: no new log folder will be created.")
        print(f"{'='*60}\n")
        runs = self.config.get('simulator', {}).get('runs')
        for run_idx in range(runs):
            print(f"\n=== Starting Simulation Run {run_idx + 1} of {runs} ===")
            self.race_reset()
            while not self._all_drivers_finished():
                for driver in self.race_state.drivers:
                    self._update_driver(driver, self.tick_duration)
                self.race_state.update_driver_positions()
                self._record_position_change_metrics()
                self._check_overtakes()
                self._record_tick_metrics()
                self.race_state.current_tick += 1
                self.race_state.elapsed_time += self.tick_duration
            self.race_finished = True
            
            # Train DQN agents after each episode
            if self.config.get('simulator').get('agent_mode') == 'training':
                self._train_dqn_agents(run_idx)
            
            print(f"\nRACE FINISHED! All drivers have completed {self.total_laps} laps.")
            self._print_results(run_idx)

        # Run saving agent model after each batch run 
        if self.config.get('simulator').get('agent_mode') == "training":
            self._save_dqn_agents()
        else:
            print("Training mode disabled, if you want to save your results adjust agent_mode to be 'training' ")
        
        # Visualise the learning of the agents, over each run
        if self.config.get("agent_review_mode"):
            self._visualise_agent_learning()

        self._visualise_results()

    def _train_dqn_agents(self, run_number, num_train_steps: int = 10, batch_size: int = 64):
        """Train all DQN agents using their replay buffers.
        
        Args:
            num_train_steps: Number of training steps per agent
            batch_size: Batch size for training
        """
        for driver in self.race_state.drivers:
            if isinstance(driver.agent, DQNAgent):
                losses = []
                for _ in range(num_train_steps):
                    loss = driver.agent.train_step(batch_size=batch_size)
                    if loss is not None:
                        losses.append(loss)
                
                # Decay epsilon once per episode (not per training step)
                driver.agent.on_episode_end()
                
                if losses:
                    avg_loss = sum(losses) / len(losses)
                    print(f"  {driver.agent.name} - Avg Loss: {avg_loss:.4f}, "
                          f"Epsilon: {driver.agent.epsilon:.4f}, "
                          f"Buffer Size: {len(driver.agent.replay_buffer)}")
                    self.agent_learning_visualisations[driver.agent][run_number] = {
                        "avg_loss": avg_loss,
                        "epsilon": driver.agent.epsilon,
                        "buffer_size": len(driver.agent.replay_buffer)
                    }
    
    def _print_results(self, run_number):
        """Print final race results and metrics."""
        print(f"\n{'='*60}")
        print("RACE RESULTS")
        print(f"{'='*60}")
        
        # Sort drivers by finish time (or total_race_time if they didn't finish)
        def get_sort_key(driver):
            if driver.name in self.driver_finish_times:
                return self.driver_finish_times[driver.name]
            else:
                return float('inf')  # unfinished drivers go to the end
        
        sorted_drivers = sorted(self.race_state.drivers, key=get_sort_key)
        
        print(f"\n{'Pos':<5} {'Driver':<25} {'Laps':<6} {'Total Time':<12} {'Gap':<10}")
        print("-" * 60)
        
        # Use first finisher's time as reference
        leader_time = next((self.driver_finish_times[d.name] for d in sorted_drivers if d.name in self.driver_finish_times), 0)
        
        for i, driver in enumerate(sorted_drivers, 1):
            finish_time = self.driver_finish_times.get(driver.name, driver.total_race_time)
            gap = finish_time - leader_time
            gap_str = f"+{gap:.3f}s" if gap > 0 else "Leader"
            print(f"P{i:<4} {driver.name:<25} {driver.completed_laps:<6} "
                  f"{finish_time:.3f}s   {gap_str}")
            
            if run_number is not None:
                # Update career stats before storing results
                driver.cumulative_positions_gained += driver.starting_position - i
                driver.runs_completed += 1

                decision_events = list(getattr(driver, "decision_events", []))
                decision_summary = self._build_decision_summary(decision_events)
                reward_component_totals = {
                    component: float(getattr(driver, "reward_component_totals", {}).get(component, 0.0))
                    for component in REWARD_COMPONENTS
                }
                reward_component_totals_raw = {
                    component: float(getattr(driver, "reward_component_totals_raw", {}).get(component, 0.0))
                    for component in REWARD_COMPONENTS
                }
                per_driver_result = {
                    "laps": driver.completed_laps,
                    "finish_time": finish_time,
                    "gap": gap,
                    "position": i,
                    "starting_position": driver.starting_position,
                    "complexity_profile": self.active_complexity,
                    "overtakes_attempted": getattr(driver, "overtakes_attempted", 0),
                    "overtakes_succeeded": getattr(driver, "overtakes_succeeded", 0),
                    "total_absolute_position_changes": int(
                        getattr(driver, "total_absolute_position_changes", 0)
                    ),
                    "total_in_race_position_gains": int(
                        getattr(driver, "total_in_race_position_gains", 0)
                    ),
                    "total_in_race_position_losses": int(
                        getattr(driver, "total_in_race_position_losses", 0)
                    ),
                    "reward_total": float(getattr(driver, "reward_total", 0.0)),
                    "reward_component_totals": reward_component_totals,
                    "reward_component_totals_raw": reward_component_totals_raw,
                    "reward_active_components": [
                        component
                        for component in REWARD_COMPONENTS
                        if bool(self.active_reward_activation.get(component, False))
                    ],
                    "reward_schema_version": str(self.reward_contract.get("schema_version", "unknown")),
                    "stochasticity_level": self.active_stochasticity_level,
                    "decision_summary": decision_summary,
                }
                if self.telemetry_log_decision_events:
                    per_driver_result["decision_events"] = decision_events
                if self.telemetry_include_legacy_tick_histories:
                    per_driver_result["position_history"] = list(driver.position_history)
                    per_driver_result["gap_to_ahead_history"] = list(driver.gap_to_ahead_history)
                    per_driver_result["gap_to_behind_history"] = list(driver.gap_to_behind_history)
                    per_driver_result["lap_progress_history"] = list(driver.lap_progress_history)
                self._append_result_record(
                    {
                        "run_number": int(run_number),
                        "driver_name": driver.name,
                        "data": per_driver_result,
                    }
                )

        # Fastest lap
        if self.lap_records:
            fastest = min(self.lap_records, key=lambda r: r.lap_time)
            print(f"\nFastest Lap: {fastest.driver_name} - "
                  f"{fastest.lap_time:.3f}s (Lap {fastest.lap_number})")
        
        print(f"\nTotal Race Time: {self.race_state.elapsed_time:.2f}s")
        print(f"Total Ticks: {self.race_state.current_tick:,}")

    def _save_dqn_agents(self):
        dqn_agents = []
        for driver in self.race_state.drivers:
            if isinstance(driver.agent, DQNAgent):
                # Save trained models
                dqn_agents.append(driver.agent)

        print("\nSaving trained models...")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        checkpoint_tag = str(self.config.get("simulator", {}).get("checkpoint_tag", "")).strip()
        safe_checkpoint_tag = "".join(ch for ch in checkpoint_tag if ch.isalnum() or ch in ("-", "_"))
        
        for agent in dqn_agents:
            model_path = models_dir / f"{agent.name}_trained.pth"
            agent.save(str(model_path))
            print(f"  Saved: {model_path}")
            if safe_checkpoint_tag:
                tagged_model_path = models_dir / f"{agent.name}_{safe_checkpoint_tag}.pth"
                agent.save(str(tagged_model_path))
                print(f"  Saved: {tagged_model_path}")

    def _visualise_agent_learning(self):
        """Visualise agent learning metrics (avg_loss, epsilon, buffer_size) per run for each agent."""
        import matplotlib.pyplot as plt

        if not self.agent_learning_visualisations:
            print("No agent learning data to visualise.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        metric_names = ["avg_loss", "epsilon", "buffer_size"]
        titles = ["Average Loss per Run", "Epsilon per Run", "Replay Buffer Size per Run"]

        for agent, run_data in self.agent_learning_visualisations.items():
            # Sort by run number for consistent plotting
            runs = sorted(run_data.keys())
            avg_losses = [run_data[r]["avg_loss"] for r in runs]
            epsilons = [run_data[r]["epsilon"] for r in runs]
            buffer_sizes = [run_data[r]["buffer_size"] for r in runs]
            label = getattr(agent, "name", str(agent))

            axes[0].plot(runs, avg_losses, marker='o', label=label)
            axes[1].plot(runs, epsilons, marker='o', label=label)
            axes[2].plot(runs, buffer_sizes, marker='o', label=label)

        for ax, title, ylabel in zip(axes, titles, metric_names):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Run Number")
        plt.tight_layout()
        plt.show()

    def _visualise_results(self):
        import matplotlib.pyplot as plt
        import numpy as np

        race_results = self._load_race_results_from_logs()
        if not race_results:
            print("No race results to visualise.")
            return

        # Shared data prep
        runs = sorted(race_results.keys())
        driver_names = set()
        for run in runs:
            driver_names.update(race_results[run].keys())
        driver_names = sorted(driver_names)
        run_ticks = list(runs)
        has_position_change_counters = all(
            isinstance(
                race_results[run].get(driver, {}).get("total_absolute_position_changes", None),
                (int, float),
            )
            and isinstance(
                race_results[run].get(driver, {}).get("total_in_race_position_gains", None),
                (int, float),
            )
            for run in runs
            for driver in driver_names
        )
        if not has_position_change_counters:
            print(
                "Position-change charts are using fallback reconstruction from legacy fields. "
                "Run a new simulation to get exact per-tick position change totals."
            )

        # Per-driver color and label lookups
        # Colors come from config (already resolved in __init__ as self.driver_colors)
        driver_color_map = {
            d.name: c
            for d, c in zip(self.race_state.drivers, self.driver_colors)
        }
        # Fallback for any driver name not in the live driver list
        fallback_colors = plt.cm.tab10(np.linspace(0, 1, max(len(driver_names), 1)))
        for i, name in enumerate(driver_names):
            if name not in driver_color_map:
                driver_color_map[name] = fallback_colors[i]

        # Agent type labels
        agent_type_map: Dict[str, str] = {}
        for driver in self.race_state.drivers:
            cls = type(driver.agent).__name__ if driver.agent else "None"
            if cls == "RandomAgent":
                label = "random"
            elif cls == "DQNAgent":
                label = "DQN"
            elif cls == "BaseAgent":
                label = "base"
            else:
                label = cls
            agent_type_map[driver.name] = label

        def driver_label(name: str) -> str:
            agent = agent_type_map.get(name, "?")
            return f"{name} ({agent})"

        def _extract_position_deltas(d_data: Dict) -> List[int]:
            """Return per-change position deltas (positive means position gained)."""
            pos_hist = d_data.get("position_history", [])
            if isinstance(pos_hist, list) and len(pos_hist) > 1:
                deltas = []
                for t in range(1, len(pos_hist)):
                    try:
                        deltas.append(int(pos_hist[t - 1]) - int(pos_hist[t]))
                    except Exception:
                        continue
                if deltas:
                    return deltas

            # Fallback for lightweight telemetry mode where per-tick history is absent.
            events = d_data.get("decision_events", [])
            if isinstance(events, list) and events:
                deltas = []
                for event in events:
                    before = event.get("position_before", None)
                    after = event.get("position_after", None)
                    if before is None or after is None:
                        continue
                    try:
                        deltas.append(int(before) - int(after))
                    except Exception:
                        continue
                return deltas

            # Last fallback for older logs: use net start->finish change as one delta.
            start_pos = d_data.get("starting_position", None)
            finish_pos = d_data.get("position", None)
            if start_pos is not None and finish_pos is not None:
                try:
                    return [int(start_pos) - int(finish_pos)]
                except Exception:
                    pass

            return []

        def _extract_position_totals(d_data: Dict) -> Tuple[int, int, int]:
            """Return (absolute_changes, gains, losses) for one driver in one run."""
            abs_changes = d_data.get("total_absolute_position_changes", None)
            gains = d_data.get("total_in_race_position_gains", None)
            losses = d_data.get("total_in_race_position_losses", None)

            if abs_changes is not None and gains is not None:
                try:
                    abs_changes_i = int(abs_changes)
                    gains_i = int(gains)
                    if losses is None:
                        losses_i = max(0, abs_changes_i - gains_i)
                    else:
                        losses_i = int(losses)
                    return abs_changes_i, gains_i, losses_i
                except Exception:
                    pass

            deltas = _extract_position_deltas(d_data)
            gains_i = int(sum(delta for delta in deltas if delta > 0))
            losses_i = int(sum(-delta for delta in deltas if delta < 0))
            abs_changes_i = int(sum(abs(delta) for delta in deltas))
            return abs_changes_i, gains_i, losses_i

        bar_labels = [driver_label(d) for d in driver_names]
        bar_colors = [driver_color_map[d] for d in driver_names]

        # WINDOW 1: race performance across runs
        fig1, axes1 = plt.subplots(2, 1, figsize=(14, 12))
        fig1.suptitle("Race Performance - All Runs", fontsize=14, y=1.0)

        # 1. Average finishing position per driver (bar chart)
        avg_positions = []
        for driver in driver_names:
            pos = [
                race_results[run].get(driver, {}).get("position", np.nan)
                for run in runs
            ]
            pos = [p for p in pos if not np.isnan(p)]
            avg_positions.append(np.mean(pos) if pos else np.nan)
        axes1[0].bar(bar_labels, avg_positions, color=bar_colors)
        axes1[0].set_title("Average Finishing Position per Driver")
        axes1[0].set_ylabel("Average Position (lower is better)")
        axes1[0].invert_yaxis()
        axes1[0].grid(axis='y')

        # 2. Cumulative positions gained per run, per driver (line chart)
        for driver in driver_names:
            cumulative = 0
            cumulative_series = []
            for run in runs:
                d_data = race_results[run].get(driver, {})
                start_pos = d_data.get("starting_position", np.nan)
                finish_pos = d_data.get("position", np.nan)
                if not (np.isnan(start_pos) or np.isnan(finish_pos)):
                    cumulative += start_pos - finish_pos
                cumulative_series.append(cumulative)
            axes1[1].plot(
                runs,
                cumulative_series,
                marker='o',
                label=driver_label(driver),
                color=driver_color_map[driver],
            )
        axes1[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes1[1].set_title("Cumulative Positions Gained per Run")
        axes1[1].set_xlabel("Run Number")
        axes1[1].set_ylabel("Cumulative Positions Gained")
        axes1[1].set_xticks(range(0, max(run_ticks) + 100, 100))
        axes1[1].legend()
        axes1[1].grid(True)

        fig1.tight_layout()

        # WINDOW 2: position change analysis
        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 12))
        fig2.suptitle("Position Change Analysis - All Runs", fontsize=14, y=1.0)

        # 3. Total absolute position change across all runs per driver (bar chart)
        # Uses persisted per-run counters when present; falls back to reconstructed deltas.
        abs_changes = []
        for driver in driver_names:
            total = 0
            for run in runs:
                d_data = race_results[run].get(driver, {})
                abs_total, _, _ = _extract_position_totals(d_data)
                total += abs_total
            abs_changes.append(total)
        
        # Keep full bar visibility (including low/zero values) by anchoring at 0.
        y_max_abs = max(abs_changes) if abs_changes else 0
        y_pad_abs = max(1.0, y_max_abs * 0.05)
        
        bars_abs = axes2[0].bar(bar_labels, abs_changes, color=bar_colors)
        axes2[0].set_title("Total Absolute Position Changes Across All Runs")
        axes2[0].set_ylabel("Total |dPosition| across all ticks and runs")
        axes2[0].set_ylim(0, y_max_abs + y_pad_abs)
        axes2[0].grid(axis='y')
        label_offset_abs = max(0.2, y_pad_abs * 0.1)
        for bar, value in zip(bars_abs, abs_changes):
            axes2[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offset_abs,
                f"{value:.0f}",
                ha='center',
                va='bottom',
                fontsize=9,
            )

        # 4. Total in-race position gains across all runs per driver (bar chart)
        # Uses persisted per-run counters when present; falls back to reconstructed deltas.
        total_increments = []
        for driver in driver_names:
            increments = 0
            for run in runs:
                d_data = race_results[run].get(driver, {})
                _, gains_total, _ = _extract_position_totals(d_data)
                increments += gains_total
            total_increments.append(increments)
        
        # Keep full bar visibility (including low/zero values) by anchoring at 0.
        y_max_inc = max(total_increments) if total_increments else 0
        y_pad_inc = max(1.0, y_max_inc * 0.05)
        
        bars_inc = axes2[1].bar(bar_labels, total_increments, color=bar_colors)
        axes2[1].set_title("Total In-Race Position Gains Across All Runs")
        axes2[1].set_ylabel("Total Positions Gained (in-race, all runs)")
        axes2[1].set_ylim(0, y_max_inc + y_pad_inc)
        axes2[1].grid(axis='y')
        label_offset_inc = max(0.2, y_pad_inc * 0.1)
        for bar, value in zip(bars_inc, total_increments):
            axes2[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offset_inc,
                f"{value:.0f}",
                ha='center',
                va='bottom',
                fontsize=9,
            )

        fig2.tight_layout()

        # WINDOW 3+: average finishing position by starting position
        # One subplot per driver in a 3x3 grid; paginated if > 9 drivers.
        _GRID_ROWS = 3
        _GRID_COLS = 3
        _PER_PAGE = _GRID_ROWS * _GRID_COLS

        # Build per-driver mapping: starting_position -> [finishing_positions]
        driver_start_finish: Dict[str, Dict[int, list]] = {}
        for driver in driver_names:
            sf: Dict[int, list] = {}
            for run in runs:
                d_data = race_results[run].get(driver, {})
                sp = d_data.get("starting_position", None)
                fp = d_data.get("position", None)
                if sp is not None and fp is not None and not (np.isnan(sp) or np.isnan(fp)):
                    sf.setdefault(int(sp), []).append(fp)
            driver_start_finish[driver] = sf

        # Compute shared y-axis limits across all drivers for fair comparison
        all_avg_finishes = [
            np.mean(fps)
            for sf in driver_start_finish.values()
            for fps in sf.values()
        ]
        if all_avg_finishes:
            global_y_min = min(all_avg_finishes)
            global_y_max = max(all_avg_finishes)
        else:
            global_y_min, global_y_max = 1.0, float(len(driver_names))
        y_pad = 0.5
        shared_ylim = (global_y_min - y_pad, global_y_max + y_pad)

        page_num = 0
        for page_start in range(0, len(driver_names), _PER_PAGE):
            page_drivers = driver_names[page_start:page_start + _PER_PAGE]
            page_num += 1

            fig_sp, axes_sp = plt.subplots(_GRID_ROWS, _GRID_COLS, figsize=(14, 12))
            title_suffix = f" (page {page_num})" if len(driver_names) > _PER_PAGE else ""
            fig_sp.suptitle(
                f"Average Finishing Position by Starting Position{title_suffix}",
                fontsize=14, y=1.0
            )
            axes_sp_flat = axes_sp.flatten()

            for i, driver in enumerate(page_drivers):
                ax = axes_sp_flat[i]
                sf = driver_start_finish[driver]
                if sf:
                    start_positions = sorted(sf.keys())
                    avg_finishes = [np.mean(sf[sp]) for sp in start_positions]
                    ax.bar(
                        start_positions, avg_finishes,
                        color=driver_color_map[driver],
                        edgecolor='#333333', linewidth=0.5
                    )
                    ax.set_title(driver_label(driver), fontsize=10)
                    ax.set_xlabel("Starting Position")
                    ax.set_ylabel("Avg Finish Position")
                    ax.set_xticks(start_positions)
                    ax.set_ylim(shared_ylim)
                    ax.invert_yaxis()
                    ax.grid(axis='y')
                else:
                    ax.set_visible(False)

            # Hide any unused grid cells on the last page
            for j in range(len(page_drivers), _GRID_ROWS * _GRID_COLS):
                axes_sp_flat[j].set_visible(False)

            fig_sp.tight_layout()

        # WINDOW 4: per-tick telemetry for the last run
        last_run = runs[-1]
        fig3, axes3 = plt.subplots(3, 1, figsize=(14, 16))
        fig3.suptitle(f"In-Race Telemetry \u2014 Run {last_run}", fontsize=14, y=1.0)

        def _legend_or_no_data_note(ax, note: str) -> None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
                return
            ax.text(
                0.5,
                0.5,
                note,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#666666",
            )

        # 7. Position throughout the race (every tick)
        for driver in driver_names:
            d_data = race_results[last_run].get(driver, {})
            lap_prog = d_data.get("lap_progress_history", [])
            pos_hist = d_data.get("position_history", [])
            if lap_prog and pos_hist:
                axes3[0].plot(lap_prog, pos_hist, linewidth=0.8,
                              label=driver_label(driver), color=driver_color_map[driver])
        axes3[0].set_title("Race Position (every tick)")
        axes3[0].set_xlabel("Race Progress (laps)")
        axes3[0].set_ylabel("Race Position")
        axes3[0].invert_yaxis()
        _legend_or_no_data_note(
            axes3[0],
            "No per-tick position history in this run. "
            "Enable simulator.telemetry.include_legacy_tick_histories=true to plot this.",
        )
        axes3[0].grid(True)

        # 8. Gap to car ahead (every tick)
        for driver in driver_names:
            d_data = race_results[last_run].get(driver, {})
            lap_prog = d_data.get("lap_progress_history", [])
            gap_ahead = d_data.get("gap_to_ahead_history", [])
            if lap_prog and gap_ahead:
                axes3[1].plot(lap_prog, gap_ahead, linewidth=0.8,
                              label=driver_label(driver), color=driver_color_map[driver])
        axes3[1].set_title("Gap to Car Ahead (every tick)")
        axes3[1].set_xlabel("Race Progress (laps)")
        axes3[1].set_ylabel("Gap (seconds)")
        _legend_or_no_data_note(
            axes3[1],
            "No per-tick gap-to-ahead history in this run. "
            "Enable simulator.telemetry.include_legacy_tick_histories=true to plot this.",
        )
        axes3[1].grid(True)

        # 9. Gap to car behind (every tick)
        for driver in driver_names:
            d_data = race_results[last_run].get(driver, {})
            lap_prog = d_data.get("lap_progress_history", [])
            gap_behind = d_data.get("gap_to_behind_history", [])
            if lap_prog and gap_behind:
                axes3[2].plot(lap_prog, gap_behind, linewidth=0.8,
                              label=driver_label(driver), color=driver_color_map[driver])
        axes3[2].set_title("Gap to Car Behind (every tick)")
        axes3[2].set_xlabel("Race Progress (laps)")
        axes3[2].set_ylabel("Gap (seconds)")
        _legend_or_no_data_note(
            axes3[2],
            "No per-tick gap-to-behind history in this run. "
            "Enable simulator.telemetry.include_legacy_tick_histories=true to plot this.",
        )
        axes3[2].grid(True)

        fig3.tight_layout()

        # WINDOW 5: competitive analysis
        # Adaptive rolling window: 50 runs or 10% of total runs, whichever is smaller.
        roll_window = min(50, max(1, len(runs) // 10))

        fig5, axes5 = plt.subplots(1, 3, figsize=(20, 6))
        fig5.suptitle(
            f"Competitive Analysis  (rolling window = {roll_window} runs)",
            fontsize=14, y=1.02
        )

        # Helper: generic rolling mean that tolerates NaN values
        def _rolling_mean(values, window):
            out = []
            for i in range(len(values)):
                start = max(0, i - window + 1)
                valid = [v for v in values[start : i + 1] if not np.isnan(v)]
                out.append(np.mean(valid) if valid else np.nan)
            return out

        # 10. Rolling average finishing position
        ax_pos = axes5[0]
        for driver in driver_names:
            raw = [
                race_results[run].get(driver, {}).get("position", np.nan)
                for run in runs
            ]
            smoothed = _rolling_mean(raw, roll_window)
            ax_pos.plot(
                runs, smoothed,
                label=driver_label(driver),
                color=driver_color_map[driver],
                linewidth=2
            )
        ax_pos.invert_yaxis()
        ax_pos.set_title(f"Rolling {roll_window}-Run Avg Finishing Position")
        ax_pos.set_xlabel("Run Number")
        ax_pos.set_ylabel("Avg Position (lower = better)")
        ax_pos.legend(fontsize=8)
        ax_pos.grid(True)

        # 11. Head-to-head win rate matrix
        ax_hth = axes5[1]
        n_d = len(driver_names)
        win_matrix = np.full((n_d, n_d), np.nan)
        for i, d1 in enumerate(driver_names):
            for j, d2 in enumerate(driver_names):
                if i == j:
                    continue
                wins, total = 0, 0
                for run in runs:
                    p1 = race_results[run].get(d1, {}).get("position", None)
                    p2 = race_results[run].get(d2, {}).get("position", None)
                    if p1 is not None and p2 is not None:
                        total += 1
                        if p1 < p2:
                            wins += 1
                win_matrix[i, j] = (wins / total * 100) if total > 0 else np.nan

        # Mask diagonal (NaN -> white) for imshow
        masked = np.ma.masked_invalid(win_matrix)
        cmap_hth = plt.cm.RdYlGn.copy()
        cmap_hth.set_bad(color="lightgrey")
        im = ax_hth.imshow(masked, cmap=cmap_hth, vmin=0, vmax=100, aspect="auto")
        plt.colorbar(im, ax=ax_hth, label="Win %", shrink=0.8)
        tick_labels = [driver_label(d) for d in driver_names]
        ax_hth.set_xticks(range(n_d))
        ax_hth.set_yticks(range(n_d))
        ax_hth.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=8)
        ax_hth.set_yticklabels(tick_labels, fontsize=8)
        ax_hth.set_title("Head-to-Head Win Rate (%)\nRow beats column")
        for i in range(n_d):
            for j in range(n_d):
                val = win_matrix[i, j]
                if not np.isnan(val):
                    ax_hth.text(
                        j, i, f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=9,
                        color="black" if 20 < val < 80 else "white"
                    )

        # 12. Overtake success rate per agent (rolling average)
        ax_ot = axes5[2]
        for driver in driver_names:
            raw_rates = []
            for run in runs:
                d_data = race_results[run].get(driver, {})
                attempted = d_data.get("overtakes_attempted", 0)
                succeeded = d_data.get("overtakes_succeeded", 0)
                raw_rates.append(
                    (succeeded / attempted * 100) if attempted > 0 else np.nan
                )
            smoothed_ot = _rolling_mean(raw_rates, roll_window)
            ax_ot.plot(
                runs, smoothed_ot,
                label=driver_label(driver),
                color=driver_color_map[driver],
                linewidth=2
            )
        ax_ot.set_ylim(0, 100)
        ax_ot.set_title(f"Rolling {roll_window}-Run Overtake Success Rate (%)")
        ax_ot.set_xlabel("Run Number")
        ax_ot.set_ylabel("Success Rate (%)")
        ax_ot.legend(fontsize=8)
        ax_ot.grid(True)

        fig5.tight_layout()

        plt.show()

def init_simulator(race_state, config, track):
    """ 
    Create simulator logic, with the choice of either stepping through time and visualising,
    or running simulation(s) with no visualisation to completion.
    """
    simulator = RaceSimulator(race_state, config, track)
    
    method = config.get('simulator', {}).get('method', '').lower()
    
    if method == 'real-time':
        print("Starting real-time simulator with visualizations...")
        simulator.run_with_visualization()
    elif method == 'batch':
        print("Starting batch simulator with no visualizations...")
        # Run based on number of runs specified
        simulator.run_batch()
    else:
        # Default to real-time if method not specified
        print("Starting real-time simulator (default)...")
        simulator.run_with_visualization()
    
    return simulator

