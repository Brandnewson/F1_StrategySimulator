import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from helpers.simulatorHelpers import (
    build_mini_loops,
    build_speed_profile,
    plot_speed_profile,
    get_speed_at_distance,
)

from helpers.simulatorVisualisers import run_visualisation
from feedback import create_driver_feedback
from agents.DQN import DQNAgent
from base_agents import RiskLevel

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
        # Preserve career stats — these must survive the initial-state restore
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
            # Shallow copy shares list references — give each run fresh histories
            driver.position_history = []
            driver.gap_to_ahead_history = []
            driver.gap_to_behind_history = []
            driver.lap_progress_history = []
            # Re-apply career stats that were accumulated before this reset
            stats = career_stats[driver.name]
            driver.cumulative_positions_gained = stats["cumulative_positions_gained"]
            driver.runs_completed = stats["runs_completed"]
        # Re-assign starting grid positions with balanced coverage for DQN drivers
        import random
        num_drivers = len(self.race_state.drivers)
        grid_gap_meters = 8

        if not hasattr(self, "_start_pos_cycle_idx"):
            self._start_pos_cycle_idx = 0

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
        random.shuffle(remaining_positions)
        for driver in self.race_state.drivers:
            if driver in dqn_drivers:
                continue
            pos = remaining_positions.pop(0)
            driver.starting_position = pos
            driver.position = pos
            driver.current_distance = ((num_drivers - pos) * grid_gap_meters) / 1000.0
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
        
        # Build mini-loop lookup for speed calculation
        self.mini_loops = build_mini_loops(track_config)
        self.speed_profile = build_speed_profile(self.mini_loops, self.track_distance)
        
        # Simulation settings
        sim_config = config.get("simulator", {})
        self.tick_duration = sim_config.get("tick_duration", 0.01)  # seconds
        self.tick_rate = sim_config.get("tick_rate", 100)  # Hz
        
        # Overtake cooldown to prevent rapid re-attempts (in ticks)
        self.overtake_cooldown_ticks = int(5.0 / self.tick_duration)  # 5 second cooldown
        self.driver_overtake_cooldowns: Dict[str, int] = {}  # driver_name -> tick when can attempt again
        
        # Metrics storage
        self.lap_records: List[LapRecord] = []
        self.race_finished = False
        self.winner = None
        self.driver_finish_times: Dict[str, float] = {}  # driver_name -> time when they finished
        
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
            if isinstance(driver.agent, DQNAgent) and self.config.get('simulator').get('agent_mode') == "training":
                driver.agent.set_training_mode(True)
            if isinstance(driver.agent, DQNAgent) and self.config.get('simulator').get('agent_mode') == "evaluation":
                driver.agent.set_training_mode(False)
        
        # create dicts to store different visualisations
        self.agent_learning_visualisations = defaultdict(dict)
        self.race_results = defaultdict(dict)
        self._initialise_run_logging()

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
            },
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
            
            # --- Per-lap shaped reward for DQN agents ---
            if hasattr(driver, 'agent') and hasattr(driver.agent, 'store_transition'):
                lap_reward = self._calculate_lap_reward(driver)
                next_zone = self._find_next_zone_for_driver(driver)
                done = driver.completed_laps >= self.total_laps
                driver.agent.store_transition(lap_reward, driver, self.race_state, done, next_zone)

            driver.current_lap_time = 0.0
            
            # Record finish time when driver completes their laps
            if driver.completed_laps >= self.total_laps and driver.name not in self.driver_finish_times:
                self.driver_finish_times[driver.name] = driver.total_race_time
                if self.winner is None:
                    self.winner = driver
                    print(f"  {driver.name} FINISHED! (1st to cross the line)")
                else:
                    print(f"  {driver.name} FINISHED!")
    
    def _all_drivers_finished(self) -> bool:
        """Check if all drivers have completed the required laps."""
        return all(driver.completed_laps >= self.total_laps for driver in self.race_state.drivers)
    
    def _calculate_gap_to_leader(self, driver) -> float:
        """Calculate time gap to race leader."""
        leader = min(self.race_state.drivers, key=lambda d: d.position)
        if driver == leader:
            return 0.0
        return driver.total_race_time - leader.total_race_time
    
    def _check_overtakes(self):
        """Check for overtake opportunities and process agent decisions.
        
        Overtake logic:
        - Driver must be BEHIND another driver (by track position, not race position)
        - Driver considers overtake when APPROACHING an overtaking zone (within 200m before)
        - Overtake is resolved when driver ENTERS the zone
        - Cooldown prevents rapid re-attempts
        """
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
            driver_ahead = self._find_driver_ahead_on_track(driver)
            if driver_ahead is None:
                continue
            
            # Calculate actual gap on track
            gap_distance = self._calculate_track_gap(driver, driver_ahead)
            if gap_distance > 0.1:  # Must be within 100m to consider overtake
                continue
                
            # Check if approaching an overtaking zone (within 200m before the zone)
            for zone in zones:
                zone_dist = zone.get("distance_from_start", 0)
                
                # Driver should be approaching the zone (within 200m before)
                approach_distance = zone_dist - driver.current_distance
                # Handle wrap-around at track end
                if approach_distance < -self.track_distance / 2:
                    approach_distance += self.track_distance
                
                # Approaching zone: between 0 and 200m before
                if 0 < approach_distance <= 0.2:
                    # Get agent decision with feedback
                    action = driver.agent.get_action(driver, self.race_state, upcoming_zone=zone)
                    # Store the last action for use at the zone
                    driver.last_action = action
                    if action.attempt_overtake:
                        # Mark driver as attempting overtake
                        driver.attempting_overtake = True
                        break

                # At the zone: resolve overtake attempt
                elif abs(driver.current_distance - zone_dist) < 0.05:  # Within 50m of zone
                    if driver.attempting_overtake:
                        pos_before = driver.position  # capture position before attempt updates it
                        success = self._attempt_overtake(driver, driver_ahead, zone)
                        # Use the last action if available
                        action = getattr(driver, "last_action", None)
                        # Calculate reward for DQN agents
                        if hasattr(driver.agent, 'store_transition') and action is not None:
                            reward = self._calculate_reward(driver, success, action, zone=zone, pos_before=pos_before)
                            next_zone = self._find_next_zone_for_driver(driver)
                            done = driver.completed_laps >= self.total_laps
                            driver.agent.store_transition(reward, driver, self.race_state, done, next_zone)
                        driver.attempting_overtake = False
                        # Set cooldown regardless of success
                        self.driver_overtake_cooldowns[driver.name] = current_tick + self.overtake_cooldown_ticks
                        break
    
    def _find_driver_ahead_on_track(self, driver):
        """Find the driver immediately ahead on track (by distance, same lap or ahead)."""
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
            return None
        
        # Return the closest driver ahead
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
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

    def _calculate_reward(self, driver, overtake_success, action, zone=None, pos_before=None):
        """Calculate reward for an action taken by a DQN agent.

        Reward structure:
        - Success: base +10, up to +5 bonus for harder zones, up to +30% late-race multiplier
        - Failure: -2 (conservative), -5 (normal), -8 (aggressive)
        - Position gained from this overtake: +2 per place
        - Step penalty: -0.01
        - End-of-race: +3 per net position gained from starting grid
        """
        reward = 0.0

        zone_difficulty = zone.get("difficulty", 0.5) if zone else 0.5
        laps_done_frac = driver.completed_laps / max(self.total_laps, 1)

        if overtake_success:
            # Higher reward for succeeding at harder zones
            base = 10.0 + zone_difficulty * 5.0
            # Late-race overtakes are worth more (up to 30% bonus)
            late_race_mult = 1.0 + 0.3 * laps_done_frac
            reward += base * late_race_mult
            # Bonus for gaining a place from this specific overtake
            if pos_before is not None:
                reward += (pos_before - driver.position) * 2.0
        else:
            # Risk-adjusted failure: bigger gamble = heavier penalty
            risk_penalties = {"CONSERVATIVE": -2.0, "NORMAL": -5.0, "AGGRESSIVE": -8.0}
            reward += risk_penalties.get(action.risk_level.name, -5.0)

        # Step penalty (encourages efficiency)
        reward -= 0.01

        # End-of-race bonus: reward overall improvement from starting position
        if driver.completed_laps >= self.total_laps:
            reward += (driver.starting_position - driver.position) * 3.0

        return reward

    def _calculate_lap_reward(self, driver) -> float:
        """Calculate a per-lap shaped reward for DQN agents.

        Provides frequent, informative gradient signals between the rare
        overtake decision points. Rewards position changes DURING THIS LAP
        to avoid reward scale inflation over multiple laps.

        The magnitudes are deliberately smaller than overtake rewards so that
        the agent still prioritises overtake decisions but gets enough signal
        to learn positional awareness.
        """
        reward = 0.0
        num_drivers = len(self.race_state.drivers)

        # Reward position changes during THIS lap only (not cumulative from race start)
        pos_hist = driver.position_history
        if len(pos_hist) > 0:
            ticks_per_lap = self.race_state.base_lap_ticks if self.race_state.base_lap_ticks > 0 else 100
            start_of_lap_idx = max(0, len(pos_hist) - ticks_per_lap)
            pos_start = pos_hist[start_of_lap_idx]
            pos_end = driver.position
            
            # Reward position gained this lap only
            positions_gained_this_lap = pos_start - pos_end
            reward += positions_gained_this_lap * 0.5  # Small reward per position gained

        # Small reward for being in a good absolute position (encourages
        # staying at the front rather than being content with grid-relative gains)
        # Best position = 1 → bonus ~0.1; worst = 0
        reward += (num_drivers - driver.position) * 0.05

        # End-of-race bonus (final lap only)
        if driver.completed_laps >= self.total_laps:
            reward += (driver.starting_position - driver.position) * 3.0

        return reward

    def _attempt_overtake(self, overtaking_driver, target_driver, zone: Dict) -> bool:
        """Attempt an overtake with probability based on zone difficulty and risk level.

        Risk level modifies both the success probability and the consequences:
        - AGGRESSIVE: +0.15 success probability bonus, but 100m penalty on failure
        - NORMAL:     standard probability, 50m penalty on failure (unchanged)
        - CONSERVATIVE: -0.10 success probability penalty, but only 20m penalty on failure

        This gives the agent a meaningful risk/reward trade-off to learn from.
        """
        difficulty = zone.get("difficulty")
        success_probability = 1.0 - difficulty

        # Track attempt count on the driver
        overtaking_driver.overtakes_attempted += 1

        # --- Risk level modifies success probability ---
        action = getattr(overtaking_driver, "last_action", None)
        risk_level = action.risk_level if action is not None else RiskLevel.NORMAL

        risk_prob_modifiers = {
            RiskLevel.CONSERVATIVE: -0.10,
            RiskLevel.NORMAL: 0.0,
            RiskLevel.AGGRESSIVE: 0.15,
        }
        success_probability += risk_prob_modifiers.get(risk_level, 0.0)

        # --- Gap and speed modifiers (unchanged logic) ---
        speed_diff = overtaking_driver.speed - target_driver.speed  # in km/h
        gap = self._calculate_track_gap(overtaking_driver, target_driver)  # in km

        # Favor overtakes when gap is small and speed difference is positive
        if gap < 0.04:  # less than 40m
            success_probability *= 1.1
            if self.config.get("debugMode", False):
                print(f"  Small gap resulted in probability increasing from {success_probability/1.1:.2f} to {success_probability:.2f}")
        else:  # penalize larger gaps larger than 40m
            success_probability *= max(0.0, 1.0 - (gap - 0.04) * 5)  # reduce prob for larger gaps
            if self.config.get("debugMode", False):
                print(f"  Larger gap resulted in probability decreasing from {success_probability / max(0.01, 1.0 - (gap - 0.04) * 5):.2f} to {success_probability:.2f}")

        # Clamp probability to valid range
        success_probability = np.clip(success_probability, 0.0, 0.95)

        # Print probability of overtaking each time
        if self.config.get("debugMode", False):
            print(f"  Overtake probability for {overtaking_driver.name} ({risk_level.name}) "
                  f"attempting to overtake {target_driver.name} at {zone.get('name')}: {success_probability:.2f}")

        success = np.random.random() < success_probability

        if success:
            print(f"  OVERTAKE: {overtaking_driver.name} overtook {target_driver.name} "
                  f"at {zone.get('name')}!")

            # Track successful overtake
            overtaking_driver.overtakes_succeeded += 1

            # Swap track positions slightly
            old_distance = overtaking_driver.current_distance
            overtaking_driver.current_distance = target_driver.current_distance + 0.01
            target_driver.current_distance = old_distance

            # Update positions
            self.race_state.update_driver_positions()
            return True
        else:
            # Risk-adjusted distance penalty on failure:
            # AGGRESSIVE = 100m, NORMAL = 50m, CONSERVATIVE = 20m
            risk_penalty_km = {
                RiskLevel.CONSERVATIVE: 0.02,
                RiskLevel.NORMAL: 0.05,
                RiskLevel.AGGRESSIVE: 0.10,
            }
            penalty_km = risk_penalty_km.get(risk_level, 0.05)
            new_dist = overtaking_driver.current_distance - penalty_km
            if new_dist < 0:
                # wrap forward to maintain consistent lap semantics
                new_dist = max(0.0, new_dist + self.track_distance)
            overtaking_driver.current_distance = new_dist

            # Update positions
            self.race_state.update_driver_positions()

            print(f"  FAILED: {overtaking_driver.name} ({risk_level.name}) failed to overtake "
                  f"{target_driver.name} at {zone.get('name')}")
            return False
    
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

                per_driver_result = {
                    "laps": driver.completed_laps,
                    "finish_time": finish_time,
                    "gap": gap,
                    "position": i,
                    "starting_position": driver.starting_position,
                    "position_history": list(driver.position_history),
                    "gap_to_ahead_history": list(driver.gap_to_ahead_history),
                    "gap_to_behind_history": list(driver.gap_to_behind_history),
                    "lap_progress_history": list(driver.lap_progress_history),
                    "overtakes_attempted": getattr(driver, "overtakes_attempted", 0),
                    "overtakes_succeeded": getattr(driver, "overtakes_succeeded", 0),
                }
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
        
        for agent in dqn_agents:
            model_path = models_dir / f"{agent.name}_trained.pth"
            agent.save(str(model_path))
            print(f"  Saved: {model_path}")

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

        # ── Per-driver color and label lookups ───────────────────────────────
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

        bar_labels = [driver_label(d) for d in driver_names]
        bar_colors = [driver_color_map[d] for d in driver_names]

        # ── WINDOW 1: race performance across runs ───────────────────────────
        fig1, axes1 = plt.subplots(2, 1, figsize=(14, 12))
        fig1.suptitle("Race Performance — All Runs", fontsize=14, y=1.0)

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
                axes1[1].plot(runs, cumulative_series, marker='o',
                  label=driver_label(driver), color=driver_color_map[driver])
        axes1[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes1[1].set_title("Cumulative Positions Gained per Run")
        axes1[1].set_xlabel("Run Number")
        axes1[1].set_ylabel("Cumulative Positions Gained")
        axes1[1].set_xticks(range(0, max(run_ticks) + 100, 100))
        axes1[1].legend()
        axes1[1].grid(True)

        fig1.tight_layout()

        # ── WINDOW 2: position change analysis ──────────────────────────────
        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 12))
        fig2.suptitle("Position Change Analysis — All Runs", fontsize=14, y=1.0)

        # 3. Total absolute position change across all runs per driver (bar chart)
        # Walks per-tick position_history: counts every position change (gain or loss)
        abs_changes = []
        for driver in driver_names:
            total = 0
            for run in runs:
                pos_hist = race_results[run].get(driver, {}).get("position_history", [])
                for t in range(1, len(pos_hist)):
                    total += abs(pos_hist[t] - pos_hist[t - 1])
                abs_changes.append(total)
        
        # Focus on top 20% of results
        sorted_abs = sorted(abs_changes, reverse=True)
        top_20_pct_idx = max(0, len(sorted_abs) // 5)
        y_min_abs = sorted_abs[top_20_pct_idx] if top_20_pct_idx < len(sorted_abs) else min(abs_changes)
        y_max_abs = max(abs_changes)
        y_pad_abs = (y_max_abs - y_min_abs) * 0.1
        
        axes2[0].bar(bar_labels, abs_changes, color=bar_colors)
        axes2[0].set_title("Total Absolute Position Changes Across All Runs (per tick)")
        axes2[0].set_ylabel("Total |Δposition| across all ticks and runs")
        axes2[0].set_ylim(y_min_abs - y_pad_abs, y_max_abs + y_pad_abs)
        axes2[0].grid(axis='y')

        # 4. Total in-race position gains across all runs per driver (bar chart)
        total_increments = []
        for driver in driver_names:
            increments = 0
            for run in runs:
                pos_hist = race_results[run].get(driver, {}).get("position_history", [])
                for t in range(1, len(pos_hist)):
                    delta = pos_hist[t - 1] - pos_hist[t]
                    if delta > 0:
                        increments += delta
                total_increments.append(increments)
        
        # Focus on top 20% of results
        sorted_increments = sorted(total_increments, reverse=True)
        top_20_pct_idx = max(0, len(sorted_increments) // 5)
        y_min_inc = sorted_increments[top_20_pct_idx] if top_20_pct_idx < len(sorted_increments) else min(total_increments)
        y_max_inc = max(total_increments)
        y_pad_inc = (y_max_inc - y_min_inc) * 0.1 if (y_max_inc - y_min_inc) > 0 else 1
        
        axes2[1].bar(bar_labels, total_increments, color=bar_colors)
        axes2[1].set_title("Total In-Race Position Gains Across All Runs")
        axes2[1].set_ylabel("Total Positions Gained (in-race, all runs)")
        axes2[1].set_ylim(y_min_inc - y_pad_inc, y_max_inc + y_pad_inc)
        axes2[1].grid(axis='y')

        fig2.tight_layout()

        # ── WINDOW 3+: average finishing position by starting position ───────
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

        # ── WINDOW 4: per-tick telemetry for the last run ────────────────────
        last_run = runs[-1]
        fig3, axes3 = plt.subplots(3, 1, figsize=(14, 16))
        fig3.suptitle(f"In-Race Telemetry \u2014 Run {last_run}", fontsize=14, y=1.0)

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
        axes3[0].legend()
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
        axes3[1].legend()
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
        axes3[2].legend()
        axes3[2].grid(True)

        fig3.tight_layout()

        # ── WINDOW 5: competitive analysis ───────────────────────────────────
        # Adaptive rolling window: 50 runs or 10% of total runs, whichever is smaller.
        roll_window = min(50, max(1, len(runs) // 10))

        fig5, axes5 = plt.subplots(1, 3, figsize=(20, 6))
        fig5.suptitle(
            f"Competitive Analysis  (rolling window = {roll_window} runs)",
            fontsize=14, y=1.02
        )

        # ── helper: generic rolling mean that tolerates NaN values ──────────
        def _rolling_mean(values, window):
            out = []
            for i in range(len(values)):
                start = max(0, i - window + 1)
                valid = [v for v in values[start : i + 1] if not np.isnan(v)]
                out.append(np.mean(valid) if valid else np.nan)
            return out

        # ── 10. Rolling average finishing position ───────────────────────────
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

        # ── 11. Head-to-head win rate matrix ────────────────────────────────
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

        # Mask diagonal (NaN → white) for imshow
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

        # ── 12. Overtake success rate per agent (rolling average) ────────────
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

