import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from helpers.simulatorHelpers import (
    build_mini_loops,
    build_speed_profile,
    plot_speed_profile,
    get_speed_at_distance,
)

from helpers.simulatorVisualisers import run_visualisation
from feedback import create_driver_feedback
from agents.DQN import DQNAgent

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
        # Restore all driver fields from initial state
        for driver, init_state in zip(self.race_state.drivers, self._initial_driver_states):
            driver.__dict__.clear()
            driver.__dict__.update(init_state)
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
                        success = self._attempt_overtake(driver, driver_ahead, zone)
                        # Use the last action if available
                        action = getattr(driver, "last_action", None)
                        # Calculate reward for DQN agents
                        if hasattr(driver.agent, 'store_transition') and action is not None:
                            reward = self._calculate_reward(driver, success, action)
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
    
    def _calculate_reward(self, driver, overtake_success, action):
        """Calculate reward for an action taken by a DQN agent.
        
        Reward structure:
        - Successful overtake: +10
        - Failed overtake: -5
        - Position improvement: +5
        - Position loss: -5
        - Small time penalty per step: -0.01 (encourages faster racing)
        """
        reward = 0.0
        
        # Overtake reward
        if overtake_success:
            reward += 10.0
        else:
            reward -= 5.0
        
        # Time penalty (encourage efficiency)
        reward -= 0.01
        
        # Future: Add position-based rewards, tyre management, etc.
        
        return reward
    
    def _attempt_overtake(self, overtaking_driver, target_driver, zone: Dict) -> bool:
        """Attempt an overtake with probability based on zone difficulty."""
        difficulty = zone.get("difficulty")
        success_probability = 1.0 - difficulty
        
        # Apply mulitiplier for success probability as a function of speed difference and gap to car ahead
        # TODO: Favor overtakes when speed difference is significant (reflects tyre choices and car performance)

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
                print(f"  Larger gap resulted in probability decreasing from {success_probability / max(0.0, 1.0 - (gap - 0.04) * 5):.2f} to {success_probability:.2f}")
        
        
        # Print probability of overtaking each time 
        if self.config.get("debugMode", False):
            print(f"  Overtake probability for {overtaking_driver.name} attempting to overtake {target_driver.name} at {zone.get('name')}: {success_probability:.2f}")
        
        success = np.random.random() < success_probability
        
        if success:
            print(f"  OVERTAKE: {overtaking_driver.name} overtook {target_driver.name} "
                  f"at {zone.get('name')}!")
            
            # Swap track positions slightly
            old_distance = overtaking_driver.current_distance
            overtaking_driver.current_distance = target_driver.current_distance + 0.01
            target_driver.current_distance = old_distance
            
            # Update positions
            self.race_state.update_driver_positions()
            return True
        else:
            # Apply distance penalty for failed attempt (lose 50m),
            # clamp/wrap so we don't create negative/ambiguous positions.
            penalty_km = 0.05
            new_dist = overtaking_driver.current_distance - penalty_km
            if new_dist < 0:
                # wrap forward to maintain consistent lap semantics
                new_dist = max(0.0, new_dist + self.track_distance)
            overtaking_driver.current_distance = new_dist

            # Update positions
            self.race_state.update_driver_positions()

            print(f"  FAILED: {overtaking_driver.name} failed to overtake "
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
                self.race_results[run_number][driver.name] = {
                    "laps": driver.completed_laps,
                    "finish_time": finish_time,
                    "gap": gap,
                    "position": i,
                }

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

        if not self.race_results:
            print("No race results to visualise.")
            return

        # Prepare data
        runs = sorted(self.race_results.keys())
        driver_names = set()
        for run in runs:
            driver_names.update(self.race_results[run].keys())
        driver_names = sorted(driver_names)

        # 1. x-axis: run number, y-axis: finishing position for each driver
        fig, axes = plt.subplots(3, 1, figsize=(12, 16))
        for driver in driver_names:
            positions = [self.race_results[run].get(driver, {}).get("position", np.nan) for run in runs]
            axes[0].plot(runs, positions, marker='o', label=driver)
        axes[0].set_title("Finishing Position per Run")
        axes[0].set_xlabel("Run Number")
        axes[0].set_ylabel("Finishing Position (1=Winner)")
        axes[0].invert_yaxis()  # So 1st is at the top
        axes[0].legend()
        axes[0].grid(True)

        # 2. Histogram of average driver finishes over all runs
        avg_positions = []
        for driver in driver_names:
            pos = [self.race_results[run].get(driver, {}).get("position", np.nan) for run in runs]
            pos = [p for p in pos if not np.isnan(p)]
            avg = np.mean(pos) if pos else np.nan
            avg_positions.append(avg)
        colors1 = plt.cm.tab10(np.linspace(0, 1, len(driver_names)))
        axes[1].bar(driver_names, avg_positions, color=colors1)
        axes[1].set_title("Average Finishing Position per Driver")
        axes[1].set_ylabel("Average Position (Lower is Better)")
        axes[1].invert_yaxis()
        axes[1].grid(axis='y')

        # 3. Most positions gained per driver over all runs
        # For each run, positions gained = starting position - finishing position
        # Assume starting position is the order in driver_names (or you can store grid positions if available)
        most_gained = []
        for driver in driver_names:
            gains = []
            for run in runs:
                finish_pos = self.race_results[run].get(driver).get("position")
                start_pos = driver_names.index(driver) + 1  # 1-based grid
                if not np.isnan(finish_pos):
                    gains.append(start_pos - finish_pos)
            most_gained.append(max(gains) if gains else 0)
        colors2 = plt.cm.tab20(np.linspace(0, 1, len(driver_names)))
        axes[2].bar(driver_names, most_gained, color=colors2)
        axes[2].set_title("Most Positions Gained in a Single Run")
        axes[2].set_ylabel("Max Positions Gained")
        axes[2].grid(axis='y')

        plt.tight_layout()
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

