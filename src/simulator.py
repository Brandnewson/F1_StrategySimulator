import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List
from dataclasses import dataclass

from helpers.simulatorHelpers import (
    build_mini_loops,
    build_speed_profile,
    plot_speed_profile,
    get_speed_at_distance,
)

from helpers.simulatorVisualisers import run_visualisation

@dataclass
class LapRecord:
    """Record of a completed lap."""
    driver_name: str
    lap_number: int
    lap_time: float
    position: int
    gap_to_leader: float


class RaceSimulator:
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
    
    
    
    def _update_driver(self, driver, dt: float):
        """Update a single driver's position for one tick."""
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
            
            # Check for race finish
            if driver.completed_laps >= self.total_laps and not self.race_finished:
                self.race_finished = True
                self.winner = driver
                print(f"\nRACE FINISHED! Winner: {driver.name}")
    
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
                    # Get agent decision
                    action = driver.agent.get_action(driver, self.race_state)
                    
                    if action.attempt_overtake:
                        # Mark driver as attempting overtake
                        driver.attempting_overtake = True
                        break
                
                # At the zone: resolve overtake attempt
                elif abs(driver.current_distance - zone_dist) < 0.05:  # Within 50m of zone
                    if driver.attempting_overtake:
                        success = self._attempt_overtake(driver, driver_ahead, zone)
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
        
        while not self.race_finished:
            # Run many ticks per step for batch mode
            for _ in range(1000):
                if self.race_finished:
                    break
                    
                for driver in self.race_state.drivers:
                    self._update_driver(driver, self.tick_duration)
                
                self.race_state.update_driver_positions()
                self._check_overtakes()
                self.race_state.current_tick += 1
                self.race_state.elapsed_time += self.tick_duration
        
        self._print_results()
    
    def _print_results(self):
        """Print final race results and metrics."""
        print(f"\n{'='*60}")
        print("RACE RESULTS")
        print(f"{'='*60}")
        
        # Sort drivers by position
        sorted_drivers = sorted(self.race_state.drivers, key=lambda d: d.position)
        
        print(f"\n{'Pos':<5} {'Driver':<25} {'Laps':<6} {'Total Time':<12} {'Gap':<10}")
        print("-" * 60)
        
        leader_time = sorted_drivers[0].total_race_time if sorted_drivers else 0
        
        for driver in sorted_drivers:
            gap = driver.total_race_time - leader_time
            gap_str = f"+{gap:.3f}s" if gap > 0 else "Leader"
            print(f"P{driver.position:<4} {driver.name:<25} {driver.completed_laps:<6} "
                  f"{driver.total_race_time:.3f}s   {gap_str}")
        
        # Fastest lap
        if self.lap_records:
            fastest = min(self.lap_records, key=lambda r: r.lap_time)
            print(f"\nFastest Lap: {fastest.driver_name} - "
                  f"{fastest.lap_time:.3f}s (Lap {fastest.lap_number})")
        
        print(f"\nTotal Race Time: {self.race_state.elapsed_time:.2f}s")
        print(f"Total Ticks: {self.race_state.current_tick:,}")


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
        simulator.run_batch()
    else:
        # Default to real-time if method not specified
        print("Starting real-time simulator (default)...")
        simulator.run_with_visualization()
    
    return simulator

