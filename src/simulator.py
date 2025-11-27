import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List
from dataclasses import dataclass


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
        self.mini_loops = self._build_mini_loops(track_config)
        
        # Simulation settings
        sim_config = config.get("simulator", {})
        self.tick_duration = sim_config.get("tick_duration", 0.01)  # seconds
        self.tick_rate = sim_config.get("tick_rate", 100)  # Hz
        
        # Time penalty for being overtaken (secs)
        self.overtake_time_penalty = 0.3

        # Time penalty for failed overtake attempt (secs)   
        self.failed_overtake_time_penalty = 0.4
        
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
        
        # Colors for drivers
        self.driver_colors = plt.cm.tab10(np.linspace(0, 1, len(race_state.drivers)))
        
    def _build_mini_loops(self, track_config: Dict) -> List[Dict]:
        """Build sorted list of mini loops with speed calculations."""
        mini_loops_cfg = track_config.get("mini_loops", {})
        
        # Normalize dict to list
        if isinstance(mini_loops_cfg, dict):
            loops = list(mini_loops_cfg.values())
        else:
            loops = mini_loops_cfg
            
        # Calculate speed for each mini loop (distance / time = km/s)
        processed_loops = []
        for loop in loops:
            start = loop.get("start_distance", 0)
            end = loop.get("end_distance", 0)
            base_time = loop.get("base_lap_time", 10.0)
            distance = end - start
            
            if base_time > 0:
                speed = distance / base_time  # km per second
            else:
                speed = 0.01  # fallback
                
            processed_loops.append({
                "start": start,
                "end": end,
                "name": loop.get("name", "Unknown"),
                "base_time": base_time,
                "speed": speed  # km/s
            })
            
        # Sort by start distance
        processed_loops.sort(key=lambda x: x["start"])
        return processed_loops
    
    def _get_speed_at_distance(self, distance: float) -> float:
        """Get the speed (km/s) for a given track distance."""
        # Wrap distance within track bounds
        wrapped_distance = distance % self.track_distance
        
        for loop in self.mini_loops:
            if loop["start"] <= wrapped_distance < loop["end"]:
                return loop["speed"]
        
        # Fallback to last loop if at end of track
        if self.mini_loops:
            return self.mini_loops[-1]["speed"]
        return 0.05  # Default speed
    
    def _get_distance_to_xy(self, distance: float) -> tuple:
        """Convert track distance to X,Y coordinates for visualization."""
        if self.track_coords is None:
            return (0, 0)
            
        # Wrap distance
        wrapped_distance = distance % self.track_distance
        
        # Find nearest coordinate point
        coords = self.track_coords
        nearest_idx = (coords["distance"] - wrapped_distance).abs().idxmin()
        return (coords.at[nearest_idx, "X"], coords.at[nearest_idx, "Y"])
    
    def _update_driver(self, driver, dt: float):
        """Update a single driver's position for one tick."""
        # Get current speed based on track position
        speed = self._get_speed_at_distance(driver.current_distance)
        
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
        speed_diff = overtaking_driver.speed - target_driver.speed  # in km/h
        gap = self._calculate_track_gap(overtaking_driver, target_driver)  # in km

        # Favor overtakes when gap is small and speed difference is positive
        if gap < 0.04:  # less than 40m
            success_probability *= 1.1
            print(f"    Small gap resulted in probability increasing from {success_probability/1.1:.2f} to {success_probability:.2f}")
        else:  # penalize larger gaps larger than 40m
            success_probability *= max(0.0, 1.0 - (gap - 0.04) * 5)  # reduce prob for larger gaps
            print(f"    Larger gap resulted in probability decreasing from {success_probability / max(0.0, 1.0 - (gap - 0.04) * 5):.2f} to {success_probability:.2f}")
        
        # TODO: Favor overtakes when speed difference is significant (reflects tyre choices and car performance)
        

        # Print probability of overtaking each time 
        print(f"  Overtake probability for {overtaking_driver.name} attempting to overtake {target_driver.name} at {zone.get('name')}: {success_probability:.2f}")
        
        success = np.random.random() < success_probability
        
        if success:
            print(f"  OVERTAKE: {overtaking_driver.name} overtook {target_driver.name} "
                  f"at {zone.get('name')}!")
            
            # Apply time penalty to overtaken driver
            target_driver.current_lap_time += self.overtake_time_penalty
            target_driver.total_race_time += self.overtake_time_penalty
            
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
        print(f"\n{'='*60}")
        print(f"Starting Race: {self.total_laps} laps at {self.config['track']['name'].upper()}")
        print(f"{'='*60}\n")
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot track
        coords = self.track_coords
        ax.plot(coords["X"], coords["Y"], color="#888888", linewidth=8, 
                alpha=0.3, zorder=1, label="_nolegend_")
        ax.plot(coords["X"], coords["Y"], color="#333333", linewidth=2, 
                zorder=2, label="_nolegend_")
        
        # Mark start/finish
        start_x, start_y = coords.at[0, "X"], coords.at[0, "Y"]
        ax.scatter(start_x, start_y, s=200, c="green", marker="s", 
                   zorder=10, label="Start/Finish")
        
        # Initialize driver markers
        driver_dots = []
        for i, driver in enumerate(self.race_state.drivers):
            dot, = ax.plot([], [], 'o', markersize=15, 
                          color=self.driver_colors[i],
                          markeredgecolor='white', markeredgewidth=2,
                          zorder=20, label=driver.name)
            driver_dots.append(dot)
        
        # Info text
        info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_title(f"F1 Race Simulator - {self.config['track']['name'].upper()}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        
        def init():
            """Initialize animation."""
            for dot in driver_dots:
                dot.set_data([], [])
            info_text.set_text("")
            return driver_dots + [info_text]
        
        def animate(frame):
            """Animation frame update."""
            # Run simulation step
            self._simulation_step()
            
            # Update driver positions on track
            for i, driver in enumerate(self.race_state.drivers):
                x, y = self._get_distance_to_xy(driver.current_distance)
                driver_dots[i].set_data([x], [y])
            
            # Update info text
            info_lines = [f"Lap: {self.race_state.drivers[0].completed_laps + 1}/{self.total_laps}",
                         f"Time: {self.race_state.elapsed_time:.1f}s", ""]
            
            leader = min(self.race_state.drivers, key=lambda d: d.position)
            for driver in sorted(self.race_state.drivers, key=lambda d: d.position):
                if driver == leader:
                    gap_str = "Leader"
                else:
                    # Calculate gap based on track position (how far behind the leader)
                    leader_total = leader.completed_laps * self.track_distance + leader.current_distance
                    driver_total = driver.completed_laps * self.track_distance + driver.current_distance
                    gap_dist = leader_total - driver_total  # Positive if driver is behind
                    
                    # Convert distance gap to approximate time gap using average speed
                    avg_speed = self.track_distance / self.race_state.base_lap_time if self.race_state.base_lap_time > 0 else 0.065
                    gap_time = gap_dist / avg_speed if avg_speed > 0 else 0
                    gap_str = f"+{gap_time:.2f}s" if gap_time > 0 else f"{gap_time:.2f}s"
                info_lines.append(
                    f"P{driver.position}: {driver.name:20s} "
                    f"Lap {driver.completed_laps + 1} | {gap_str}"
                )
            
            if self.race_finished:
                info_lines.append("")
                info_lines.append(f"WINNER: {self.winner.name}!")
            
            info_text.set_text("\n".join(info_lines))
            
            return driver_dots + [info_text]
        
        # Create animation
        # Estimate max frames: (total_laps * base_lap_time) / (animation_interval/1000 * ticks_per_frame * tick_duration)
        estimated_race_time = self.total_laps * self.race_state.base_lap_time
        estimated_frames = int(estimated_race_time / (self.ticks_per_frame * self.tick_duration)) + 10000
        
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=estimated_frames,
                            interval=self.animation_interval,
                            blit=True, repeat=False,
                            cache_frame_data=False)
        
        # Handle race completion
        def on_close(event):
            """Handle window close."""
            self.race_finished = True
        
        fig.canvas.mpl_connect('close_event', on_close)
        
        plt.tight_layout()
        plt.show()
        
        # Print final results
        self._print_results()
    
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

