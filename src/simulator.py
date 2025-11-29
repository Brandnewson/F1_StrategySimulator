import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
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
        
        # Build smooth speed profile from mini-loops
        self.speed_profile = self._build_speed_profile()
        
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

        # plot speed profile for debugging
        if self.config.get("debugMode"):
            self.plot_speed_profile(show_loops=True)
        
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
    
    def _build_speed_profile(self, speed_multiplier: float = 1.0) -> Dict:
        """
        Build a smooth speed profile that transitions gradually between mini-loops.
        
        Instead of sudden speed jumps at loop boundaries, this creates a continuous
        speed curve using cubic spline interpolation. The profile is calibrated so
        that the total time to traverse each loop matches its base_lap_time.
        
        Args:
            speed_multiplier: Factor to adjust all speeds (e.g., for tyre compounds).
                              >1.0 = faster, <1.0 = slower
        
        Returns:
            Dict containing:
                - 'spline': CubicSpline interpolator for speed lookup
                - 'distances': Array of distance sample points
                - 'speeds': Array of speed values at each distance
                - 'multiplier': The speed multiplier used
        """
        if not self.mini_loops:
            # Fallback if no loops defined
            return {
                'spline': None,
                'distances': np.array([0, self.track_distance]),
                'speeds': np.array([0.05, 0.05]),
                'multiplier': speed_multiplier
            }
        
        # Step 1: Create anchor points at loop midpoints with average speeds
        anchor_distances = []
        anchor_speeds = []
        
        for loop in self.mini_loops:
            # Use midpoint of each loop as anchor
            midpoint = (loop["start"] + loop["end"]) / 2
            base_speed = loop["speed"] * speed_multiplier
            anchor_distances.append(midpoint)
            anchor_speeds.append(base_speed)
        
        # Step 2: Add boundary points for smooth wrap-around (track is circular)
        # Extend anchors to handle the track boundary smoothly
        
        # Add a point before the first loop (wrap from last loop)
        last_loop = self.mini_loops[-1]
        last_speed = last_loop["speed"] * speed_multiplier
        # Virtual point before track start (negative distance, wrapping from end)
        pre_start_dist = -(self.track_distance - (last_loop["start"] + last_loop["end"]) / 2)
        
        # Add a point after the last loop (wrap to first loop)
        first_loop = self.mini_loops[0]
        first_speed = first_loop["speed"] * speed_multiplier
        post_end_dist = self.track_distance + (first_loop["start"] + first_loop["end"]) / 2
        
        # Build extended arrays for periodic spline fitting
        extended_distances = [pre_start_dist] + anchor_distances + [post_end_dist]
        extended_speeds = [last_speed] + anchor_speeds + [first_speed]
        
        # Step 3: Create cubic spline interpolator
        # Use 'natural' boundary condition for smooth transitions
        spline = CubicSpline(extended_distances, extended_speeds, bc_type='natural')
        
        # Step 4: Sample the spline at high resolution for fast lookup
        # Use enough points for smooth interpolation (~10m resolution)
        num_samples = max(100, int(self.track_distance * 100))  # ~100 points per km
        sample_distances = np.linspace(0, self.track_distance, num_samples)
        sample_speeds = spline(sample_distances)
        
        # Ensure no negative speeds (can happen with aggressive spline fitting)
        sample_speeds = np.maximum(sample_speeds, 0.01)
        
        # Step 5: Calibrate speeds to match expected loop times
        # This is crucial: we want the integral of time = integral of (distance / speed)
        # to match the sum of base_lap_times
        calibrated_speeds = self._calibrate_speed_profile(sample_distances, sample_speeds)
        
        # Rebuild spline with calibrated speeds
        calibrated_spline = CubicSpline(sample_distances, calibrated_speeds, bc_type='natural')
        
        return {
            'spline': calibrated_spline,
            'distances': sample_distances,
            'speeds': calibrated_speeds,
            'multiplier': speed_multiplier
        }
    
    def _calibrate_speed_profile(self, distances: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        """
        Calibrate the speed profile so total lap time matches sum of base_lap_times.
        
        This adjusts the overall speed scale while preserving the shape of the profile.
        
        Args:
            distances: Array of distance sample points
            speeds: Array of uncalibrated speed values
            
        Returns:
            Calibrated speed array
        """
        # Calculate expected total lap time from config
        expected_total_time = sum(loop["base_time"] for loop in self.mini_loops)
        
        # Calculate actual time with current speed profile
        # Time = integral of (1/speed) over distance
        # Using trapezoidal integration
        dt = np.diff(distances)
        avg_speeds = (speeds[:-1] + speeds[1:]) / 2
        actual_time = np.sum(dt / avg_speeds)
        
        # Calculate scaling factor
        if actual_time > 0:
            scale_factor = actual_time / expected_total_time
        else:
            scale_factor = 1.0
        
        # Apply scaling to speeds (faster speeds = shorter time)
        calibrated_speeds = speeds * scale_factor
        
        return calibrated_speeds
    
    def plot_speed_profile(self, show_loops: bool = True) -> None:
        """
        Plot the speed profile for debugging and visualization.
        
        Shows the smooth speed curve compared to the original step-function
        speeds from mini-loops.
        
        Args:
            show_loops: If True, overlay the original loop speeds as step function
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        distances = self.speed_profile['distances']
        speeds_kms = self.speed_profile['speeds']
        speeds_kmh = speeds_kms * 3600  # Convert to km/h for readability
        
        # Plot smooth speed profile
        ax1.plot(distances, speeds_kmh, 'b-', linewidth=2, label='Smooth Speed Profile')
        
        if show_loops:
            # Overlay original step-function speeds
            for loop in self.mini_loops:
                loop_speed_kmh = loop["speed"] * 3600
                ax1.hlines(loop_speed_kmh, loop["start"], loop["end"], 
                          colors='r', linestyles='--', linewidth=1.5, alpha=0.7)
                ax1.axvline(loop["start"], color='gray', linestyle=':', alpha=0.5)
                # Add loop name at midpoint
                midpoint = (loop["start"] + loop["end"]) / 2
                ax1.text(midpoint, loop_speed_kmh + 5, loop["name"], 
                        ha='center', fontsize=8, rotation=45)
        
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_title('Smooth Speed Profile vs Original Loop Speeds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative time along track
        dt = np.diff(distances)
        avg_speeds = (speeds_kms[:-1] + speeds_kms[1:]) / 2
        segment_times = dt / avg_speeds
        cumulative_time = np.concatenate([[0], np.cumsum(segment_times)])
        
        ax2.plot(distances, cumulative_time, 'g-', linewidth=2)
        ax2.set_xlabel('Track Distance (km)')
        ax2.set_ylabel('Cumulative Time (s)')
        ax2.set_title(f'Cumulative Lap Time (Total: {cumulative_time[-1]:.2f}s)')
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines at loop boundaries
        for loop in self.mini_loops:
            ax2.axvline(loop["start"], color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def _get_speed_at_distance(self, distance: float) -> float:
        """Get the speed (km/s) for a given track distance using smooth profile."""
        # Wrap distance within track bounds
        wrapped_distance = distance % self.track_distance
        
        # Use spline interpolation for smooth speed
        if self.speed_profile['spline'] is not None:
            speed = float(self.speed_profile['spline'](wrapped_distance))
            # Ensure positive speed
            return max(speed, 0.01)
        
        # Fallback to loop-based lookup
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
            
            # # Apply time penalty to overtaken driver
            # target_driver.current_lap_time += self.overtake_time_penalty
            # target_driver.total_race_time += self.overtake_time_penalty
            
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
        # keep the start marker object so we can build the legend explicitly
        start_marker = ax.scatter(start_x, start_y, s=200, c="green", marker="s",
                                  zorder=10, label="Start/Finish")
        
        # --- Overtaking zone side labels (placed off-track) ---
        # Place a plain text label for each overtaking zone offset to the side of the circuit.
        # Uses a small perpendicular offset from the track tangent and a slight stagger to reduce overlap.
        if self.track_coords is not None and getattr(self.race_state, "overtaking_zones", None):
            coords = self.track_coords
            last_idx = len(coords) - 1
            for i, zone in enumerate(self.race_state.overtaking_zones, start=1):
                try:
                    zone_dist = float(zone.get("distance_from_start", zone.get("distance", 0)))
                except Exception:
                    zone_dist = 0.0
                # find nearest coordinate for this zone distance (coords['distance'] in same units as zone_dist)
                nearest_idx = (coords["distance"] - zone_dist).abs().idxmin()
                zx = coords.at[nearest_idx, "X"]
                zy = coords.at[nearest_idx, "Y"]

                # approximate tangent using neighbours
                prev_idx = max(nearest_idx - 1, 0)
                next_idx = min(nearest_idx + 1, last_idx)
                tx = coords.at[next_idx, "X"] - coords.at[prev_idx, "X"]
                ty = coords.at[next_idx, "Y"] - coords.at[prev_idx, "Y"]
                norm = np.hypot(tx, ty) if (tx != 0 or ty != 0) else 1.0
                # perpendicular direction (unit)
                px, py = -ty / norm, tx / norm

                # offset in meters (track coords appear to be in meters)
                base_offset_m = 120  # base offset distance from track
                stagger = (i % 3) * 25  # stagger every zone to reduce label overlap
                offset_m = base_offset_m + stagger

                label_x = zx + px * offset_m
                label_y = zy + py * offset_m

                # display label with difficulty in parentheses, wrapped if long
                zone_name = zone.get("name", f"Zone {i}")
                difficulty = zone.get("difficulty", None)
                if difficulty is not None:
                    label_text = f"{zone_name} ({difficulty:.2f})"
                else:
                    label_text = zone_name

                # draw a subtle connector line and the label box on the side
                ax.plot([zx, label_x], [zy, label_y], color="#666666", linewidth=0.8,
                        alpha=0.6, zorder=14)
                ax.text(label_x, label_y, label_text, fontsize=9, zorder=15,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"))

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

        # Build legend explicitly to avoid matplotlib auto-picking other artists (e.g. text bboxes)
        legend_handles = [start_marker] + driver_dots
        legend_labels = ["Start/Finish"] + [d.name for d in self.race_state.drivers]
        ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=9)
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

                    # Convert distance gap to an estimated time gap using current speeds.
                    # Use closing-time if trailing car is faster: time = gap / (v_trailing - v_leader)
                    # Otherwise fallback to leader's speed to estimate how long leader needs to cover that gap.
                    # driver.speed and leader.speed are in km/h; convert to km/s.
                    leader_kms = max(leader.speed / 3600.0, 1e-6)
                    driver_kms = max(driver.speed / 3600.0, 1e-6)
                    relative_kms = driver_kms - leader_kms

                    if relative_kms > 1e-6:
                        # trailing car is closing — estimate time until it reaches the leader's current position
                        # gap_time = gap_dist / relative_kms
                        gap_time = gap_dist / leader_kms
                    else:
                        # trailing car not closing — estimate time the leader needs to cover the gap
                        gap_time = gap_dist / leader_kms

                    # If gap_time is negative or NaN for any reason, clamp to 0
                    try:
                        gap_time = float(gap_time)
                        if gap_time < 0 or np.isnan(gap_time) or np.isinf(gap_time):
                            gap_time = 0.0
                    except Exception:
                        gap_time = 0.0

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
        
        try:
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            # allow graceful exit during debugging / long layout operations
            print("Animation interrupted by user (KeyboardInterrupt). Closing.")
        
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

