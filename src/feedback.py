"""
Feedback and state observation classes for RL agents.

This module provides a flexible way to encode race state information
for agents, allowing easy extension with new metrics.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DriverFeedback:
    """Encapsulates all observable information available to an agent at a decision point.
    
    This class is designed to be easily extensible - add new fields as needed
    without breaking existing code. The `to_vector()` method handles conversion
    to a fixed-size numpy array for neural network input.
    """
    
    # Driver state
    driver_name: str
    current_position: int
    starting_position: int
    speed: float  # km/h
    current_distance: float  # km on track
    completed_laps: int
    current_lap_time: float  # seconds
    total_race_time: float  # seconds
    
    # Car state
    tyre_age: int  # laps on current tyres
    fuel_load: float  # percentage
    tyre_compound: str = "medium"  # soft, medium, hard
    
    # Race context
    total_laps: int = 0
    laps_remaining: int = 0
    track_distance: float = 0.0  # total track length in km
    num_drivers: int = 3  # number of drivers in the race (used for normalization)

    # Position change statistics
    positions_gained_this_run: int = 0    # starting_position - current_position (positive = gained)
    avg_positions_gained: float = 0.0     # mean of (start - finish) across previous completed runs
    
    # Relative positioning
    gap_to_ahead: Optional[float] = None  # seconds
    gap_to_behind: Optional[float] = None  # seconds
    distance_to_ahead: Optional[float] = None  # km on track
    distance_to_behind: Optional[float] = None  # km on track
    position_ahead_name: Optional[str] = None
    position_behind_name: Optional[str] = None

    # Overtaking zone context
    upcoming_zone_distance: Optional[float] = None  # km to zone
    upcoming_zone_difficulty: Optional[float] = None  # 0-1
    upcoming_zone_name: Optional[str] = None
    in_overtaking_zone: bool = False

    # Relative performance vs car ahead
    speed_diff_to_ahead: float = 0.0       # driver.speed - ahead.speed (km/h); positive = faster
    gap_closing_rate: float = 0.0          # prev_gap - current_gap per tick; positive = closing

    # Per-race overtake statistics
    overtakes_attempted_this_race: int = 0
    overtake_success_rate: float = 0.5     # 0.5 neutral prior when no attempts yet
    
    def to_vector(self, normalize: bool = True) -> np.ndarray:
        """Convert feedback to a fixed-size numpy array for NN input.
        
        Args:
            normalize: Whether to normalize values to reasonable ranges.
            
        Returns:
            numpy array of shape (state_dim,)
        """
        features = []
        
        # Position features (normalize to 0-1 range based on actual driver count)
        n = max(self.num_drivers, 1)
        if normalize:
            features.append(self.current_position / n)
            features.append(self.starting_position / n)
            # Positions gained this run and career average (range -(n-1)..+(n-1))
            features.append(self.positions_gained_this_run / n)
            features.append(self.avg_positions_gained / n)
        else:
            features.append(float(self.current_position))
            features.append(float(self.starting_position))
            features.append(float(self.positions_gained_this_run))
            features.append(float(self.avg_positions_gained))
        
        # Speed (normalize to typical F1 speeds: 0-350 km/h)
        if normalize:
            features.append(self.speed / 350.0)
        else:
            features.append(self.speed)
        
        # Distance progress on current lap (0-1)
        if normalize and self.track_distance > 0:
            features.append(self.current_distance / self.track_distance)
        else:
            features.append(self.current_distance)
        
        # Lap progress (0-1)
        if normalize and self.total_laps > 0:
            features.append(self.completed_laps / self.total_laps)
            features.append(self.laps_remaining / self.total_laps)
        else:
            features.append(float(self.completed_laps))
            features.append(float(self.laps_remaining))
        
        # NOTE: tyre_age, fuel_load, tyre_compound are excluded from the state
        # vector because they are currently static (never updated during the race).
        # Re-add them once tyre degradation and fuel burn are modelled.
        
        # Gap to car ahead (normalize to typical gaps: 0-30 seconds)
        if self.gap_to_ahead is not None:
            if normalize:
                features.append(min(self.gap_to_ahead / 30.0, 1.0))
            else:
                features.append(self.gap_to_ahead)
        else:
            features.append(0.0)  # No car ahead (or very far)
        
        # Gap to car behind (normalize to typical gaps: 0-30 seconds)
        if self.gap_to_behind is not None:
            if normalize:
                features.append(min(self.gap_to_behind / 30.0, 1.0))
            else:
                features.append(self.gap_to_behind)
        else:
            features.append(0.0)  # No car behind
        
        # Distance to car ahead on track (normalize to typical: 0-1 km)
        if self.distance_to_ahead is not None:
            if normalize:
                features.append(min(self.distance_to_ahead / 1.0, 1.0))
            else:
                features.append(self.distance_to_ahead)
        else:
            features.append(1.0)  # Far ahead
        
        # Distance to car behind on track (normalize to typical: 0-1 km)
        if self.distance_to_behind is not None:
            if normalize:
                features.append(min(self.distance_to_behind / 1.0, 1.0))
            else:
                features.append(self.distance_to_behind)
        else:
            features.append(1.0)  # Far behind
        
        # Overtaking zone features
        if self.upcoming_zone_distance is not None:
            if normalize:
                # Normalize to decision range (e.g., 0-0.3 km)
                features.append(min(self.upcoming_zone_distance / 0.3, 1.0))
            else:
                features.append(self.upcoming_zone_distance)
        else:
            features.append(1.0)  # No zone nearby
        
        if self.upcoming_zone_difficulty is not None:
            features.append(self.upcoming_zone_difficulty)  # Already 0-1
        else:
            features.append(0.5)  # Default difficulty
        
        features.append(1.0 if self.in_overtaking_zone else 0.0)

        # Speed differential to car ahead (positive = faster than them → better for overtake)
        if normalize:
            features.append(np.clip(self.speed_diff_to_ahead / 200.0, -1.0, 1.0))
        else:
            features.append(self.speed_diff_to_ahead)

        # Gap closing rate (positive = gap decreasing → approaching car ahead)
        if normalize:
            features.append(np.clip(self.gap_closing_rate / 5.0, -1.0, 1.0))
        else:
            features.append(self.gap_closing_rate)

        # Overtake attempts this race (normalize to expected max of ~20 per race)
        if normalize:
            features.append(min(self.overtakes_attempted_this_race / 20.0, 1.0))
        else:
            features.append(float(self.overtakes_attempted_this_race))

        # Personal overtake success rate (already 0–1)
        features.append(self.overtake_success_rate)

        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def get_state_dim() -> int:
        """Return the dimensionality of the state vector."""
        # Count features from to_vector():
        # 1: current_position
        # 2: starting_position
        # 3: positions_gained_this_run
        # 4: avg_positions_gained (career average)
        # 5: speed
        # 6: distance progress
        # 7-8: lap progress (completed, remaining)
        # 9: gap to ahead
        # 10: gap to behind
        # 11: distance to ahead
        # 12: distance to behind
        # 13: zone distance
        # 14: zone difficulty
        # 15: in zone
        # 16: speed_diff_to_ahead
        # 17: gap_closing_rate
        # 18: overtakes_attempted_this_race
        # 19: overtake_success_rate
        return 19


def create_driver_feedback(driver, race_state, upcoming_zone) -> DriverFeedback:
    """Factory function to create DriverFeedback from driver and race state.
    
    Args:
        driver: DriverState object
        race_state: RaceState object
        upcoming_zone: dict with upcoming overtaking zone info
        
    Returns:
        DriverFeedback instance
    """
    # Calculate gaps to cars ahead/behind
    gap_ahead = None
    gap_behind = None
    dist_ahead = None
    dist_behind = None
    name_ahead = None
    name_behind = None
    speed_diff_to_ahead = 0.0
    gap_closing_rate = 0.0

    # Sort drivers by position to find adjacent cars
    sorted_drivers = sorted(race_state.drivers, key=lambda d: d.position)
    driver_idx = next((i for i, d in enumerate(sorted_drivers) if d == driver), None)

    if driver_idx is not None:
        # Car ahead
        if driver_idx > 0:
            ahead_driver = sorted_drivers[driver_idx - 1]
            gap_ahead = ahead_driver.total_race_time - driver.total_race_time
            ahead_total_dist = ahead_driver.completed_laps * race_state.track_distance + ahead_driver.current_distance
            driver_total_dist = driver.completed_laps * race_state.track_distance + driver.current_distance
            dist_ahead = ahead_total_dist - driver_total_dist
            name_ahead = ahead_driver.name
            # Speed differential (positive = this driver is faster → better for overtake)
            speed_diff_to_ahead = driver.speed - ahead_driver.speed
            # Gap closing rate: positive = gap decreased since last tick = closing in
            gap_hist = getattr(driver, "gap_to_ahead_history", [])
            if gap_hist:
                gap_closing_rate = gap_hist[-1] - gap_ahead

        # Car behind
        if driver_idx < len(sorted_drivers) - 1:
            behind_driver = sorted_drivers[driver_idx + 1]
            gap_behind = driver.total_race_time - behind_driver.total_race_time
            driver_total_dist = driver.completed_laps * race_state.track_distance + driver.current_distance
            behind_total_dist = behind_driver.completed_laps * race_state.track_distance + behind_driver.current_distance
            dist_behind = driver_total_dist - behind_total_dist
            name_behind = behind_driver.name

    # Per-race overtake statistics
    attempted = getattr(driver, "overtakes_attempted", 0)
    succeeded = getattr(driver, "overtakes_succeeded", 0)
    overtake_success_rate = (succeeded / attempted) if attempted > 0 else 0.5

    # Extract upcoming zone info
    zone_distance = None
    zone_difficulty = None
    zone_name = None

    if upcoming_zone is not None:
        zone_dist_from_start = upcoming_zone.get("distance_from_start", 0.0)
        zone_distance = zone_dist_from_start - driver.current_distance
        if zone_distance < 0:
            zone_distance += race_state.track_distance  # Handle wrap-around
        zone_difficulty = upcoming_zone.get("difficulty", 0.5)
        zone_name = upcoming_zone.get("name", "Unknown")

    total_laps_cfg = race_state.config.get("race_settings", {}).get("total_laps", 0)
    runs_done = getattr(driver, "runs_completed", 0)
    avg_gained = (
        getattr(driver, "cumulative_positions_gained", 0.0) / runs_done
        if runs_done > 0 else 0.0
    )

    return DriverFeedback(
        driver_name=driver.name,
        current_position=driver.position,
        starting_position=driver.starting_position,
        speed=driver.speed,
        current_distance=driver.current_distance,
        completed_laps=driver.completed_laps,
        current_lap_time=driver.current_lap_time,
        total_race_time=driver.total_race_time,
        tyre_age=driver.tyre_age,
        fuel_load=driver.fuel_load,
        tyre_compound=driver.tyre_compound,
        total_laps=total_laps_cfg,
        laps_remaining=total_laps_cfg - driver.completed_laps,
        track_distance=race_state.track_distance,
        num_drivers=len(race_state.drivers),
        positions_gained_this_run=driver.starting_position - driver.position,
        avg_positions_gained=avg_gained,
        gap_to_ahead=gap_ahead,
        gap_to_behind=gap_behind,
        distance_to_ahead=dist_ahead,
        distance_to_behind=dist_behind,
        position_ahead_name=name_ahead,
        position_behind_name=name_behind,
        upcoming_zone_distance=zone_distance,
        upcoming_zone_difficulty=zone_difficulty,
        upcoming_zone_name=zone_name,
        in_overtaking_zone=getattr(driver, "in_overtaking_zone", False),
        speed_diff_to_ahead=speed_diff_to_ahead,
        gap_closing_rate=gap_closing_rate,
        overtakes_attempted_this_race=attempted,
        overtake_success_rate=overtake_success_rate,
    )
