import numpy as np
from typing import Dict, List, Any, Optional


class DriverState:
    """Represents the state of a single driver during the race."""
    
    def __init__(self, name: str, starting_position: int, track_distance: float):
        self.name = name
        self.current_distance = 0.0  # Current position on track in km
        self.current_lap = 0
        self.current_lap_time = 0.0  # Time spent on current lap in seconds
        self.total_race_time = 0.0  # Total elapsed time in seconds
        self.position = starting_position  # Current race position
        self.completed_laps = 0
        
        # Car state
        self.tyre_compound = "medium"  # Can be soft, medium, hard
        self.tyre_age = 0  # Laps on current tyres
        self.fuel_load = 100.0  # Fuel percentage
        
        # Track position
        self.track_distance = track_distance  # Total track distance in km
        self.speed = 0.0  # Current speed in km/h
        
        # Overtaking state
        self.in_overtaking_zone = False
        self.attempting_overtake = False
        

class RaceState:
    """Represents the overall state of the race.

    Reads all track-related information directly from the provided `config` object
    (`config['track']`). The `track_data` variable is not used anywhere.
    """

    def __init__(self, drivers: List[DriverState], config: Dict):
        self.drivers = drivers
        self.config = config or {}

        # Extract simulator-specific config from full config
        simulator_cfg = self.config.get("simulator", {})

        # Simulation state
        self.current_tick = 0
        self.elapsed_time = 0.0  # Total elapsed time in seconds
        self.tick_duration = simulator_cfg.get("tick_duration")  # seconds per tick
        self.tick_rate = simulator_cfg.get("tick_rate")  # ticks per second

        # Track configuration (from config['track'])
        track_cfg = self.config.get("track", {})
        self.track_distance = track_cfg.get("distance")

        # Compute base lap time and ticks from mini_loops in config
        self.base_lap_time = self._calculate_base_lap_time()
        self.base_lap_ticks = int(self.base_lap_time / self.tick_duration) if self.tick_duration > 0 else 0

        # Overtaking zones (normalized list) - read from config['track']
        self.overtaking_zones = []
        cfg_z = track_cfg.get("overtakingZones") or track_cfg.get("overtaking_zones") or {}
        if isinstance(cfg_z, dict):
            self.overtaking_zones = list(cfg_z.values())
        elif isinstance(cfg_z, list):
            self.overtaking_zones = cfg_z

        print(f"\nRace State Initialized:")
        print(f"  Base Lap Time: {self.base_lap_time:.2f} seconds")
        print(f"  Base Lap Ticks: {self.base_lap_ticks} ticks")
        print(f"  Tick Duration: {self.tick_duration} seconds")
        print(f"  Number of Drivers: {len(self.drivers)}")
        print(f"  Number of Overtaking Zones: {len(self.overtaking_zones)}")
        
    def _calculate_base_lap_time(self) -> float:
        """Calculate the total base lap time from all mini loops."""
        # Read mini_loops from config['track'] and normalize dict -> list
        track_cfg = self.config.get("track", {})
        mini_loops = []
        cfg_ml = track_cfg.get("mini_loops", {})
        if isinstance(cfg_ml, dict):
            mini_loops = list(cfg_ml.values())
        elif isinstance(cfg_ml, list):
            mini_loops = cfg_ml

        total_time = sum(float(loop.get("base_lap_time")) for loop in mini_loops)
        return total_time
    
    def update_driver_positions(self):
        """Update race positions based on track distance and completed laps."""
        # Sort drivers by completed laps (descending) and current distance (descending)
        sorted_drivers = sorted(
            self.drivers,
            key=lambda d: (d.completed_laps, d.current_distance),
            reverse=True
        )
        
        # Assign positions
        for idx, driver in enumerate(sorted_drivers, start=1):
            driver.position = idx


def attempt_overtake(overtaking_driver: DriverState, target_driver: DriverState, 
                     overtaking_zone: Dict, race_state: RaceState) -> bool:
    """
    Attempt an overtake maneuver.
    
    Args:
        overtaking_driver: The driver attempting to overtake
        target_driver: The driver being overtaken
        overtaking_zone: The overtaking zone data (difficulty, name, distance)
        race_state: Current race state
        
    Returns:
        bool: True if overtake was successful, False otherwise
    """
    # Check if both drivers are in the same overtaking zone
    zone_distance = overtaking_zone.get("distance_from_start", 0)
    zone_tolerance = 0.1  # 100m tolerance for being "in" the zone
    
    # Check if overtaking driver is within the zone
    overtaking_in_zone = abs(overtaking_driver.current_distance - zone_distance) < zone_tolerance
    target_in_zone = abs(target_driver.current_distance - zone_distance) < zone_tolerance
    
    if not (overtaking_in_zone and target_in_zone):
        return False
    
    # Check if drivers are close enough (within 50m)
    distance_gap = abs(target_driver.current_distance - overtaking_driver.current_distance)
    if distance_gap > 0.05:  # 50m in km
        return False
    
    # Check if overtaking driver is behind
    if overtaking_driver.current_distance >= target_driver.current_distance:
        return False
    
    # Calculate overtake probability based on zone difficulty
    # Lower difficulty = easier to overtake
    zone_difficulty = overtaking_zone.get("difficulty", 0.5)
    base_success_rate = 1.0 - zone_difficulty  # Invert difficulty to get success rate
    
    # Add some randomness
    success = np.random.random() < base_success_rate
    
    if success:
        print(f"  🏁 {overtaking_driver.name} successfully overtook {target_driver.name} at {overtaking_zone.get('name')}!")
        # Swap positions slightly
        overtaking_driver.current_distance = target_driver.current_distance + 0.01
        return True
    else:
        print(f"  ❌ {overtaking_driver.name} failed to overtake {target_driver.name} at {overtaking_zone.get('name')}")
        return False


def check_overtaking_zones(race_state: RaceState):
    """Check if any drivers are in overtaking zones and mark their state."""
    for driver in race_state.drivers:
        driver.in_overtaking_zone = False
        
        for zone in race_state.overtaking_zones:
            zone_distance = zone.get("distance_from_start")
            zone_tolerance = 0.1  # 100m tolerance
            
            if abs(driver.current_distance - zone_distance) < zone_tolerance:
                driver.in_overtaking_zone = True
                break


def init_simulator(config, track):
    """
    Initialise the simulator with the given configuration and track data.
    This gives us our race state, and driver state.
    """
    competitors = config.get("competitors", [])
                    
    # Initialize driver states
    drivers = []
    for idx, competitor in enumerate(competitors):
        driver = DriverState(
            name=competitor.get("name", f"Driver{idx+1}"),
            starting_position=idx + 1,
            track_distance=config.get("track", {}).get("distance", 0)
        )
        drivers.append(driver)
    
    # Initialize race state
    # Pass the full config to RaceState so it can extract simulator settings as needed
    race_state = RaceState(
        drivers=drivers,
        config=config
    )
    
    print(f"\nInitialized {len(drivers)} drivers:")
    for driver in drivers:
        print(f"  - {driver.name} (P{driver.position})")
    
    return race_state
    