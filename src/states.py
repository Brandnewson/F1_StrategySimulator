import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import textwrap
from matplotlib.lines import Line2D

# Agents
from agents import BaseAgent, make_param_agent, make_random_agent

class DriverState:
    """Represents the state of a single driver during the race."""
    
    def __init__(self, name: str, starting_position: int, track_distance: float):
        self.name = name
        # default start at 0.0 — actual grid offsets are assigned in init_race_state
        self.current_distance = 0.0
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
        # Agent assigned to this driver (set in init_simulator)
        self.agent: Optional[BaseAgent] = None
        

class RaceState:
    """Represents the overall state of the race.

    Reads all track-related information directly from the provided `config` object
    (`config['track']`). The `track_data` variable is not used anywhere.
    """

    def __init__(self, drivers: List[DriverState], config: Dict, track_data: Optional[Dict]):
        self.drivers = drivers
        self.config = config
        self.track_data = track_data
        if isinstance(self.track_data, dict) and "coordinates" in self.track_data:
            self.track_coords = self.track_data["coordinates"]

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

        # visualise overtaking zones for debugging
        if isinstance(self.track_data, dict) and "coordinates" in self.track_data and self.config.get("debugMode", False):
            self._visualise_overtaking_zones()
        
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

    def _visualise_overtaking_zones(self, figsize=(10,6), show=True):
        """
        Visualise overtaking zones on the track.
        Expects track coordinates DataFrame in self.track_coords with columns: 'X','Y','distance'.
        Each overtaking zone must have 'distance_from_start' and optionally 'name' and 'difficulty'.
        Legend shows zone name and difficulty. Labels are wrapped and offset to reduce overlap.
        """
        if self.track_coords is None:
            print("No track coordinates available to visualise overtaking zones.")
            return None

        coords = self.track_coords
        required_cols = {"X", "Y", "distance"}
        if not required_cols.issubset(set(coords.columns)):
            print(f"Track coordinates missing required columns {required_cols}. Found: {coords.columns.tolist()}")
            return None

        plt.figure(figsize=figsize)
        # plot track centreline
        plt.plot(coords["X"], coords["Y"], color="#444444", linewidth=2, label="Track centreline")

        # Prepare legend handles/labels
        legend_handles = []
        legend_labels = []

        last_idx = len(coords) - 1

        # First pass: compute marker positions and base perpendiculars/offsets
        label_items = []  # will hold dicts with marker pos, perp vector, base offset, zone meta
        # scale reference based on track extent
        x_span = coords["X"].max() - coords["X"].min()
        y_span = coords["Y"].max() - coords["Y"].min()
        diag = np.hypot(x_span, y_span) if (x_span != 0 or y_span != 0) else 1.0

        for idx, zone in enumerate(self.overtaking_zones, start=1):
            zone_dist = zone.get("distance_from_start", 0.0)
            nearest_idx = (coords["distance"] - float(zone_dist)).abs().idxmin()
            zx = coords.at[nearest_idx, "X"]
            zy = coords.at[nearest_idx, "Y"]

            # visual encoding
            difficulty = float(zone.get("difficulty", 0.5))
            marker_size = max(40, 180 * (1.0 - difficulty + 0.05))
            cmap_val = np.clip(1.0 - difficulty, 0.0, 1.0)
            color = plt.cm.RdYlGn(cmap_val)

            zone_name = zone.get("name", f"Zone{idx}")
            wrapped_name = textwrap.fill(zone_name, width=18)

            # compute local tangent and perpendicular
            prev_idx = max(nearest_idx - 1, 0)
            next_idx = min(nearest_idx + 1, last_idx)
            tx = coords.at[next_idx, "X"] - coords.at[prev_idx, "X"]
            ty = coords.at[next_idx, "Y"] - coords.at[prev_idx, "Y"]
            norm = np.hypot(tx, ty) if (tx != 0 or ty != 0) else 1.0
            px, py = -ty / norm, tx / norm

            # base offset scaled to track size and marker size
            base_offset = (diag * 0.02) + (marker_size / 1000.0)

            label_items.append({
                "idx": idx,
                "zone": zone,
                "zx": zx,
                "zy": zy,
                "px": px,
                "py": py,
                "base_offset": base_offset,
                "offset": base_offset,
                "marker_size": marker_size,
                "color": color,
                "wrapped_name": wrapped_name,
                "zone_name": zone_name,
                "difficulty": difficulty
            })

            # legend entry
            legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markeredgecolor='#222222',
                                         markersize=8))
            legend_labels.append(f"{zone_name} (diff={difficulty:.2f})")

        # Resolve close labels: for each pair closer than threshold, push in opposite perp directions
        positions = np.array([[it["zx"], it["zy"]] for it in label_items])
        if len(positions) > 1:
            # threshold relative to track diag
            min_dist = diag * 0.08
            n = len(label_items)
            for i in range(n):
                for j in range(i + 1, n):
                    xi, yi = positions[i]
                    xj, yj = positions[j]
                    dist = np.hypot(xj - xi, yj - yi)
                    if dist < min_dist and dist > 0:
                        # determine which way along perp to move each label so they go opposite
                        vi = np.array([xj - xi, yj - yi])
                        perp_i = np.array([label_items[i]["px"], label_items[i]["py"]])
                        perp_j = np.array([label_items[j]["px"], label_items[j]["py"]])
                        # if dot(vi, perp_i) >= 0 then move i in +perp_i and j in -perp_j, else invert
                        sign = 1.0 if np.dot(vi, perp_i) >= 0 else -1.0
                        push = (min_dist - dist) * 0.6  # scale push
                        label_items[i]["offset"] += sign * push
                        label_items[j]["offset"] -= sign * push

        # Draw markers and labels using final offsets
        for it in label_items:
            zx = it["zx"]
            zy = it["zy"]
            color = it["color"]
            marker_size = it["marker_size"]
            wrapped_name = it["wrapped_name"]
            offset = it["offset"]
            px = it["px"]
            py = it["py"]

            plt.scatter(zx, zy, s=marker_size, color=color, edgecolor="#222222", zorder=5)

            text_x = zx + px * offset
            text_y = zy + py * offset

            plt.text(text_x, text_y, wrapped_name, fontsize=9, verticalalignment="center", zorder=6,
                     bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))

        plt.title("Track with Overtaking Zones")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(alpha=0.3)

        # include centreline in legend + zones
        main_handle = Line2D([0], [0], color="#444444", lw=2)
        all_handles = [main_handle] + legend_handles
        all_labels = ["Track centreline"] + legend_labels

        plt.legend(all_handles, all_labels, loc="upper right", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        if show:
            plt.show()
        return plt
    
def attempt_overtake(overtaking_driver: DriverState, target_driver: DriverState, 
                     overtaking_zone: Dict, race_state: RaceState) -> bool:
    """
    Attempt an overtake maneuver. (DEBUGGING FUNCTION)
    
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


def init_race_state(config, track):
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
        # Attach colour information from competitor config (support both British 'colour'
        # and American 'color' keys). Normalize simple multi-word names by removing
        # spaces (e.g. "dark blue" -> "darkblue") so they match matplotlib CSS names
        # when possible. Hex strings (starting with '#') are preserved.
        comp_colour = None
        if isinstance(competitor, dict):
            comp_colour = competitor.get("colour") or competitor.get("color")
        if isinstance(comp_colour, str):
            cc = comp_colour.strip()
            # preserve hex codes and tuples — only collapse simple names with spaces
            if cc.startswith("#"):
                driver.colour = cc
                driver.color = cc
            else:
                cleaned = cc.replace(" ", "")
                driver.colour = cleaned
                driver.color = cleaned
                # Keep original around for debugging if needed
                driver._original_colour = comp_colour
        else:
            driver.colour = None
            driver.color = None
        drivers.append(driver)
    
    # Assign realistic starting grid offsets:
    # last car (highest starting_position) -> distance 0.0
    # cars ahead -> positive distance ahead by grid_gap_meters each
    num_drivers = len(drivers)
    grid_gap_meters = 8  # 8 meters between grid positions (configurable)
    for d in drivers:
        # position: 1..N (1 is pole). compute distance ahead of the last car:
        # distance_km = (num_drivers - starting_position) * gap_m
        d.current_distance = ((num_drivers - d.position) * grid_gap_meters) / 1000.0

    # Initialize race state
    # Pass the full config to RaceState so it can extract simulator settings as needed
    race_state = RaceState(
        drivers=drivers,
        config=config,
        track_data=track
    )
    # Wire agents to drivers. If competitor config contains an 'agent' entry, use it.
    for idx, (driver, competitor) in enumerate(zip(drivers, competitors)):
        agent_spec = (competitor.get("agent") or "base").lower() if isinstance(competitor, dict) else "base"
        if agent_spec == "random":
            driver.agent = make_random_agent(name=f"{driver.name}_rand")
        elif agent_spec.startswith("param:"):
            # format: param:cons,norm,aggr  e.g. param:0.2,0.6,0.2
            try:
                parts = agent_spec.split(":", 1)[1].split(",")
                c, n, a = [float(x) for x in parts]
                driver.agent = make_param_agent(name=f"{driver.name}_param", cons_weight=c, norm_weight=n, aggr_weight=a)
            except Exception:
                driver.agent = BaseAgent(name=f"{driver.name}_base")
        elif agent_spec in ("aggressive", "aggr"):
            driver.agent = make_param_agent(name=f"{driver.name}_aggr", cons_weight=0.05, norm_weight=0.25, aggr_weight=0.7)
        elif agent_spec in ("conservative", "cons"):
            driver.agent = make_param_agent(name=f"{driver.name}_cons", cons_weight=0.7, norm_weight=0.25, aggr_weight=0.05)
        else:
            driver.agent = BaseAgent(name=f"{driver.name}_base")

    print(f"\nInitialized {len(drivers)} drivers:")
    for driver in drivers:
        print(f"  - {driver.name} (P{driver.position}) agent={driver.agent.name}")

    if config.get("debugMode"):
        print("Debug mode is ON. Testing demo agent decisions...")
        # Simple demo run: ask each agent for an action at a decision point and optionally attempt overtakes
        print("\nDemo: agent decisions at decision points:")
        # If no overtaking zones configured, add a temporary demo zone so agents have context
        demo_zone_added = False
        if not race_state.overtaking_zones:
            demo_zone = {"name": "DemoZone", "distance_from_start": 0.06, "difficulty": 0.3}
            race_state.overtaking_zones.append(demo_zone)
            demo_zone_added = True

        # Place drivers near the demo zone for the demo
        for i, d in enumerate(drivers):
            # leader at zone distance, others slightly behind
            d.current_distance = race_state.overtaking_zones[0].get("distance_from_start", 0.06) - (i * 0.01)
            # simple gap estimate in seconds for agent inputs
            d.gap_to_ahead = None if d.position == 1 else 0.8

        # Let each agent decide and attempt an overtake if they choose to
        for d in drivers:
            action = d.agent.get_action(d, race_state)
            print(f"  - {d.name}: risk={action.risk_level.name}, attempt_overtake={action.attempt_overtake}")
            if action.attempt_overtake and d.position > 1 and race_state.overtaking_zones:
                # find driver ahead by position
                ahead = next((x for x in drivers if x.position == d.position - 1), None)
                if ahead:
                    success = attempt_overtake(d, ahead, race_state.overtaking_zones[0], race_state)
                    print(f"    -> overtake attempted against {ahead.name}: {'SUCCESS' if success else 'FAIL'}")

        # remove demo zone if it was added
        if demo_zone_added:
            race_state.overtaking_zones.pop(0)

    return race_state
