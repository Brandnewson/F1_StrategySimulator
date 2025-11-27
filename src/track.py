import matplotlib.pyplot as plt
import pandas as pd

def load_track(config):
    """Load track data based on the configuration."""
    track_config = config.get("track", {})
    track_name = track_config.get("name")

    # Extract basic track data
    track_data = {
        "name": track_name,
        "num_corners": track_config.get("num_corners"),
        "distance": track_config.get("distance"),
        "corners": [],
        "mini_loops": [],
        "overtaking_zones": []
    }

    # Dynamically load corners based on num_corners
    num_corners = track_config.get("num_corners", 0)
    for i in range(1, num_corners + 1):
        corner_key = f"corner{i}"
        corner_data = track_config.get(corner_key, {})
        if corner_data:  # Only add if corner data exists
            track_data["corners"].append({
                "name": corner_data.get("name"),
                "difficulty": corner_data.get("difficulty"),
                "base_time": corner_data.get("base_time")
            })

    # Load mini loops
    for i in range(1, 11):  # Assuming there are up to 10 mini loops
        mini_loop_key = f"miniLoop{i}"
        mini_loop_data = track_config.get(mini_loop_key, {})
        if mini_loop_data:  # Only add if mini loop data exists
            track_data["mini_loops"].append({
                "name": mini_loop_data.get("name"),
                "start_distance": mini_loop_data.get("start_distance"),
                "end_distance": mini_loop_data.get("end_distance"),
                "base_lap_time": mini_loop_data.get("base_lap_time")
            })

    # Load overtaking zones
    num_overtaking_zones = track_config.get("num_overtaking_zones", 0)
    for i in range(1, num_overtaking_zones + 1):
        overtaking_zone_key = f"overtakingZone{i}"
        overtaking_zone_data = track_config.get(overtaking_zone_key, {})
        if overtaking_zone_data:  # Only add if overtaking zone data exists
            track_data["overtaking_zones"].append({
                "name": overtaking_zone_data.get("name"),
                "difficulty": overtaking_zone_data.get("difficulty"),
                "distance_from_start": overtaking_zone_data.get("distance_from_start")
            })

    print(f"Loaded track: {track_name}")
    coordinates = plot_miniloops(config)
    track_data["coordinates"] = coordinates
    return track_data

def plot_miniloops(config):
    """Plot the mini loops on the track based on the coordinate file."""
    track_config = config.get("track", {})
    coordinate_file = track_config.get("coordinate_file")
    total_distance = track_config.get("distance", 0)  # Total track distance in km
    mini_loops = track_config.get("mini_loops", {})

    # Load the coordinate file
    coordinates = pd.read_csv(coordinate_file)

    # Ensure the file has the required columns
    if not {"X", "Y"}.issubset(coordinates.columns):
        raise ValueError("The coordinate file must contain 'X' and 'Y' columns.")

    # Calculate the number of points per kilometer
    num_points = len(coordinates)
    points_per_km = num_points / total_distance

    # Add a 'distance' column to the coordinates DataFrame
    coordinates["distance"] = coordinates.index / points_per_km
    if config.get("debugMode", False):

        # Plot the track with mini loops
        plt.figure(figsize=(10, 6))
        for loop_key, mini_loop in mini_loops.items():
            start_distance = mini_loop.get("start_distance", 0)
            end_distance = mini_loop.get("end_distance", 0)
            loop_name = mini_loop.get("name", loop_key)

            # Filter points within the mini loop range
            loop_points = coordinates[
                (coordinates["distance"] >= start_distance) & (coordinates["distance"] <= end_distance)
            ]

            # Plot the mini loop
            plt.plot(loop_points["X"], loop_points["Y"], label=loop_name)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Mini Loops on the Track")
        plt.legend()
        plt.grid()
        plt.show(block=False)
    return coordinates