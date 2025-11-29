import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from typing import Dict, List, Tuple


def build_mini_loops(track_config: Dict) -> List[Dict]:
    """Normalize mini-loop config and compute loop speeds (km/s)."""
    mini_loops_cfg = track_config.get("mini_loops", {})
    if isinstance(mini_loops_cfg, dict):
        loops = list(mini_loops_cfg.values())
    else:
        loops = mini_loops_cfg

    processed = []
    for loop in loops:
        start = loop.get("start_distance", 0.0)
        end = loop.get("end_distance", 0.0)
        base_time = loop.get("base_lap_time", 10.0)
        distance = end - start
        speed = distance / base_time if base_time > 0 else 0.01
        processed.append({
            "start": float(start),
            "end": float(end),
            "name": loop.get("name", "Unknown"),
            "base_time": float(base_time),
            "speed": float(speed)  # km/s
        })
    processed.sort(key=lambda x: x["start"])
    return processed


def _calibrate_speed_profile(distances: np.ndarray, speeds: np.ndarray, mini_loops: List[Dict]) -> np.ndarray:
    """
    Calibrate the speed profile so total lap time matches sum of base_lap_times.
    
    This adjusts the overall speed scale while preserving the shape of the profile.
    
    Args:
        distances: Array of distance sample points
        speeds: Array of uncalibrated speed values
        
    Returns:
        Calibrated speed array
    """
    expected_total_time = sum(loop["base_time"] for loop in mini_loops)
    dt = np.diff(distances)
    avg_speeds = (speeds[:-1] + speeds[1:]) / 2
    actual_time = np.sum(dt / avg_speeds) if avg_speeds.size > 0 else expected_total_time
    scale_factor = (actual_time / expected_total_time) if actual_time > 0 else 1.0
    calibrated = speeds * scale_factor
    return calibrated


def build_speed_profile(mini_loops: List[Dict], track_distance: float, speed_multiplier: float = 1.0) -> Dict:
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
    if not mini_loops:
        return {
            "spline": None,
            "distances": np.array([0.0, track_distance]),
            "speeds": np.array([0.05, 0.05]),
            "multiplier": speed_multiplier
        }

    anchor_dists = []
    anchor_speeds = []
    for loop in mini_loops:
        midpoint = (loop["start"] + loop["end"]) / 2.0
        anchor_dists.append(midpoint)
        anchor_speeds.append(loop["speed"] * speed_multiplier)

    # wrap-around anchors for smooth periodic spline
    last_mid = (mini_loops[-1]["start"] + mini_loops[-1]["end"]) / 2.0
    first_mid = (mini_loops[0]["start"] + mini_loops[0]["end"]) / 2.0
    pre_start = -(track_distance - last_mid)
    post_end = track_distance + first_mid

    ext_dists = [pre_start] + anchor_dists + [post_end]
    ext_speeds = [anchor_speeds[-1]] + anchor_speeds + [anchor_speeds[0]]

    spline = CubicSpline(ext_dists, ext_speeds, bc_type='natural')

    num_samples = max(100, int(track_distance * 100))
    sample_distances = np.linspace(0.0, track_distance, num_samples)
    sample_speeds = spline(sample_distances)
    sample_speeds = np.maximum(sample_speeds, 0.01)

    calibrated = _calibrate_speed_profile(sample_distances, sample_speeds, mini_loops)
    calibrated_spline = CubicSpline(sample_distances, calibrated, bc_type='natural')

    return {
        "spline": calibrated_spline,
        "distances": sample_distances,
        "speeds": calibrated,
        "multiplier": speed_multiplier
    }


def plot_speed_profile(speed_profile: Dict, mini_loops: List[Dict], show_loops: bool = True) -> None:
    """Visual debug: smooth profile vs original loops and cumulative lap time."""
    distances = speed_profile["distances"]
    speeds_kms = speed_profile["speeds"]
    speeds_kmh = speeds_kms * 3600.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(distances, speeds_kmh, 'b-', linewidth=2, label='Smooth Speed Profile')

    if show_loops:
        for loop in mini_loops:
            loop_speed_kmh = loop["speed"] * 3600.0
            ax1.hlines(loop_speed_kmh, loop["start"], loop["end"],
                       colors='r', linestyles='--', linewidth=1.5, alpha=0.7)
            ax1.axvline(loop["start"], color='gray', linestyle=':', alpha=0.5)
            mid = (loop["start"] + loop["end"]) / 2.0
            ax1.text(mid, loop_speed_kmh + 5, loop["name"], ha='center', fontsize=8, rotation=45)

    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title('Smooth Speed Profile vs Original Loop Speeds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    dt = np.diff(distances)
    avg_speeds = (speeds_kms[:-1] + speeds_kms[1:]) / 2.0
    segment_times = dt / avg_speeds
    cumulative_time = np.concatenate([[0.0], np.cumsum(segment_times)])

    ax2.plot(distances, cumulative_time, 'g-', linewidth=2)
    ax2.set_xlabel('Track Distance (km)')
    ax2.set_ylabel('Cumulative Time (s)')
    ax2.set_title(f'Cumulative Lap Time (Total: {cumulative_time[-1]:.2f}s)')
    ax2.grid(True, alpha=0.3)

    for loop in mini_loops:
        ax2.axvline(loop["start"], color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()


def get_speed_at_distance(distance: float, speed_profile: Dict, mini_loops: List[Dict], track_distance: float) -> float:
    """Return speed (km/s) at given distance using profile; fallback to loops if no spline."""
    if track_distance <= 0:
        return 0.01
    wrapped = distance % track_distance
    spline = speed_profile.get("spline")
    if spline is not None:
        v = float(spline(wrapped))
        return max(v, 0.01)
    # fallback
    for loop in mini_loops:
        if loop["start"] <= wrapped < loop["end"]:
            return loop["speed"]
    return mini_loops[-1]["speed"] if mini_loops else 0.05


def get_distance_to_xy(distance: float, track_coords, track_distance: float) -> Tuple[float, float]:
    """Map distance (km) to X,Y using track_coords (pandas-like DataFrame with 'distance','X','Y')."""
    if track_coords is None or track_distance <= 0:
        return (0.0, 0.0)
    wrapped = distance % track_distance
    nearest_idx = (track_coords["distance"] - wrapped).abs().idxmin()
    return (track_coords.at[nearest_idx, "X"], track_coords.at[nearest_idx, "Y"])