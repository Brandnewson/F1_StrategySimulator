"""
Generate appendix-ready simulator reference figures.

Outputs:
- report/diagrams/simulator_track_overtaking_zones.png
- report/diagrams/simulator_realtime_snapshot.png

The first figure is a clean simulator-native map of the Spa centreline with the
configured overtaking zones overlaid. The second figure is a static snapshot of
the real-time visualisation rendered from an actual evaluation-state simulator
run using trained policies.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT_DIR = ROOT / "report" / "diagrams"
ZONE_OUT = OUT_DIR / "simulator_track_overtaking_zones.png"
SNAPSHOT_OUT = OUT_DIR / "simulator_realtime_snapshot.png"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helpers.simulatorHelpers import get_distance_to_xy
from simulator import RaceSimulator
from states import init_race_state
from track import load_track


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_driver_name(raw_name: str) -> str:
    mapping = {
        "DQN_team_a_A1_rainbow_lite": "Team A car 1",
        "DQN_team_a_A2_rainbow_lite": "Team A car 2",
        "DQN_team_b_A3_rainbow_lite": "Team B car 1",
        "DQN_team_b_A4_rainbow_lite": "Team B car 2",
        "Base Agent": "Base agent",
    }
    return mapping.get(str(raw_name), str(raw_name))


def make_zone_figure(config: dict, output_path: Path) -> None:
    track = load_track(config)
    race_state = init_race_state(config, track)
    plot_module = race_state._visualise_overtaking_zones(figsize=(12, 8), show=False)
    if plot_module is None:
        raise RuntimeError("Simulator overtaking-zone visualisation could not be generated.")
    plt.gca().set_title("Spa-Francorchamps track and overtaking zones")
    plt.gcf().set_dpi(220)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def make_snapshot_config(base_config: dict) -> dict:
    cfg = copy.deepcopy(base_config)

    cfg["debugMode"] = False
    cfg.setdefault("simulator", {})
    cfg["simulator"]["method"] = "batch"
    cfg["simulator"]["agent_mode"] = "evaluation"
    cfg["simulator"]["runs"] = 1
    cfg.setdefault("race_settings", {})
    cfg["race_settings"]["total_laps"] = int(cfg["race_settings"].get("total_laps", 5) or 5)

    cfg.setdefault("complexity", {})
    cfg["complexity"]["active_profile"] = "low_marl_teams"

    cfg.setdefault("dqn_params", {})
    cfg["dqn_params"]["algo"] = "rainbow_lite"

    cfg.setdefault("marl", {})
    cfg["marl"]["reward_mode"] = "alpha"
    cfg["marl"]["reward_sharing_alpha"] = 0.75
    cfg["marl"]["teams"] = {
        "team_a": {"alpha": 0.75},
        "team_b": {"alpha": 0.75},
    }

    cfg["competitors"] = [
        {
            "name": "DQN_team_a_A1_rainbow_lite",
            "agent": "dqn",
            "colour": "#022050",
            "team_id": "team_a",
        },
        {
            "name": "DQN_team_a_A2_rainbow_lite",
            "agent": "dqn",
            "colour": "#0055AA",
            "team_id": "team_a",
        },
        {
            "name": "DQN_team_b_A3_rainbow_lite",
            "agent": "dqn",
            "colour": "#8B1E3F",
            "team_id": "team_b",
        },
        {
            "name": "DQN_team_b_A4_rainbow_lite",
            "agent": "dqn",
            "colour": "#D1495B",
            "team_id": "team_b",
        },
        {
            "name": "Base Agent",
            "agent": "base",
            "colour": "#F28E2B",
        },
    ]
    return cfg


def advance_to_snapshot_frame(simulator: RaceSimulator, max_frames: int = 450) -> int:
    chosen_frame = 0
    for frame in range(1, max_frames + 1):
        simulator._simulation_step()
        total_events = sum(len(getattr(driver, "decision_events", [])) for driver in simulator.race_state.drivers)
        if total_events >= 4 and simulator.race_state.elapsed_time >= 20.0:
            chosen_frame = frame
            break
        chosen_frame = frame
        if simulator.race_finished:
            break
    return chosen_frame


def render_snapshot(simulator: RaceSimulator, output_path: Path) -> None:
    coords = simulator.track_coords
    if coords is None:
        raise RuntimeError("Simulator track coordinates are unavailable for snapshot rendering.")

    fig, ax = plt.subplots(figsize=(13, 9))

    ax.plot(coords["X"], coords["Y"], color="#888888", linewidth=8, alpha=0.25, zorder=1)
    ax.plot(coords["X"], coords["Y"], color="#333333", linewidth=2.2, zorder=2)

    start_x, start_y = coords.at[0, "X"], coords.at[0, "Y"]
    start_marker = ax.scatter(
        start_x,
        start_y,
        s=220,
        c="#2CA02C",
        marker="s",
        edgecolors="white",
        linewidths=1.0,
        zorder=10,
        label="Start/Finish",
    )

    last_idx = len(coords) - 1
    for index, zone in enumerate(simulator.race_state.overtaking_zones, start=1):
        zone_dist = float(zone.get("distance_from_start", zone.get("distance", 0.0)) or 0.0)
        nearest_idx = (coords["distance"] - zone_dist).abs().idxmin()
        zx = coords.at[nearest_idx, "X"]
        zy = coords.at[nearest_idx, "Y"]

        prev_idx = max(nearest_idx - 1, 0)
        next_idx = min(nearest_idx + 1, last_idx)
        tx = coords.at[next_idx, "X"] - coords.at[prev_idx, "X"]
        ty = coords.at[next_idx, "Y"] - coords.at[prev_idx, "Y"]
        norm = np.hypot(tx, ty) if (tx != 0 or ty != 0) else 1.0
        px, py = -ty / norm, tx / norm

        base_offset = 120 + (index % 3) * 25
        label_x = zx + px * base_offset
        label_y = zy + py * base_offset

        label_text = zone.get("name", f"Zone {index}")

        ax.plot([zx, label_x], [zy, label_y], color="#666666", linewidth=0.8, alpha=0.55, zorder=14)
        ax.text(
            label_x,
            label_y,
            label_text,
            fontsize=8.5,
            zorder=15,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="none"),
        )

    driver_markers = []
    for index, driver in enumerate(simulator.race_state.drivers):
        x, y = get_distance_to_xy(driver.current_distance, simulator.track_coords, simulator.track_distance)
        display_name = format_driver_name(driver.name)
        marker, = ax.plot(
            [x],
            [y],
            "o",
            markersize=12,
            color=simulator.driver_colors[index],
            markeredgecolor="white",
            markeredgewidth=1.8,
            zorder=20,
            label=display_name,
        )
        driver_markers.append(marker)

    leader = min(simulator.race_state.drivers, key=lambda d: d.position)
    info_lines = [
        f"Lap: {leader.completed_laps + 1}/{simulator.total_laps}",
        f"Time: {simulator.race_state.elapsed_time:.1f}s",
        "",
    ]

    for driver in sorted(simulator.race_state.drivers, key=lambda d: d.position):
        if driver == leader:
            gap_str = "Leader"
        else:
            leader_total = leader.completed_laps * simulator.track_distance + leader.current_distance
            driver_total = driver.completed_laps * simulator.track_distance + driver.current_distance
            gap_dist = max(0.0, leader_total - driver_total)
            leader_kms = max(float(getattr(leader, "speed", 0.0)) / 3600.0, 1e-6)
            gap_time = gap_dist / leader_kms
            gap_str = f"+{gap_time:.2f}s"

        info_lines.append(
            f"P{driver.position}: {format_driver_name(driver.name)[:18]:18s} "
            f"Lap {driver.completed_laps + 1} | {gap_str}"
        )

    info_lines.extend(
        [
            "",
            "Mode: real-time visualisation reference",
            "Profile: low_marl_teams evaluation",
        ]
    )

    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="#CCCCCC"),
    )

    ax.set_title("Real-time simulator snapshot on Spa-Francorchamps", fontsize=15, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)

    legend_handles = [start_marker] + driver_markers
    legend_labels = ["Start/finish"] + [format_driver_name(driver.name) for driver in simulator.race_state.drivers]
    ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_realtime_snapshot(base_config: dict, output_path: Path) -> None:
    cfg = make_snapshot_config(base_config)
    track = load_track(cfg)
    race_state = init_race_state(cfg, track)
    simulator = RaceSimulator(race_state, cfg, track)
    frame = advance_to_snapshot_frame(simulator)
    render_snapshot(simulator, output_path)
    print(f"Saved real-time snapshot after {frame} simulation frames.")


def main() -> None:
    config_path = ROOT / "config.json"
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    make_zone_figure(copy.deepcopy(config), ZONE_OUT)
    print(f"Saved overtaking-zone map to {ZONE_OUT}")

    make_realtime_snapshot(copy.deepcopy(config), SNAPSHOT_OUT)
    print(f"Saved simulator snapshot to {SNAPSHOT_OUT}")


if __name__ == "__main__":
    main()
