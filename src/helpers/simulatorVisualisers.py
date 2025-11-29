import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Any

from helpers.simulatorHelpers import get_distance_to_xy

def run_visualisation(simulator: Any) -> None:
    """Run visualization for the given RaceSimulator instance."""
    # Short aliases
    self = simulator

    print(f"\n{'='*60}")
    print(f"Starting Race: {self.total_laps} laps at {self.config['track']['name'].upper()}")
    print(f"{'='*60}\n")
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot track
    coords = self.track_coords
    ax.plot(coords["X"], coords["Y"], color="#888888", linewidth=8, alpha=0.3, zorder=1, label="_nolegend_")
    ax.plot(coords["X"], coords["Y"], color="#333333", linewidth=2, zorder=2, label="_nolegend_")

    # Start/finish marker
    start_x, start_y = coords.at[0, "X"], coords.at[0, "Y"]
    start_marker = ax.scatter(start_x, start_y, s=200, c="green", marker="s",
                              zorder=10, label="Start/Finish")

    # Overtaking zone side labels (off-track)
    if self.track_coords is not None and getattr(self.race_state, "overtaking_zones", None):
        coords = self.track_coords
        last_idx = len(coords) - 1
        for i, zone in enumerate(self.race_state.overtaking_zones, start=1):
            try:
                zone_dist = float(zone.get("distance_from_start", zone.get("distance", 0)))
            except Exception:
                zone_dist = 0.0
            nearest_idx = (coords["distance"] - zone_dist).abs().idxmin()
            zx = coords.at[nearest_idx, "X"]
            zy = coords.at[nearest_idx, "Y"]

            prev_idx = max(nearest_idx - 1, 0)
            next_idx = min(nearest_idx + 1, last_idx)
            tx = coords.at[next_idx, "X"] - coords.at[prev_idx, "X"]
            ty = coords.at[next_idx, "Y"] - coords.at[prev_idx, "Y"]
            norm = np.hypot(tx, ty) if (tx != 0 or ty != 0) else 1.0
            px, py = -ty / norm, tx / norm

            base_offset_m = 120
            stagger = (i % 3) * 25
            offset_m = base_offset_m + stagger

            label_x = zx + px * offset_m
            label_y = zy + py * offset_m

            zone_name = zone.get("name", f"Zone {i}")
            difficulty = zone.get("difficulty", None)
            label_text = f"{zone_name} ({difficulty:.2f})" if difficulty is not None else zone_name

            ax.plot([zx, label_x], [zy, label_y], color="#666666", linewidth=0.8, alpha=0.6, zorder=14)
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

    legend_handles = [start_marker] + driver_dots
    legend_labels = ["Start/Finish"] + [d.name for d in self.race_state.drivers]
    ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    def init():
        for dot in driver_dots:
            dot.set_data([], [])
        info_text.set_text("")
        return driver_dots + [info_text]

    def animate(frame):
        # Run simulation step
        self._simulation_step()

        # Update positions
        for i, driver in enumerate(self.race_state.drivers):
            x, y = get_distance_to_xy(driver.current_distance, self.track_coords, self.track_distance)
            driver_dots[i].set_data([x], [y])

        # Build info lines
        info_lines = [f"Lap: {self.race_state.drivers[0].completed_laps + 1}/{self.total_laps}",
                      f"Time: {self.race_state.elapsed_time:.1f}s", ""]

        leader = min(self.race_state.drivers, key=lambda d: d.position)
        for driver in sorted(self.race_state.drivers, key=lambda d: d.position):
            if driver == leader:
                gap_str = "Leader"
            else:
                leader_total = leader.completed_laps * self.track_distance + leader.current_distance
                driver_total = driver.completed_laps * self.track_distance + driver.current_distance
                gap_dist = leader_total - driver_total

                leader_kms = max(leader.speed / 3600.0, 1e-6)
                driver_kms = max(driver.speed / 3600.0, 1e-6)

                if driver_kms - leader_kms > 1e-6:
                    gap_time = gap_dist / leader_kms
                else:
                    gap_time = gap_dist / leader_kms

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

    # Estimate frames and run animation
    estimated_race_time = self.total_laps * self.race_state.base_lap_time
    estimated_frames = int(estimated_race_time / (self.ticks_per_frame * self.tick_duration)) + 10000

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=estimated_frames,
                        interval=self.animation_interval,
                        blit=True, repeat=False,
                        cache_frame_data=False)

    def on_close(event):
        self.race_finished = True

    fig.canvas.mpl_connect('close_event', on_close)

    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("Animation interrupted by user (KeyboardInterrupt). Closing.")

    # Print final results
    self._print_results()