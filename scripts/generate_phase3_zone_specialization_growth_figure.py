"""
Generate the Phase 3 zone-specialisation growth figure from repo metrics.

Available data in the repo:
- 500-episode rainbow-lite s2 trials for seeds 101, 202, 303, 404, 505
- 750-episode rainbow-lite s2 reruns for seeds 101, 202, 303

The figure therefore shows:
- connected trajectories for the three paired seeds
- 500-only markers for seeds 404 and 505
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing plotting dependencies. Run this script inside the "
        "'f1StrategySim' conda environment, for example: "
        "`conda run -n f1StrategySim python "
        "scripts/generate_phase3_zone_specialization_growth_figure.py`."
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
PHASE3_DIR = ROOT / "metrics" / "phase3"
OUT_PATH = ROOT / "report" / "diagrams" / "phase3_zone_specialization_growth.png"

DATA_FILES = {
    500: {
        101: "rainbow_marl_s2_s101.json",
        202: "rainbow_marl_s2_s202.json",
        303: "rainbow_marl_s2_s303.json",
        404: "rainbow_marl_s2_s404.json",
        505: "rainbow_marl_s2_s505.json",
    },
    750: {
        101: "rainbow_marl_750_s2_s101.json",
        202: "rainbow_marl_750_s2_s202.json",
        303: "rainbow_marl_750_s2_s303.json",
    },
}

# Fallback values from research_findings/phase3_full_analysis.md for repo states
# where the original metrics/phase3 JSONs are not checked in.
FALLBACK_ZONE_DIFF = {
    500: {
        101: 0.022,
        202: 0.188,
        303: 0.063,
        404: 0.043,
        505: 0.520,
    },
    750: {
        101: 0.163,
        202: 0.405,
        303: 0.048,
    },
}

SEED_COLORS = {
    101: "#4C78A8",
    202: "#8B1E3F",
    303: "#54A24B",
    404: "#9C755F",
    505: "#ECA82C",
}


def load_zone_diff(path: Path, episodes: int, seed: int) -> float:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return float(payload["metrics"]["strategy_differentiation"]["zone_differentiation_index"])

    return float(FALLBACK_ZONE_DIFF[episodes][seed])


def collect_data() -> dict[int, dict[int, float]]:
    values: dict[int, dict[int, float]] = {}
    for episodes, mapping in DATA_FILES.items():
        values[episodes] = {}
        for seed, filename in mapping.items():
            values[episodes][seed] = load_zone_diff(PHASE3_DIR / filename, episodes, seed)
    return values


def style_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def build_figure(values: dict[int, dict[int, float]]) -> None:
    style_plot()

    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f7f8")

    paired_seeds = sorted(set(values[500]).intersection(values[750]))
    all_500_seeds = sorted(values[500])
    unpaired_500_seeds = [seed for seed in all_500_seeds if seed not in paired_seeds]

    for seed in paired_seeds:
        x = [500, 750]
        y = [values[500][seed], values[750][seed]]
        is_highlight = seed == 202
        ax.plot(
            x,
            y,
            color=SEED_COLORS[seed],
            linewidth=3.2 if is_highlight else 1.9,
            linestyle="-" if is_highlight else "--",
            marker="o",
            markersize=8 if is_highlight else 6.5,
            markerfacecolor="white",
            markeredgewidth=2 if is_highlight else 1.5,
            alpha=0.98,
            label=f"Seed {seed}" + (" (highlighted)" if is_highlight else ""),
            zorder=4 if is_highlight else 3,
        )

    for seed in unpaired_500_seeds:
        ax.scatter(
            [500],
            [values[500][seed]],
            s=70,
            color=SEED_COLORS[seed],
            marker="D",
            edgecolor="white",
            linewidth=0.9,
            label=f"Seed {seed} (500 only)",
            zorder=5,
        )

    mean_x = np.array([500, 750], dtype=float)
    mean_y = np.array(
        [
            np.mean([values[500][seed] for seed in paired_seeds]),
            np.mean([values[750][seed] for seed in paired_seeds]),
        ]
    )
    std_y = np.array(
        [
            np.std([values[500][seed] for seed in paired_seeds]),
            np.std([values[750][seed] for seed in paired_seeds]),
        ]
    )
    ax.errorbar(
        mean_x,
        mean_y,
        yerr=std_y,
        color="#374151",
        linewidth=1.6,
        linestyle=":",
        capsize=4,
        marker="s",
        markersize=5.5,
        label="Mean +/- SD (paired seeds)",
        zorder=2,
    )

    ax.annotate(
        "Seed 202\ncanonical example",
        xy=(750, values[750][202]),
        xytext=(705, 0.48),
        fontsize=10,
        color=SEED_COLORS[202],
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": SEED_COLORS[202],
            "linewidth": 1.3,
            "shrinkA": 4,
            "shrinkB": 4,
        },
        bbox={"facecolor": "white", "edgecolor": SEED_COLORS[202], "alpha": 0.95, "pad": 2.5},
    )

    ax.text(
        0.015,
        0.965,
        "Index = 0: both agents favour the same zones\nIndex = 1: complete zone partition",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.96, "pad": 2.8},
    )

    ax.text(
        0.985,
        0.06,
        "Only seeds 101/202/303 have 750-episode reruns",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4b5563",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.96, "pad": 2.5},
    )

    ax.set_title("Phase 3: Zone specialisation emerges over training duration", pad=14)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Zone differentiation index")

    ax.set_xlim(470, 780)
    ax.set_ylim(0.0, 0.62)
    ax.set_xticks([500, 750])
    ax.set_yticks(np.arange(0.0, 0.61, 0.1))

    ax.grid(axis="y", color="#d1d5db", linewidth=0.9)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    values = collect_data()
    print("Phase 3 zone differentiation values (rainbow-lite, s2):")
    for episodes in sorted(values):
        for seed in sorted(values[episodes]):
            print(f"  {episodes} episodes, seed {seed}: {values[episodes][seed]:.3f}")
    build_figure(values)
    print(f"Figure written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
