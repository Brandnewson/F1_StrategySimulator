"""
Generate the cooperative-advantage cliff figure for Phases 5-7.

The plotted value is the percentage change at alpha=0.75 relative to the
competitive alpha=0.0 baseline within each phase:

- N=3: Phase 5 (2 DQN + 1 Base), using joint_dqn_beat_base_rate
- N=4: Phase 7A (3 DQN + 1 Base), using joint_dqn_beat_base_rate
- N=5: Phase 6 (4 DQN + 1 Base), using the average of team_a/team_b
       both_beat_base_rate because the team game has two symmetric teams
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
        "scripts/generate_phase5to7_cooperative_advantage_cliff_figure.py`."
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "report" / "diagrams" / "phase5to7_cooperative_advantage_cliff.png"

STOCH_LEVELS = ["s0", "s1", "s2"]
SEEDS = [101, 202, 303]

CONDITIONS = [
    {
        "agent_count": 3,
        "phase_name": "Phase 5",
        "baseline": ROOT / "metrics" / "phase5",
        "coop": ROOT / "metrics" / "phase5",
        "baseline_stem": "rainbow_a000_{stoch}_s{seed}.json",
        "coop_stem": "rainbow_a075_{stoch}_s{seed}.json",
        "metric_mode": "joint",
    },
    {
        "agent_count": 4,
        "phase_name": "Phase 7A",
        "baseline": ROOT / "metrics" / "phase7a",
        "coop": ROOT / "metrics" / "phase7a",
        "baseline_stem": "3dqn_a000_{stoch}_s{seed}.json",
        "coop_stem": "3dqn_a075_{stoch}_s{seed}.json",
        "metric_mode": "joint",
    },
    {
        "agent_count": 5,
        "phase_name": "Phase 6",
        "baseline": ROOT / "metrics" / "phase6",
        "coop": ROOT / "metrics" / "phase6",
        "baseline_stem": "teams_a000_b000_{stoch}_s{seed}.json",
        "coop_stem": "teams_a075_b075_{stoch}_s{seed}.json",
        "metric_mode": "team_avg",
    },
]


def load_metric(path: Path, metric_mode: str) -> float:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metrics = payload["metrics"]
    vs_base_metrics = payload.get("vs_base_metrics") or {}
    if metric_mode == "joint":
        if "joint_dqn_beat_base_rate" in vs_base_metrics:
            return float(vs_base_metrics["joint_dqn_beat_base_rate"]["mean"])
        if "joint_beat_base_rate" in vs_base_metrics:
            return float(vs_base_metrics["joint_beat_base_rate"]["mean"])
        if "joint_dqn_beat_base_rate" in metrics:
            return float(metrics["joint_dqn_beat_base_rate"]["mean"])
        if "joint_beat_base_rate" in metrics:
            return float(metrics["joint_beat_base_rate"]["mean"])
        team_metrics = payload.get("team_metrics") or {}
        if "team_a_both_beat_base_rate" in team_metrics and "team_b_both_beat_base_rate" in team_metrics:
            team_a = float(team_metrics["team_a_both_beat_base_rate"]["mean"])
            team_b = float(team_metrics["team_b_both_beat_base_rate"]["mean"])
            return (team_a + team_b) / 2.0
        raise KeyError(f"No joint beat-base metric found in {path}")
    if metric_mode == "team_avg":
        team_metrics = payload.get("team_metrics") or {}
        if not team_metrics:
            if "joint_dqn_beat_base_rate" in vs_base_metrics:
                return float(vs_base_metrics["joint_dqn_beat_base_rate"]["mean"])
            if "joint_dqn_beat_base_rate" in metrics:
                return float(metrics["joint_dqn_beat_base_rate"]["mean"])
            raise KeyError(f"No team metrics found in {path}")
        team_a = float(team_metrics["team_a_both_beat_base_rate"]["mean"])
        team_b = float(team_metrics["team_b_both_beat_base_rate"]["mean"])
        return (team_a + team_b) / 2.0
    raise ValueError(f"Unsupported metric mode: {metric_mode}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def collect_condition_summary(condition: dict) -> dict:
    baseline_vals = []
    coop_vals = []

    for stoch in STOCH_LEVELS:
        for seed in SEEDS:
            baseline_path = condition["baseline"] / condition["baseline_stem"].format(stoch=stoch, seed=seed)
            coop_path = condition["coop"] / condition["coop_stem"].format(stoch=stoch, seed=seed)
            baseline_vals.append(load_metric(baseline_path, condition["metric_mode"]))
            coop_vals.append(load_metric(coop_path, condition["metric_mode"]))

    baseline_mean = mean(baseline_vals)
    coop_mean = mean(coop_vals)
    pct_change = 100.0 * (coop_mean - baseline_mean) / baseline_mean

    return {
        "agent_count": condition["agent_count"],
        "phase_name": condition["phase_name"],
        "baseline_mean": baseline_mean,
        "coop_mean": coop_mean,
        "pct_change": pct_change,
    }


def style_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def build_figure(summary: list[dict]) -> None:
    style_plot()

    x = np.array([item["agent_count"] for item in summary], dtype=float)
    y = np.array([item["pct_change"] for item in summary], dtype=float)
    colors = ["#2E8B57", "#7A7A7A", "#C0392B"]

    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f7f8")

    bars = ax.bar(
        x,
        y,
        width=0.55,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    ax.axhline(
        0,
        color="#4B5563",
        linestyle="--",
        linewidth=1.3,
        zorder=2,
    )
    ax.text(
        4.62,
        2.1,
        "Competitive baseline ($\\alpha=0.0$)",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#4B5563",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.6},
    )

    ax.axvline(
        3.5,
        color="#6B7280",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.text(
        3.53,
        86,
        "Scaling cliff",
        rotation=90,
        ha="left",
        va="top",
        fontsize=10,
        color="#6B7280",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.6},
    )

    for bar, item in zip(bars, summary):
        value = item["pct_change"]
        label = f"{value:+.0f}%"
        if value >= 0:
            text_y = value + 3.0
            va = "bottom"
        else:
            text_y = value - 4.0
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            text_y,
            label,
            ha="center",
            va=va,
            fontsize=12,
            fontweight="bold",
            color=bar.get_facecolor(),
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            -43,
            item["phase_name"],
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#374151",
        )

    ax.set_xticks(x)
    ax.set_xlim(2.45, 5.55)
    ax.set_ylim(-45, 95)
    ax.set_yticks(np.arange(-40, 81, 20))

    ax.set_xlabel("Total agents (DQN agents + 1 fixed Base adversary)")
    ax.set_ylabel("Cooperative advantage at $\\alpha=0.75$ (%)")
    ax.set_title("Phases 5-7: Cooperative advantage collapses at agent count $\\geq 4$", pad=14)

    ax.grid(axis="y", color="#D1D5DB", linewidth=0.9)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = [collect_condition_summary(condition) for condition in CONDITIONS]

    print("Cooperative advantage summary from repo metrics:")
    for item in summary:
        print(
            f"  N={item['agent_count']} ({item['phase_name']}): "
            f"baseline={item['baseline_mean']:.3f}, "
            f"alpha0.75={item['coop_mean']:.3f}, "
            f"change={item['pct_change']:+.1f}%"
        )

    build_figure(summary)
    print(f"Figure written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
