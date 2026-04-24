"""
Generate the Phase 4 collapse-rate figure directly from repo metrics.

This reads the Phase 3 rainbow-lite baseline plus the Phase 4 alpha-sweep
trial outputs and applies the same degeneracy rule used elsewhere in the
repo: a trial is counted as collapsed if A1 win rate is <= 0.15 or >= 0.85.
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
        "`conda run -n f1StrategySim python scripts/generate_phase4_collapse_figure.py`."
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
PHASE3_DIR = ROOT / "metrics" / "phase3"
PHASE4_DIR = ROOT / "metrics" / "phase4"
OUT_PATH = ROOT / "report" / "diagrams" / "phase4_collapse_vs_alpha.png"

STOCH_LEVELS = ["s0", "s1", "s2"]
SEEDS = [101, 202, 303]
PLOT_ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]

PHASE3_BASELINE_STEMS = {
    "s0": ["rainbow_marl_s0_s101", "rainbow_marl_s0_s202", "rainbow_marl_s0_s303"],
    "s1": ["rainbow_marl_s1_s101", "rainbow_marl_s1_s202", "rainbow_marl_s1_s303"],
    "s2": ["rainbow_marl_s2_s101", "rainbow_marl_s2_s202", "rainbow_marl_s2_s303"],
}

PHASE4_LABELS = {
    0.25: "025",
    0.50: "05",
    0.75: "075",
    1.00: "10",
}


def is_degenerate(win_rate: float) -> bool:
    return win_rate <= 0.15 or win_rate >= 0.85


def load_win_rate(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metrics = payload.get("metrics", {})
    wr = metrics.get("win_rate_a1_vs_a2", {}).get("mean")
    if wr is None:
        wr = payload.get("objective_score")
    if wr is None:
        raise ValueError(f"Could not find A1 win rate in {path}")
    return float(wr)


def collect_phase4_trials() -> dict[float, list[float]]:
    trials_by_alpha: dict[float, list[float]] = {alpha: [] for alpha in PLOT_ALPHAS}

    for stoch in STOCH_LEVELS:
        for stem in PHASE3_BASELINE_STEMS[stoch]:
            trials_by_alpha[0.00].append(load_win_rate(PHASE3_DIR / f"{stem}.json"))

    for alpha, label in PHASE4_LABELS.items():
        for stoch in STOCH_LEVELS:
            for seed in SEEDS:
                stem = f"rainbow_a{label}_{stoch}_s{seed}"
                trials_by_alpha[alpha].append(load_win_rate(PHASE4_DIR / f"{stem}.json"))

    for alpha, trials in trials_by_alpha.items():
        if len(trials) != 9:
            raise ValueError(f"Expected 9 trials for alpha={alpha:.2f}, found {len(trials)}")

    return trials_by_alpha


def collapse_summary(trials_by_alpha: dict[float, list[float]]) -> tuple[list[int], list[float]]:
    counts = []
    rates = []
    for alpha in PLOT_ALPHAS:
        degenerate_count = sum(1 for win_rate in trials_by_alpha[alpha] if is_degenerate(win_rate))
        counts.append(degenerate_count)
        rates.append(100.0 * degenerate_count / len(trials_by_alpha[alpha]))
    return counts, rates


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


def build_figure(collapse_counts: list[int], collapse_rates: list[float]) -> None:
    style_plot()

    fig, ax = plt.subplots(figsize=(9.25, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f3f4f6")

    line_color = "#173f73"
    threshold_color = "#6b7280"
    accent_color = "#8b1e3f"

    ax.plot(
        PLOT_ALPHAS,
        collapse_rates,
        color=line_color,
        linewidth=2.6,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor=line_color,
        markeredgewidth=2,
        zorder=3,
    )

    ax.axvline(
        0.625,
        linestyle="--",
        linewidth=1.4,
        color=threshold_color,
        zorder=1,
    )
    ax.text(
        0.635,
        98,
        "Destabilization threshold",
        rotation=90,
        va="top",
        ha="left",
        fontsize=10,
        color=threshold_color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8},
    )

    for alpha, rate, count in zip(PLOT_ALPHAS, collapse_rates, collapse_counts):
        offset_y = -16 if rate >= 98 else 8
        ax.annotate(
            f"{int(round(rate))}%",
            xy=(alpha, rate),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=line_color,
        )
        ax.annotate(
            f"({count}/9)",
            xy=(alpha, rate),
            xytext=(0, offset_y - 12 if rate >= 98 else offset_y + 12),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#4b5563",
        )

    ax.text(
        0.985,
        0.05,
        "N = 9 per point",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95, "pad": 2.5},
    )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 105)
    ax.set_xticks(PLOT_ALPHAS)
    ax.set_xticklabels([f"{alpha:.2f}" for alpha in PLOT_ALPHAS])
    ax.set_yticks(np.arange(0, 101, 20))

    ax.set_xlabel("Reward mixing coefficient ($\\alpha$)")
    ax.set_ylabel("Collapse Rate (%)")
    ax.set_title("Phase 4: Sharp destabilization threshold in zero-sum two-agent game", pad=14)

    ax.grid(axis="y", color="#d1d5db", linewidth=0.9)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    trials_by_alpha = collect_phase4_trials()
    collapse_counts, collapse_rates = collapse_summary(trials_by_alpha)

    print("Phase 4 collapse summary from repo metrics:")
    for alpha, count, rate in zip(PLOT_ALPHAS, collapse_counts, collapse_rates):
        print(f"  alpha={alpha:.2f}: {count}/9 collapsed ({rate:.0f}%)")

    build_figure(collapse_counts, collapse_rates)
    print(f"Figure written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
