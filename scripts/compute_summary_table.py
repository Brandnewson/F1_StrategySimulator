"""
compute_summary_table.py
------------------------
Computes cross-algorithm summary tables for Phase 2 (single-agent) and
Phase 3 (MARL) from the metrics JSON files.

Outputs:
  - metrics/summary_tables.json  (machine-readable)
  - metrics/summary_tables.md    (human-readable Markdown)

Run from the project root:
  conda run -n f1StrategySim python scripts/compute_summary_table.py
"""

import json
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
PHASE2_DIR = ROOT / "metrics" / "phase2"
PHASE3_DIR = ROOT / "metrics" / "phase3"
PHASE4_DIR = ROOT / "metrics" / "phase4"
OUT_JSON   = ROOT / "metrics" / "summary_tables.json"
OUT_MD     = ROOT / "metrics" / "summary_tables.md"

# ── helpers ──────────────────────────────────────────────────────────────────

def ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    """Wilson-approximated 95 % CI for a proportion; fall back to normal CI."""
    if n == 0:
        return (0.0, 0.0)
    se = std / math.sqrt(n)
    z  = 1.96
    return (round(mean - z * se, 4), round(mean + z * se, 4))

def mean_std_n(values: list[float]) -> dict:
    n   = len(values)
    mu  = sum(values) / n if n else 0.0
    var = sum((v - mu) ** 2 for v in values) / n if n else 0.0
    sd  = math.sqrt(var)
    lo, hi = ci95(mu, sd, n)
    return {"mean": round(mu, 4), "std": round(sd, 4), "n": n,
            "ci95_low": lo, "ci95_high": hi}

def is_degenerate(wr: float) -> bool:
    """WR ≤ 0.15 or ≥ 0.85 counts as a degenerate collapse."""
    return wr <= 0.15 or wr >= 0.85

# ── Phase 2 loading ───────────────────────────────────────────────────────────

PHASE2_ALGOS = {
    "vanilla":      "vanilla",
    "double":       "double",
    "dueling":      "dueling",
    "rainbow_lite": "rainbow",
}
STOCH_LEVELS = ["s0", "s1", "s2"]
SEEDS        = [101, 202, 303]

def load_phase2() -> dict:
    """Returns {algo: {stoch: [win_rates]}}."""
    data = defaultdict(lambda: defaultdict(list))
    for algo_key, file_prefix in PHASE2_ALGOS.items():
        for stoch in STOCH_LEVELS:
            for seed in SEEDS:
                fpath = PHASE2_DIR / f"{file_prefix}_{stoch}_s{seed}.json"
                if not fpath.exists():
                    print(f"  [phase2] missing: {fpath.name}")
                    continue
                with fpath.open() as fh:
                    d = json.load(fh)
                wr = d.get("metrics", {}).get("win_rate_vs_baseline", {}).get("mean", None)
                if wr is not None:
                    data[algo_key][stoch].append(wr)
    return data

# ── Phase 3 loading ───────────────────────────────────────────────────────────

# Only include the core 500-episode trials for the fair cross-algorithm comparison.
PHASE3_FILES = {
    "vanilla": {
        "s0": ["vanilla_marl_s0_s101", "vanilla_marl_s0_s202", "vanilla_marl_s0_s303"],
        "s1": ["vanilla_marl_s1_s101", "vanilla_marl_s1_s202", "vanilla_marl_s1_s303"],
        "s2": ["vanilla_marl_s2_s101", "vanilla_marl_s2_s202", "vanilla_marl_s2_s303"],
    },
    "double": {
        "s0": ["double_marl_500_s0_s101", "double_marl_500_s0_s202", "double_marl_500_s0_s303"],
        "s1": ["double_marl_500_s1_s101", "double_marl_500_s1_s202", "double_marl_500_s1_s303"],
        "s2": ["double_marl_500_s2_s101", "double_marl_500_s2_s202", "double_marl_500_s2_s303"],
    },
    "dueling": {
        "s0": ["dueling_marl_500_s0_s101", "dueling_marl_500_s0_s202", "dueling_marl_500_s0_s303"],
        "s1": ["dueling_marl_500_s1_s101", "dueling_marl_500_s1_s202", "dueling_marl_500_s1_s303"],
        "s2": ["dueling_marl_500_s2_s101", "dueling_marl_500_s2_s202", "dueling_marl_500_s2_s303"],
    },
    "rainbow_lite": {
        "s0": ["rainbow_marl_s0_s101", "rainbow_marl_s0_s202", "rainbow_marl_s0_s303"],
        "s1": ["rainbow_marl_s1_s101", "rainbow_marl_s1_s202", "rainbow_marl_s1_s303"],
        "s2": ["rainbow_marl_s2_s101", "rainbow_marl_s2_s202", "rainbow_marl_s2_s303"],
    },
}

def load_phase3() -> dict:
    """Returns {algo: {stoch: [{"wr": float, "zone_diff": float, "risk_diff": float}]}}."""
    data = defaultdict(lambda: defaultdict(list))
    for algo, stoch_map in PHASE3_FILES.items():
        for stoch, file_stems in stoch_map.items():
            for stem in file_stems:
                fpath = PHASE3_DIR / f"{stem}.json"
                if not fpath.exists():
                    print(f"  [phase3] missing: {fpath.name}")
                    continue
                with fpath.open() as fh:
                    d = json.load(fh)
                metrics = d.get("metrics", {})
                # Handle both old and new MARL JSON schemas
                wr_block = metrics.get("win_rate_a1_vs_a2", {})
                wr = wr_block.get("mean", None)
                if wr is None:
                    wr = d.get("objective_score", None)
                strat = metrics.get("strategy_differentiation", {})
                zone_diff = strat.get("zone_differentiation_index", None)
                risk_diff = strat.get("risk_differentiation_index", None)
                if wr is not None:
                    data[algo][stoch].append({
                        "wr":        round(wr, 4),
                        "zone_diff": round(zone_diff, 4) if zone_diff is not None else None,
                        "risk_diff": round(risk_diff, 4) if risk_diff is not None else None,
                        "degenerate": is_degenerate(wr),
                        "stem":      stem,
                    })
    return data

# ── Phase 4 loading ───────────────────────────────────────────────────────────

PHASE4_ALPHAS = ["0.25", "0.50", "0.75", "1.0"]

def _phase4_stem(alpha_str: str, stoch: str, seed: int) -> str:
    # Mapping matches the output filenames used in the run commands:
    # 0.25 -> a025, 0.50 -> a05, 0.75 -> a075, 1.0 -> a10
    label_map = {"0.25": "025", "0.50": "05", "0.75": "075", "1.0": "10"}
    label = label_map.get(alpha_str, alpha_str.replace(".", ""))
    return f"rainbow_a{label}_{stoch}_s{seed}"

def load_phase4() -> dict:
    """Returns {alpha_str: {stoch: [{"wr": float, "zone_diff": float, "risk_diff": float}]}}."""
    data = defaultdict(lambda: defaultdict(list))
    for alpha_str in PHASE4_ALPHAS:
        for stoch in STOCH_LEVELS:
            for seed in SEEDS:
                stem = _phase4_stem(alpha_str, stoch, seed)
                fpath = PHASE4_DIR / f"{stem}.json"
                if not fpath.exists():
                    print(f"  [phase4] missing: {fpath.name}")
                    continue
                with fpath.open() as fh:
                    d = json.load(fh)
                metrics = d.get("metrics", {})
                wr_block = metrics.get("win_rate_a1_vs_a2", {})
                wr = wr_block.get("mean", None)
                if wr is None:
                    wr = d.get("objective_score", None)
                strat = metrics.get("strategy_differentiation", {})
                zone_diff = strat.get("zone_differentiation_index", None)
                risk_diff = strat.get("risk_differentiation_index", None)
                if wr is not None:
                    data[alpha_str][stoch].append({
                        "wr":        round(wr, 4),
                        "zone_diff": round(zone_diff, 4) if zone_diff is not None else None,
                        "risk_diff": round(risk_diff, 4) if risk_diff is not None else None,
                        "degenerate": is_degenerate(wr),
                        "stem":      stem,
                    })
    return data

def summarise_phase4(data: dict) -> dict:
    summary = {}
    for alpha_str in PHASE4_ALPHAS:
        summary[alpha_str] = {}
        for stoch in STOCH_LEVELS:
            trials = data[alpha_str][stoch]
            if not trials:
                summary[alpha_str][stoch] = None
                continue
            wrs       = [t["wr"] for t in trials]
            zdiffs    = [t["zone_diff"] for t in trials if t["zone_diff"] is not None]
            rdiffs    = [t["risk_diff"] for t in trials if t["risk_diff"] is not None]
            deg_count = sum(1 for t in trials if t["degenerate"])
            stats = mean_std_n(wrs)
            summary[alpha_str][stoch] = {
                **stats,
                "zone_diff_mean":  round(sum(zdiffs) / len(zdiffs), 4) if zdiffs else None,
                "risk_diff_mean":  round(sum(rdiffs) / len(rdiffs), 4) if rdiffs else None,
                "n_degenerate":    deg_count,
                "collapse_rate":   round(deg_count / len(trials), 4),
                "trials":          trials,
            }
    return summary

# ── Summary computation ───────────────────────────────────────────────────────

def summarise_phase2(data: dict) -> dict:
    summary = {}
    for algo in PHASE2_ALGOS:
        summary[algo] = {}
        for stoch in STOCH_LEVELS:
            vals = data[algo][stoch]
            if not vals:
                summary[algo][stoch] = None
                continue
            stats = mean_std_n(vals)
            summary[algo][stoch] = stats
    return summary

def summarise_phase3(data: dict) -> dict:
    summary = {}
    for algo in PHASE3_FILES:
        summary[algo] = {}
        for stoch in STOCH_LEVELS:
            trials = data[algo][stoch]
            if not trials:
                summary[algo][stoch] = None
                continue
            wrs        = [t["wr"] for t in trials]
            zdiffs     = [t["zone_diff"] for t in trials if t["zone_diff"] is not None]
            rdiffs     = [t["risk_diff"] for t in trials if t["risk_diff"] is not None]
            deg_count  = sum(1 for t in trials if t["degenerate"])
            stats = mean_std_n(wrs)
            summary[algo][stoch] = {
                **stats,
                "zone_diff_mean":  round(sum(zdiffs) / len(zdiffs), 4) if zdiffs else None,
                "risk_diff_mean":  round(sum(rdiffs) / len(rdiffs), 4) if rdiffs else None,
                "n_degenerate":    deg_count,
                "collapse_rate":   round(deg_count / len(trials), 4),
                "trials":          trials,
            }
    return summary

def competitive_rate(stoch_summary: dict) -> float:
    total = sum(s["n"] for s in stoch_summary.values() if s)
    degen = sum(s["n_degenerate"] for s in stoch_summary.values() if s)
    return round(1 - degen / total, 4) if total else 0.0

# ── Markdown rendering ────────────────────────────────────────────────────────

ALGO_DISPLAY = {
    "vanilla":      "Vanilla DQN",
    "double":       "Double DQN",
    "dueling":      "Dueling DQN",
    "rainbow_lite": "Rainbow-lite",
}

def fmt(v, dec=3):
    return f"{v:.{dec}f}" if v is not None else "—"

def render_md(p2: dict, p3: dict, p4: dict = None) -> str:
    lines = []
    lines.append("# Cross-Algorithm Summary Tables\n")
    lines.append("*Generated automatically by `scripts/compute_summary_table.py`*\n")

    # ── Phase 2 table ──
    lines.append("## Phase 2: Single-Agent Win Rate vs Base Agent")
    lines.append("")
    lines.append("Win rate is the proportion of 150 evaluation races won against the fixed "
                 "gap-aware heuristic Base Agent. 95 % CI computed from the across-seed "
                 "standard deviation (n = 3 seeds). Higher is better.")
    lines.append("")
    lines.append("| Algorithm | s0 mean (CI 95 %) | s1 mean (CI 95 %) | s2 mean (CI 95 %) | "
                 "s0→s2 drop | s0 seed std |")
    lines.append("|-----------|-------------------|-------------------|-------------------|"
                 "------------|-------------|")
    for algo in ["vanilla", "double", "dueling", "rainbow_lite"]:
        row = [ALGO_DISPLAY[algo]]
        s0 = p2[algo]["s0"]
        for stoch in STOCH_LEVELS:
            s = p2[algo][stoch]
            if s:
                row.append(f"{fmt(s['mean'])} [{fmt(s['ci95_low'])}, {fmt(s['ci95_high'])}]")
            else:
                row.append("—")
        drop = None
        if p2[algo]["s0"] and p2[algo]["s2"]:
            drop = p2[algo]["s2"]["mean"] - p2[algo]["s0"]["mean"]
        row.append(fmt(drop, 3) if drop is not None else "—")
        row.append(fmt(s0["std"], 4) if s0 else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # ── Phase 3 table ──
    lines.append("## Phase 3: MARL Win Rate (A1 vs A2, 500-episode budget)")
    lines.append("")
    lines.append("Win rate is the proportion of 150 evaluation races where Agent 1 finishes "
                 "ahead of Agent 2 in independent-learner MARL (n = 3 seeds per cell). "
                 "A win rate of 0.5 indicates parity. Degenerate collapse is defined as "
                 "WR ≥ 0.85 or WR ≤ 0.15 (one agent completely dominates). "
                 "95 % CI computed from across-seed standard deviation.")
    lines.append("")
    lines.append("| Algorithm | s0 mean (CI 95 %) | s1 mean (CI 95 %) | s2 mean (CI 95 %) | "
                 "Collapse rate | Competitive rate |")
    lines.append("|-----------|-------------------|-------------------|-------------------|"
                 "--------------|-----------------|")
    for algo in ["vanilla", "double", "dueling", "rainbow_lite"]:
        row = [ALGO_DISPLAY[algo]]
        for stoch in STOCH_LEVELS:
            s = p3[algo][stoch]
            if s:
                row.append(f"{fmt(s['mean'])} [{fmt(s['ci95_low'])}, {fmt(s['ci95_high'])}]")
            else:
                row.append("—")
        algo_sum = {stoch: p3[algo][stoch] for stoch in STOCH_LEVELS}
        total    = sum(s["n"] for s in algo_sum.values() if s)
        degen    = sum(s["n_degenerate"] for s in algo_sum.values() if s)
        col_rate = round(degen / total, 3) if total else 0.0
        comp_rate = round(1 - col_rate, 3)
        row.append(f"{col_rate:.1%} ({degen}/{total})")
        row.append(f"{comp_rate:.1%}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # ── Phase 3 per-stoch detail ──
    lines.append("## Phase 3: Collapse counts by stochasticity level")
    lines.append("")
    lines.append("| Algorithm | s0 collapses | s1 collapses | s2 collapses | "
                 "Zone diff (mean) | Risk diff (mean) |")
    lines.append("|-----------|--------------|--------------|--------------|"
                 "-----------------|-----------------|")
    for algo in ["vanilla", "double", "dueling", "rainbow_lite"]:
        row = [ALGO_DISPLAY[algo]]
        zdiffs_all, rdiffs_all = [], []
        for stoch in STOCH_LEVELS:
            s = p3[algo][stoch]
            if s:
                row.append(f"{s['n_degenerate']}/{s['n']}")
                if s["zone_diff_mean"] is not None:
                    zdiffs_all.append(s["zone_diff_mean"])
                if s["risk_diff_mean"] is not None:
                    rdiffs_all.append(s["risk_diff_mean"])
            else:
                row.append("—")
        row.append(fmt(sum(zdiffs_all)/len(zdiffs_all), 3) if zdiffs_all else "—")
        row.append(fmt(sum(rdiffs_all)/len(rdiffs_all), 3) if rdiffs_all else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # ── Individual trial listing ──
    lines.append("## Phase 3: Full trial listing (500-episode core dataset)")
    lines.append("")
    lines.append("| Algorithm | Stoch | Seed | A1 WR | Degenerate | Zone diff | Risk diff |")
    lines.append("|-----------|-------|------|-------|------------|-----------|-----------|")
    for algo in ["vanilla", "double", "dueling", "rainbow_lite"]:
        for stoch in STOCH_LEVELS:
            s = p3[algo][stoch]
            if not s:
                continue
            for t in s["trials"]:
                seed = t["stem"].split("_s")[-1]
                deg  = "YES" if t["degenerate"] else "no"
                lines.append(f"| {ALGO_DISPLAY[algo]} | {stoch} | {seed} | "
                             f"{fmt(t['wr'])} | {deg} | "
                             f"{fmt(t['zone_diff'])} | {fmt(t['risk_diff'])} |")

    # ── Phase 4 table ──
    if p4:
        lines.append("## Phase 4: Incentive Regime Sweep (Rainbow-lite, alpha sweep)")
        lines.append("")
        lines.append("All trials use rainbow-lite as the algorithmic substrate. Alpha is the "
                     "reward-sharing coefficient: 0.0 = purely competitive (Phase 3 baseline), "
                     "0.25 = weakly cooperative, 0.50 = balanced mixed, 1.0 = fully cooperative. "
                     "Zone and risk differentiation indices are the primary RQ1/RQ2 evidence. "
                     "Degenerate collapse is WR >= 0.85 or <= 0.15. n = 3 seeds per cell.")
        lines.append("")
        lines.append("| Alpha | s0 WR (CI 95 %) | s1 WR (CI 95 %) | s2 WR (CI 95 %) | "
                     "Collapse rate | Zone diff (mean) | Risk diff (mean) |")
        lines.append("|-------|-----------------|-----------------|-----------------|"
                     "--------------|-----------------|-----------------|")

        # Include alpha=0.0 baseline from Phase 3 rainbow data for direct comparison
        rainbow_p3 = p3.get("rainbow_lite", {})
        baseline_row = ["0.0 (baseline)"]
        for stoch in STOCH_LEVELS:
            s = rainbow_p3.get(stoch)
            if s:
                baseline_row.append(f"{fmt(s['mean'])} [{fmt(s['ci95_low'])}, {fmt(s['ci95_high'])}]")
            else:
                baseline_row.append("—")
        total_b = sum(rainbow_p3[st]["n"] for st in STOCH_LEVELS if rainbow_p3.get(st))
        degen_b = sum(rainbow_p3[st]["n_degenerate"] for st in STOCH_LEVELS if rainbow_p3.get(st))
        col_b = f"{round(degen_b / total_b, 3):.1%} ({degen_b}/{total_b})" if total_b else "—"
        zdiffs_b = [rainbow_p3[st]["zone_diff_mean"] for st in STOCH_LEVELS
                    if rainbow_p3.get(st) and rainbow_p3[st]["zone_diff_mean"] is not None]
        rdiffs_b = [rainbow_p3[st]["risk_diff_mean"] for st in STOCH_LEVELS
                    if rainbow_p3.get(st) and rainbow_p3[st]["risk_diff_mean"] is not None]
        baseline_row.append(col_b)
        baseline_row.append(fmt(sum(zdiffs_b) / len(zdiffs_b), 3) if zdiffs_b else "—")
        baseline_row.append(fmt(sum(rdiffs_b) / len(rdiffs_b), 3) if rdiffs_b else "—")
        lines.append("| " + " | ".join(baseline_row) + " |")

        for alpha_str in PHASE4_ALPHAS:
            row = [alpha_str]
            for stoch in STOCH_LEVELS:
                s = p4.get(alpha_str, {}).get(stoch)
                if s:
                    row.append(f"{fmt(s['mean'])} [{fmt(s['ci95_low'])}, {fmt(s['ci95_high'])}]")
                else:
                    row.append("—")
            all_cells = [p4.get(alpha_str, {}).get(st) for st in STOCH_LEVELS]
            total = sum(s["n"] for s in all_cells if s)
            degen = sum(s["n_degenerate"] for s in all_cells if s)
            col_rate = f"{round(degen / total, 3):.1%} ({degen}/{total})" if total else "—"
            zdiffs = [s["zone_diff_mean"] for s in all_cells if s and s["zone_diff_mean"] is not None]
            rdiffs = [s["risk_diff_mean"] for s in all_cells if s and s["risk_diff_mean"] is not None]
            row.append(col_rate)
            row.append(fmt(sum(zdiffs) / len(zdiffs), 3) if zdiffs else "—")
            row.append(fmt(sum(rdiffs) / len(rdiffs), 3) if rdiffs else "—")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

        # Per-trial listing
        lines.append("## Phase 4: Full trial listing")
        lines.append("")
        lines.append("| Alpha | Stoch | Seed | A1 WR | Degenerate | Zone diff | Risk diff |")
        lines.append("|-------|-------|------|-------|------------|-----------|-----------|")
        for alpha_str in PHASE4_ALPHAS:
            for stoch in STOCH_LEVELS:
                s = p4.get(alpha_str, {}).get(stoch)
                if not s:
                    continue
                for t in s["trials"]:
                    seed = t["stem"].split("_s")[-1]
                    deg = "YES" if t["degenerate"] else "no"
                    lines.append(f"| {alpha_str} | {stoch} | {seed} | "
                                 f"{fmt(t['wr'])} | {deg} | "
                                 f"{fmt(t['zone_diff'])} | {fmt(t['risk_diff'])} |")
        lines.append("")

    return "\n".join(lines) + "\n"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[compute_summary_table] Loading Phase 2 data...")
    p2_raw = load_phase2()
    p2     = summarise_phase2(p2_raw)

    print("[compute_summary_table] Loading Phase 3 data...")
    p3_raw = load_phase3()
    p3     = summarise_phase3(p3_raw)

    print("[compute_summary_table] Loading Phase 4 data...")
    p4_raw = load_phase4()
    p4     = summarise_phase4(p4_raw)

    # Print console summary
    print("\n=== PHASE 2: Single-agent win rate vs Base Agent ===")
    print(f"{'Algorithm':<16} {'s0':>8} {'s1':>8} {'s2':>8}  s0→s2 drop")
    for algo in ["vanilla", "double", "dueling", "rainbow_lite"]:
        vals = []
        for stoch in STOCH_LEVELS:
            s = p2[algo][stoch]
            vals.append(f"{s['mean']:.3f}" if s else "  —  ")
        s0v = p2[algo]["s0"]["mean"] if p2[algo]["s0"] else None
        s2v = p2[algo]["s2"]["mean"] if p2[algo]["s2"] else None
        drop = f"{s2v - s0v:+.3f}" if s0v and s2v else "  —"
        print(f"  {ALGO_DISPLAY[algo]:<14} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}  {drop}")

    print("\n=== PHASE 3: MARL win rate (A1 vs A2, 500ep) ===")
    print(f"{'Algorithm':<16} {'s0':>8} {'s1':>8} {'s2':>8}  Collapse  Competitive")
    for algo in ["vanilla", "double", "dueling", "rainbow_lite"]:
        vals = []
        for stoch in STOCH_LEVELS:
            s = p3[algo][stoch]
            vals.append(f"{s['mean']:.3f}" if s else "  —  ")
        total = sum(p3[algo][st]["n"] for st in STOCH_LEVELS if p3[algo][st])
        degen = sum(p3[algo][st]["n_degenerate"] for st in STOCH_LEVELS if p3[algo][st])
        col_r = f"{degen}/{total} ({degen/total:.0%})" if total else "—"
        comp_r = f"{1-degen/total:.0%}" if total else "—"
        print(f"  {ALGO_DISPLAY[algo]:<14} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}  "
              f"{col_r:<12}  {comp_r}")

    print("\n=== PHASE 3: Collapse breakdown by stochasticity ===")
    print(f"{'Algorithm':<16} {'s0 col':>8} {'s1 col':>8} {'s2 col':>8}")
    for algo in ["vanilla", "double", "dueling", "rainbow_lite"]:
        row = []
        for stoch in STOCH_LEVELS:
            s = p3[algo][stoch]
            row.append(f"{s['n_degenerate']}/{s['n']}" if s else "—")
        print(f"  {ALGO_DISPLAY[algo]:<14} {row[0]:>8} {row[1]:>8} {row[2]:>8}")

    if any(p4[a][st] for a in PHASE4_ALPHAS for st in STOCH_LEVELS):
        print("\n=== PHASE 4: Alpha sweep (rainbow-lite, zone/risk differentiation) ===")
        print(f"{'Alpha':<8} {'s0 WR':>8} {'s1 WR':>8} {'s2 WR':>8}  Collapse  ZoneDiff  RiskDiff")
        for alpha_str in PHASE4_ALPHAS:
            vals, zdiffs, rdiffs = [], [], []
            total, degen = 0, 0
            for stoch in STOCH_LEVELS:
                s = p4[alpha_str][stoch]
                vals.append(f"{s['mean']:.3f}" if s else "  —  ")
                if s:
                    total += s["n"]
                    degen += s["n_degenerate"]
                    if s["zone_diff_mean"] is not None:
                        zdiffs.append(s["zone_diff_mean"])
                    if s["risk_diff_mean"] is not None:
                        rdiffs.append(s["risk_diff_mean"])
            col_r = f"{degen}/{total}" if total else "—"
            zd = f"{sum(zdiffs)/len(zdiffs):.3f}" if zdiffs else "  —"
            rd = f"{sum(rdiffs)/len(rdiffs):.3f}" if rdiffs else "  —"
            print(f"  {alpha_str:<6} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}  "
                  f"{col_r:<10} {zd:>8}  {rd:>8}")

    # Write outputs
    out_data = {
        "phase2": p2,
        "phase3": {
            algo: {stoch: {k: v for k, v in cell.items() if k != "trials"}
                   for stoch, cell in stoch_map.items() if cell}
            for algo, stoch_map in p3.items()
        },
        "phase4": {
            alpha_str: {stoch: {k: v for k, v in cell.items() if k != "trials"}
                        for stoch, cell in stoch_map.items() if cell}
            for alpha_str, stoch_map in p4.items()
        },
    }
    OUT_JSON.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\n[compute_summary_table] JSON written to: {OUT_JSON}")

    md = render_md(p2, p3, p4)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"[compute_summary_table] Markdown written to: {OUT_MD}")

if __name__ == "__main__":
    main()
