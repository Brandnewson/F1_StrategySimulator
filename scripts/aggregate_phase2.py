"""
Aggregate Phase 2 s0 results from 12 per-trial JSONs and print a comparison table.
Usage: conda run -n f1StrategySim python scripts/aggregate_phase2.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2_DIR = ROOT / "metrics" / "phase2"

ALGOS = ["vanilla", "double", "dueling", "rainbow_lite"]
SEEDS = [101, 202, 303]
STOCH = "s0"


def ci95(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    margin = 1.96 * std / math.sqrt(n)
    return mean, mean - margin, mean + margin


def load(algo, seed):
    name = "rainbow" if algo == "rainbow_lite" else algo
    path = PHASE2_DIR / f"{name}_s0_s{seed}.json"
    with path.open() as f:
        return json.load(f)


def extract(data):
    m = data.get("metrics", {})
    wr = m.get("win_rate_vs_baseline", {})
    tactical = m.get("tactical", {})
    race_q = m.get("race_quality", {})
    bdiag = m.get("behavioral_diagnostics", {})
    fair = m.get("fairness_diagnostics", {})
    stab = m.get("stability", {})

    return {
        "win_rate": wr.get("mean", 0.0),
        "objective": data.get("objective_score", 0.0),
        "overtake_success_rate": tactical.get("overtake_success_rate", {}).get("rate", 0.0),
        "avg_pos_delta": race_q.get("avg_position_delta_non_dqn_minus_dqn", {}).get("mean", 0.0),
        "risk_dist": bdiag.get("risk_attempt_counts", {}),
        "zone_behavior": bdiag.get("zone_behavior", {}),
        "tactical_raw_total": bdiag.get("tactical_raw_total", None),
        "fairness_violations": len(fair.get("violations", [])),
        "seed_variance": stab.get("win_rate_vs_baseline_variance_across_seeds", 0.0),
        "dnf_rate": stab.get("dnf_rate_dqn", {}).get("rate", 0.0),
    }


results = {}
for algo in ALGOS:
    seed_data = []
    for seed in SEEDS:
        try:
            raw = load(algo, seed)
            seed_data.append(extract(raw))
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")
    results[algo] = seed_data

# ── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PHASE 2 — s0 PRIMARY RANKING TRACK  (3 seeds × 150 eval races each)")
print("=" * 80)

header = f"{'Algo':<14} {'WinRate':>8} {'CI95 low':>9} {'CI95 high':>10} {'ObjScore':>9} {'OvtkSucc':>9} {'PosDelta':>9} {'SeedVar':>9} {'DNF':>6}"
print(header)
print("-" * 80)

vanilla_mean, vanilla_lo, vanilla_hi = ci95([d["win_rate"] for d in results["vanilla"]])

for algo in ALGOS:
    data = results[algo]
    win_rates = [d["win_rate"] for d in data]
    obj_scores = [d["objective"] for d in data]
    mean_wr, lo_wr, hi_wr = ci95(win_rates)
    mean_obj = sum(obj_scores) / len(obj_scores) if obj_scores else 0.0
    mean_osr = sum(d["overtake_success_rate"] for d in data) / len(data) if data else 0.0
    mean_pd  = sum(d["avg_pos_delta"] for d in data) / len(data) if data else 0.0
    mean_sv  = sum(d["seed_variance"] for d in data) / len(data) if data else 0.0
    mean_dnf = sum(d["dnf_rate"] for d in data) / len(data) if data else 0.0

    if algo == "vanilla":
        vs = "control"
    elif lo_wr > vanilla_hi:
        vs = "BETTER ▲"
    elif hi_wr < vanilla_lo:
        vs = "worse ▼"
    else:
        vs = "overlap"

    print(f"{algo:<14} {mean_wr:>8.3f} {lo_wr:>9.3f} {hi_wr:>10.3f} {mean_obj:>9.3f} {mean_osr:>9.3f} {mean_pd:>9.3f} {mean_sv:>9.5f} {mean_dnf:>6.3f}   {vs}")

print()

# ── Per-algo win rates by seed ───────────────────────────────────────────────
print("WIN RATE BY SEED")
print(f"{'Algo':<14} {'s101':>7} {'s202':>7} {'s303':>7} {'range':>8}")
print("-" * 45)
for algo in ALGOS:
    data = results[algo]
    wrs = [d["win_rate"] for d in data]
    vals = "  ".join(f"{w:.3f}" for w in wrs)
    rng = max(wrs) - min(wrs) if wrs else 0.0
    print(f"{algo:<14} {vals}   {rng:.3f}")

print()

# ── Risk distribution ────────────────────────────────────────────────────────
print("RISK DISTRIBUTION (mean across seeds — CONSERVATIVE / NORMAL / AGGRESSIVE)")
print(f"{'Algo':<14} {'CONS':>7} {'NORM':>7} {'AGGR':>7}  {'AGGR%':>6}")
print("-" * 50)
for algo in ALGOS:
    data = results[algo]
    all_cons = sum(d["risk_dist"].get("CONSERVATIVE", 0) for d in data)
    all_norm = sum(d["risk_dist"].get("NORMAL", 0) for d in data)
    all_aggr = sum(d["risk_dist"].get("AGGRESSIVE", 0) for d in data)
    total = all_cons + all_norm + all_aggr
    aggr_pct = 100 * all_aggr / total if total else 0.0
    print(f"{algo:<14} {all_cons:>7} {all_norm:>7} {all_aggr:>7}  {aggr_pct:>5.1f}%")

print()

# ── Tactical reward totals ───────────────────────────────────────────────────
print("TACTICAL RAW TOTAL (sum across seeds — lower = more costly mistakes)")
for algo in ALGOS:
    data = results[algo]
    totals = [d["tactical_raw_total"] for d in data if d["tactical_raw_total"] is not None]
    if totals:
        print(f"  {algo:<14}: {sum(totals):.2f}  (per seed: {[round(t,2) for t in totals]})")

print()

# ── La Source zone check ─────────────────────────────────────────────────────
print("LA SOURCE ZONE (zone1) — attempt rate & success rate")
for algo in ALGOS:
    for i, seed in enumerate(SEEDS):
        d = results[algo][i]
        z = d["zone_behavior"].get("zone1") or d["zone_behavior"].get("overtakingZone1", {})
        if z:
            attempts = z.get("attempts", "?")
            decisions = z.get("decisions", "?")
            sr = z.get("success_rate", 0.0)
            print(f"  {algo:<14} seed={seed}: {attempts}/{decisions} attempts, {sr:.2f} success rate")

print("\n" + "=" * 80)
