"""
Evaluate two concurrent DQN agents training simultaneously (independent learner MARL).

Phase 3 entry point. Both agents train from their own replay buffers each episode.
Primary metric: win rate of Agent 1 vs Agent 2 (relative positional advantage).
Per-agent behavioural diagnostics reported separately.

Usage:
  conda run -n f1StrategySim python scripts/evaluate_marl.py \
    --config metrics/phase3/config_vanilla_marl.json \
    --train-runs 500 --eval-runs 150 \
    --train-seed 101 --eval-seeds 101 \
    --stochasticity-level s0 \
    --out metrics/phase3/vanilla_marl_s0_s101.json
"""

import argparse
import copy
import io
import json
import os
import random
import sys
from contextlib import redirect_stdout
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from src.simulator import DEFAULT_PROTOCOL_CONTRACT, RaceSimulator, init_simulator
from src.states import init_race_state
from src.track import load_track
from runtime_profiles import resolve_complexity_profile


def _disable_visualisations() -> None:
    def _noop(*args, **kwargs):
        return None
    RaceSimulator._visualise_results = _noop
    RaceSimulator._visualise_agent_learning = _noop


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.array(values, dtype=float), ddof=1))


def _ci95(values: List[float]) -> Dict:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = _mean(values)
    std = _std(values)
    margin = 1.96 * (std / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n, "ci95_low": mean - margin, "ci95_high": mean + margin}


def _read_jsonl_records(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _group_by_run(records: List[Dict]) -> Dict[int, Dict[str, Dict]]:
    grouped: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for record in records:
        run_number = int(record.get("run_number", -1))
        driver_name = record.get("driver_name")
        if run_number < 0 or not driver_name:
            continue
        grouped[run_number][driver_name] = record.get("data", {})
    return grouped


def _remap_run_numbers(records: List[Dict], run_offset: int) -> List[Dict]:
    return [{**r, "run_number": int(r.get("run_number", 0)) + run_offset} for r in records]


def _compute_agent_diagnostics(records: List[Dict], agent_name: str) -> Dict:
    """Extract behavioural diagnostics for a single named agent."""
    risk_counts = {"CONSERVATIVE": 0, "NORMAL": 0, "AGGRESSIVE": 0}
    zone_aggregate: Dict[str, Dict] = {}
    overtakes_attempted = 0
    overtakes_succeeded = 0
    positions = []
    finish_times = []
    dnf_count = 0
    total_laps_cfg = 5

    run_data = _group_by_run(records)
    for run_idx in sorted(run_data.keys()):
        per_driver = run_data[run_idx]
        data = per_driver.get(agent_name)
        if not data:
            continue
        pos = float(data.get("position", 999))
        positions.append(pos)
        ft = float(data.get("finish_time", 1e12))
        finish_times.append(ft)
        laps = int(data.get("laps", 0) or 0)
        if laps < total_laps_cfg:
            dnf_count += 1
        overtakes_attempted += int(data.get("overtakes_attempted", 0) or 0)
        overtakes_succeeded += int(data.get("overtakes_succeeded", 0) or 0)

        decision_summary = data.get("decision_summary", {})
        if isinstance(decision_summary, dict):
            for rk, cnt in decision_summary.get("risk_attempt_counts", {}).items():
                risk_counts[rk] = risk_counts.get(rk, 0) + int(cnt or 0)
            zone_stats = decision_summary.get("zone_stats", {})
            if isinstance(zone_stats, dict):
                for zone_id, zdata in zone_stats.items():
                    if not isinstance(zdata, dict):
                        continue
                    bucket = zone_aggregate.setdefault(str(zone_id), {
                        "zone_name": str(zdata.get("zone_name", zone_id)),
                        "zone_difficulty": float(zdata.get("zone_difficulty", 0.5)),
                        "decisions": 0, "attempts": 0, "successes": 0,
                        "_reward_sum": 0.0, "_success_prob_sum": 0.0,
                    })
                    decisions = int(zdata.get("decisions", 0) or 0)
                    attempts = int(zdata.get("attempts", 0) or 0)
                    successes = int(zdata.get("successes", 0) or 0)
                    bucket["decisions"] += decisions
                    bucket["attempts"] += attempts
                    bucket["successes"] += successes
                    bucket["_reward_sum"] += float(zdata.get("avg_reward", 0.0) or 0.0) * max(1, decisions)
                    bucket["_success_prob_sum"] += float(zdata.get("avg_success_probability", 0.0) or 0.0) * max(1, decisions)

    zone_behavior = {}
    for zone_id, bucket in zone_aggregate.items():
        dec = max(1, bucket["decisions"])
        att = bucket["attempts"]
        zone_behavior[zone_id] = {
            "zone_name": bucket["zone_name"],
            "zone_difficulty": float(bucket["zone_difficulty"]),
            "decisions": bucket["decisions"],
            "attempts": att,
            "successes": bucket["successes"],
            "attempt_rate": float(att / dec),
            "success_rate": float(bucket["successes"] / att) if att > 0 else 0.0,
            "avg_reward": float(bucket["_reward_sum"] / dec),
            "avg_success_probability": float(bucket["_success_prob_sum"] / dec),
        }

    n_runs = len(positions)
    osr = float(overtakes_succeeded / overtakes_attempted) if overtakes_attempted > 0 else 0.0
    dnf_rate = float(dnf_count / n_runs) if n_runs > 0 else 0.0
    return {
        "avg_position": _ci95(positions),
        "risk_attempt_counts": risk_counts,
        "zone_behavior": zone_behavior,
        "overtake_success_rate": {"attempted": overtakes_attempted, "succeeded": overtakes_succeeded, "rate": osr},
        "dnf_rate": dnf_rate,
        "n_runs": n_runs,
    }


def _compute_marl_metrics(
    records: List[Dict],
    agent1_name: str,
    agent2_name: str,
) -> Dict:
    """Compute MARL-specific metrics: relative positional advantage + per-agent diagnostics."""
    run_data = _group_by_run(records)

    win_flags_a1: List[float] = []
    pos_deltas: List[float] = []  # pos_a2 - pos_a1 (positive means a1 better)

    # Non-stationarity signal: split evaluation into thirds
    run_indices = sorted(run_data.keys())
    n_runs = len(run_indices)
    third = max(1, n_runs // 3)
    early_wins = []
    late_wins = []

    for i, run_idx in enumerate(run_indices):
        per_driver = run_data[run_idx]
        d1 = per_driver.get(agent1_name)
        d2 = per_driver.get(agent2_name)
        if not d1 or not d2:
            continue
        p1 = float(d1.get("position", 999))
        p2 = float(d2.get("position", 999))
        win = 1.0 if p1 < p2 else 0.0
        win_flags_a1.append(win)
        pos_deltas.append(p2 - p1)
        if i < third:
            early_wins.append(win)
        if i >= n_runs - third:
            late_wins.append(win)

    win_rate_a1 = _ci95(win_flags_a1)
    pos_advantage = _ci95(pos_deltas)
    non_stationarity_signal = {
        "early_win_rate_a1": _mean(early_wins),
        "late_win_rate_a1": _mean(late_wins),
        "drift": _mean(late_wins) - _mean(early_wins),
        "interpretation": (
            "a1 strengthening" if _mean(late_wins) - _mean(early_wins) > 0.05
            else "a2 strengthening" if _mean(late_wins) - _mean(early_wins) < -0.05
            else "stable"
        ),
    }

    diag_a1 = _compute_agent_diagnostics(records, agent1_name)
    diag_a2 = _compute_agent_diagnostics(records, agent2_name)

    # Strategy differentiation index
    risk_total_a1 = sum(diag_a1["risk_attempt_counts"].values()) or 1
    risk_total_a2 = sum(diag_a2["risk_attempt_counts"].values()) or 1
    risk_diff = sum(
        abs(diag_a1["risk_attempt_counts"].get(k, 0) / risk_total_a1
            - diag_a2["risk_attempt_counts"].get(k, 0) / risk_total_a2)
        for k in ["CONSERVATIVE", "NORMAL", "AGGRESSIVE"]
    ) / 3.0

    all_zones = set(diag_a1["zone_behavior"]) | set(diag_a2["zone_behavior"])
    zone_diffs = []
    for z in all_zones:
        ar1 = diag_a1["zone_behavior"].get(z, {}).get("attempt_rate", 0.0)
        ar2 = diag_a2["zone_behavior"].get(z, {}).get("attempt_rate", 0.0)
        zone_diffs.append(abs(ar1 - ar2))
    zone_diff_index = _mean(zone_diffs)

    return {
        "win_rate_a1_vs_a2": win_rate_a1,
        "positional_advantage_a1": pos_advantage,
        "non_stationarity_signal": non_stationarity_signal,
        "agent1": {"name": agent1_name, **diag_a1},
        "agent2": {"name": agent2_name, **diag_a2},
        "strategy_differentiation": {
            "risk_differentiation_index": float(risk_diff),
            "zone_differentiation_index": float(zone_diff_index),
            "interpretation": (
                "high" if zone_diff_index > 0.25
                else "moderate" if zone_diff_index > 0.10
                else "low"
            ),
        },
    }


def _prepare_phase_config(base_config, phase, runs, run_name, stochasticity_level=None, total_laps=None):
    cfg = copy.deepcopy(base_config)
    sim = cfg.setdefault("simulator", {})
    sim["method"] = "batch"
    sim["runs"] = int(runs)
    sim["run_name"] = run_name
    sim["visualise_from_run_name"] = ""
    sim["agent_mode"] = phase
    if total_laps is not None:
        cfg.setdefault("race_settings", {})["total_laps"] = int(total_laps)
    if stochasticity_level:
        cfg.setdefault("stochasticity", {})["active_level"] = str(stochasticity_level).strip()
    cfg["agent_review_mode"] = False
    cfg["debugMode"] = False
    return cfg


def _run_phase(cfg, seed, verbose):
    _set_deterministic_seed(seed)
    if verbose:
        track = load_track(cfg)
        race_state = init_race_state(cfg, track)
        simulator = init_simulator(race_state, cfg, track)
    else:
        with redirect_stdout(io.StringIO()):
            track = load_track(cfg)
            race_state = init_race_state(cfg, track)
            simulator = init_simulator(race_state, cfg, track)
    race_log_path = simulator.race_results_log_path
    if race_log_path is None:
        raise RuntimeError("Simulation phase did not produce race_results.jsonl")
    records = _read_jsonl_records(Path(race_log_path))
    return Path(race_log_path), records


def _parse_seeds(eval_seeds_str, fallback_seed):
    if not eval_seeds_str.strip():
        return [int(fallback_seed)]
    seeds = []
    for token in eval_seeds_str.split(","):
        t = token.strip()
        if t:
            seeds.append(int(t))
    return seeds if seeds else [int(fallback_seed)]


def main():
    parser = argparse.ArgumentParser(description="Phase 3 MARL evaluation: two concurrent DQN agents.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--train-runs", type=int, default=500)
    parser.add_argument("--eval-runs", type=int, default=150)
    parser.add_argument("--train-seed", type=int, default=101)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-seeds", default="")
    parser.add_argument("--stochasticity-level", default="s0")
    parser.add_argument("--complexity-profile", default="low_marl")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--run-prefix", default="marl_eval")
    parser.add_argument("--out", default="metrics/phase3/latest_marl.json")
    parser.add_argument(
        "--alpha", type=float, default=None,
        help=(
            "Reward-sharing coefficient (0.0 = purely competitive, 1.0 = fully cooperative). "
            "Overrides config marl.reward_sharing_alpha when provided. "
            "Omitting this flag preserves the config value, defaulting to 0.0 for full "
            "idempotency with Phase 3 runs."
        ),
    )
    args = parser.parse_args()

    _disable_visualisations()

    base_config = _load_config((ROOT / args.config).resolve())

    # Force low_marl complexity
    base_config.setdefault("complexity", {})["active_profile"] = args.complexity_profile

    # Apply reward-sharing alpha override (Phase 4).  When --alpha is not supplied the
    # config value is used as-is, preserving numerical identity with Phase 3 runs.
    if args.alpha is not None:
        base_config.setdefault("marl", {})["reward_sharing_alpha"] = args.alpha

    # Resolve algo name from dqn_params
    dqn_cfg = base_config.get("dqn_params", {})
    algo_name = str(dqn_cfg.get("algo", "vanilla")).strip().lower() or "vanilla"

    # Rename the two DQN competitors with distinct, stable names
    dqn_competitors_seen = 0
    agent1_name = None
    agent2_name = None
    for c in base_config.get("competitors", []):
        if isinstance(c, dict) and str(c.get("agent", "")).lower() == "dqn":
            dqn_competitors_seen += 1
            if dqn_competitors_seen == 1:
                c["name"] = f"DQN_A1_{algo_name}"
                agent1_name = c["name"]
            elif dqn_competitors_seen == 2:
                c["name"] = f"DQN_A2_{algo_name}"
                agent2_name = c["name"]

    if not agent1_name or not agent2_name:
        raise ValueError("MARL evaluation requires exactly two competitors with agent='dqn' in config.")

    stoch = args.stochasticity_level.strip() or "s0"
    base_config.setdefault("stochasticity", {})["active_level"] = stoch

    total_laps = int(base_config.get("race_settings", {}).get("total_laps", 5) or 5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fallback_eval_seed = args.eval_seed if args.eval_seed is not None else args.train_seed
    eval_seeds = _parse_seeds(args.eval_seeds, fallback_eval_seed)

    # Training phase
    train_log_path = None
    if not args.skip_training:
        train_run_name = f"{args.run_prefix}_train_{timestamp}"
        train_cfg = _prepare_phase_config(
            base_config, phase="training",
            runs=args.train_runs, run_name=train_run_name,
            stochasticity_level=stoch, total_laps=total_laps,
        )
        print(
            f"[evaluate_marl] Training: algo={algo_name}, runs={args.train_runs}, "
            f"seed={args.train_seed}, stoch={stoch}"
        )
        train_log_path, _ = _run_phase(train_cfg, seed=args.train_seed, verbose=args.verbose)

    # Evaluation phase
    all_eval_records = []
    eval_runs_meta = []
    run_offset = 0
    for seed in eval_seeds:
        eval_run_name = f"{args.run_prefix}_eval_{timestamp}_{stoch}_s{seed}"
        eval_cfg = _prepare_phase_config(
            base_config, phase="evaluation",
            runs=args.eval_runs, run_name=eval_run_name,
            stochasticity_level=stoch, total_laps=total_laps,
        )
        print(f"[evaluate_marl] Evaluation: seed={seed}, runs={args.eval_runs}, stoch={stoch}")
        eval_log_path, eval_records = _run_phase(eval_cfg, seed=seed, verbose=args.verbose)
        all_eval_records.extend(_remap_run_numbers(eval_records, run_offset))
        eval_runs_meta.append({"seed": seed, "runs": args.eval_runs, "stoch": stoch,
                                "log_path": str(eval_log_path)})
        run_offset += args.eval_runs

    # Compute metrics
    marl_metrics = _compute_marl_metrics(all_eval_records, agent1_name, agent2_name)

    # Seed-level win rate variance
    seed_wrs = []
    for run_info in eval_runs_meta:
        seed_records = _read_jsonl_records(Path(run_info["log_path"]))
        seed_m = _compute_marl_metrics(seed_records, agent1_name, agent2_name)
        seed_wrs.append(seed_m["win_rate_a1_vs_a2"]["mean"])

    seed_variance = float(np.var(np.array(seed_wrs, dtype=float), ddof=1)) if len(seed_wrs) > 1 else 0.0

    out_payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": str((ROOT / args.config).resolve()),
        "phase": "phase4_marl" if float(base_config.get("marl", {}).get("reward_sharing_alpha", 0.0)) > 0.0 else "phase3_marl",
        "reward_sharing_alpha": float(base_config.get("marl", {}).get("reward_sharing_alpha", 0.0)),
        "algorithm": algo_name,
        "agent1_name": agent1_name,
        "agent2_name": agent2_name,
        "stochasticity_level": stoch,
        "objective_score": float(marl_metrics["win_rate_a1_vs_a2"]["mean"]),
        "phases": {
            "training": {
                "skipped": bool(args.skip_training),
                "runs": int(args.train_runs),
                "seed": int(args.train_seed),
                "train_log_path": str(train_log_path) if train_log_path else None,
            },
            "evaluation": {
                "runs_per_seed": int(args.eval_runs),
                "seeds": [int(s) for s in eval_seeds],
                "seed_runs": eval_runs_meta,
            },
        },
        "metrics": marl_metrics,
        "stability": {
            "seed_win_rate_variance": seed_variance,
            "seed_win_rates_a1": seed_wrs,
        },
    }

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    wr = marl_metrics["win_rate_a1_vs_a2"]
    diff = marl_metrics["strategy_differentiation"]
    print(f"[evaluate_marl] {agent1_name} win rate: {wr['mean']:.3f} [{wr['ci95_low']:.3f}, {wr['ci95_high']:.3f}]")
    print(f"[evaluate_marl] Zone differentiation: {diff['zone_differentiation_index']:.3f} ({diff['interpretation']})")
    print(f"[evaluate_marl] Non-stationarity drift: {marl_metrics['non_stationarity_signal']['drift']:+.3f}")
    print(f"[evaluate_marl] Metrics written to: {out_path}")


if __name__ == "__main__":
    main()
