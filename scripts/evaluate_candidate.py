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
except Exception:  # pragma: no cover
    torch = None

os.environ.setdefault("MPLBACKEND", "Agg")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))


from src.simulator import RaceSimulator, init_simulator
from src.states import init_race_state
from src.track import load_track


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


def _read_jsonl_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
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


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.array(values, dtype=float), ddof=1))


def _ci95(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = _mean(values)
    std = _std(values)
    margin = 1.96 * (std / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def _compute_win_flags_vs_group(
    run_data: Dict[int, Dict[str, Dict]],
    dqn_driver_names: List[str],
    opponent_driver_names: List[str],
) -> List[float]:
    win_flags: List[float] = []
    for run_idx in sorted(run_data.keys()):
        per_driver = run_data[run_idx]
        dqn_entries = [(name, data) for name, data in per_driver.items() if name in dqn_driver_names]
        opp_entries = [(name, data) for name, data in per_driver.items() if name in opponent_driver_names]
        if not dqn_entries or not opp_entries:
            continue

        best_dqn_position = min(float(d.get("position", 999)) for _, d in dqn_entries)
        best_opp_position = min(float(d.get("position", 999)) for _, d in opp_entries)
        win_flags.append(1.0 if best_dqn_position < best_opp_position else 0.0)
    return win_flags


def _compute_eval_metrics(
    records: List[Dict],
    dqn_driver_names: List[str],
    baseline_driver_names: List[str],
    random_driver_names: List[str],
    total_laps: int,
) -> Dict:
    run_data = _group_by_run(records)

    dqn_avg_positions: List[float] = []
    non_dqn_avg_positions: List[float] = []
    finish_time_deltas: List[float] = []

    dqn_overtakes_attempted = 0
    dqn_overtakes_succeeded = 0
    dqn_dnf_count = 0
    dqn_driver_run_count = 0

    for run_idx in sorted(run_data.keys()):
        per_driver = run_data[run_idx]
        if not per_driver:
            continue

        dqn_entries = [(name, data) for name, data in per_driver.items() if name in dqn_driver_names]
        non_dqn_entries = [(name, data) for name, data in per_driver.items() if name not in dqn_driver_names]
        if not dqn_entries or not non_dqn_entries:
            continue

        dqn_positions = [float(d.get("position", 999)) for _, d in dqn_entries]
        non_dqn_positions = [float(d.get("position", 999)) for _, d in non_dqn_entries]
        dqn_avg_positions.append(_mean(dqn_positions))
        non_dqn_avg_positions.append(_mean(non_dqn_positions))

        best_dqn_time = min(float(d.get("finish_time", 1e12)) for _, d in dqn_entries)
        best_non_dqn_time = min(float(d.get("finish_time", 1e12)) for _, d in non_dqn_entries)
        finish_time_deltas.append(best_non_dqn_time - best_dqn_time)

        for _, data in dqn_entries:
            attempted = int(data.get("overtakes_attempted", 0) or 0)
            succeeded = int(data.get("overtakes_succeeded", 0) or 0)
            laps = int(data.get("laps", 0) or 0)
            dqn_overtakes_attempted += attempted
            dqn_overtakes_succeeded += succeeded
            dqn_driver_run_count += 1
            if laps < total_laps:
                dqn_dnf_count += 1

    win_flags_vs_baseline = _compute_win_flags_vs_group(run_data, dqn_driver_names, baseline_driver_names)
    win_flags_vs_random = _compute_win_flags_vs_group(run_data, dqn_driver_names, random_driver_names)

    win_rate_vs_baseline = _ci95(win_flags_vs_baseline)
    win_rate_vs_random = _ci95(win_flags_vs_random)

    overtake_attempt_rate = (
        float(dqn_overtakes_attempted / dqn_driver_run_count) if dqn_driver_run_count > 0 else 0.0
    )
    overtake_success_rate = (
        float(dqn_overtakes_succeeded / dqn_overtakes_attempted) if dqn_overtakes_attempted > 0 else 0.0
    )
    dnf_rate = float(dqn_dnf_count / dqn_driver_run_count) if dqn_driver_run_count > 0 else 0.0

    return {
        "primary_objective": {
            "name": "win_rate_vs_baseline",
            "score": float(win_rate_vs_baseline["mean"]),
            "details": win_rate_vs_baseline,
        },
        "win_rate_vs_baseline": win_rate_vs_baseline,
        "win_rate_vs_random": win_rate_vs_random,
        "race_quality": {
            "avg_position_dqn": _ci95(dqn_avg_positions),
            "avg_position_non_dqn": _ci95(non_dqn_avg_positions),
            "avg_position_delta_non_dqn_minus_dqn": _ci95(
                [n - d for n, d in zip(non_dqn_avg_positions, dqn_avg_positions)]
            ),
            "avg_finish_time_delta_vs_non_dqn_seconds": _ci95(finish_time_deltas),
        },
        "tactical": {
            "overtake_attempt_rate_per_driver_run": overtake_attempt_rate,
            "overtake_success_rate": {
                "attempted": dqn_overtakes_attempted,
                "succeeded": dqn_overtakes_succeeded,
                "rate": overtake_success_rate,
            },
        },
        "stability": {
            "dnf_rate_dqn": {
                "dnf_count": dqn_dnf_count,
                "driver_runs": dqn_driver_run_count,
                "rate": dnf_rate,
            },
        },
        "num_eval_runs": len(run_data),
    }


def _prepare_phase_config(
    base_config: Dict,
    phase: str,
    runs: int,
    run_name: str,
    total_laps: int | None = None,
    checkpoint_tag: str | None = None,
) -> Dict:
    cfg = copy.deepcopy(base_config)
    simulator_cfg = cfg.setdefault("simulator", {})
    simulator_cfg["method"] = "batch"
    simulator_cfg["runs"] = int(runs)
    simulator_cfg["run_name"] = run_name
    simulator_cfg["visualise_from_run_name"] = ""
    simulator_cfg["agent_mode"] = phase
    if checkpoint_tag:
        simulator_cfg["checkpoint_tag"] = str(checkpoint_tag)
    else:
        simulator_cfg.pop("checkpoint_tag", None)

    if total_laps is not None:
        race_cfg = cfg.setdefault("race_settings", {})
        race_cfg["total_laps"] = int(total_laps)

    cfg["agent_review_mode"] = False
    cfg["debugMode"] = False
    return cfg


def _run_phase(cfg: Dict, seed: int, verbose: bool) -> Tuple[Path, List[Dict]]:
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


def _parse_seeds(eval_seeds: str, fallback_seed: int) -> List[int]:
    if not eval_seeds.strip():
        return [int(fallback_seed)]
    seeds: List[int] = []
    for token in eval_seeds.split(","):
        t = token.strip()
        if not t:
            continue
        seeds.append(int(t))
    return seeds if seeds else [int(fallback_seed)]


def _remap_run_numbers(records: List[Dict], run_offset: int) -> List[Dict]:
    remapped: List[Dict] = []
    for record in records:
        rec = dict(record)
        rec["run_number"] = int(rec.get("run_number", 0)) + int(run_offset)
        remapped.append(rec)
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate candidate policy/model and emit autoresearch-friendly metrics.")
    parser.add_argument("--config", default="config.json", help="Path to base config JSON.")
    parser.add_argument("--train-runs", type=int, default=200, help="Number of batch runs for training phase.")
    parser.add_argument("--eval-runs", type=int, default=200, help="Number of batch runs per evaluation seed.")
    parser.add_argument("--train-seed", type=int, default=1337, help="RNG seed for training phase.")
    parser.add_argument("--eval-seed", type=int, default=7331, help="Fallback RNG seed for evaluation phase.")
    parser.add_argument("--eval-seeds", default="", help="Comma-separated list of eval seeds (e.g. '101,202,303').")
    parser.add_argument("--skip-training", action="store_true", help="Skip training phase and only run evaluation using existing model files.")
    parser.add_argument("--verbose", action="store_true", help="Print simulator output during train/eval phases.")
    parser.add_argument("--guardrail-runs", type=int, default=500, help="Minimum eval runs before no-benefit guardrail is activated.")
    parser.add_argument("--run-prefix", default="cand_eval", help="Prefix for generated run_name fields in logs/.")
    parser.add_argument("--total-laps", type=int, default=None, help="Override race_settings.total_laps for all phases.")
    parser.add_argument("--checkpoint-tag", default="", help="Optional model checkpoint tag for training phase artifacts.")
    parser.add_argument("--out", default="metrics/latest_candidate_metrics.json", help="Path to write output metrics JSON.")
    args = parser.parse_args()

    _disable_visualisations()

    base_config = _load_config((ROOT / args.config).resolve())
    competitors = base_config.get("competitors", [])
    dqn_cfg = base_config.get("dqn_params", {}) if isinstance(base_config.get("dqn_params", {}), dict) else {}
    algo_name = str(dqn_cfg.get("algo", "vanilla")).strip().lower() or "vanilla"
    algo_options = dqn_cfg.get("algo_options", {})
    if not isinstance(algo_options, dict):
        algo_options = {}

    dqn_driver_names = [
        c.get("name")
        for c in competitors
        if isinstance(c, dict) and str(c.get("agent", "")).lower() == "dqn"
    ]
    baseline_driver_names = [
        c.get("name")
        for c in competitors
        if isinstance(c, dict) and str(c.get("agent", "")).lower() == "base"
    ]
    random_driver_names = [
        c.get("name")
        for c in competitors
        if isinstance(c, dict) and str(c.get("agent", "")).lower() == "random"
    ]

    if not dqn_driver_names:
        raise ValueError("No DQN competitors found in config. Add at least one competitor with agent='dqn'.")

    if not baseline_driver_names:
        baseline_driver_names = random_driver_names.copy()

    if not baseline_driver_names:
        baseline_driver_names = [
            c.get("name")
            for c in competitors
            if isinstance(c, dict) and str(c.get("agent", "")).lower() != "dqn"
        ]

    if not baseline_driver_names:
        raise ValueError("No baseline opponents found. Add at least one non-DQN competitor (base or random).")

    if args.total_laps is not None:
        base_config.setdefault("race_settings", {})["total_laps"] = int(args.total_laps)

    total_laps = int(base_config.get("race_settings", {}).get("total_laps", 0) or 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_log_path = None
    train_records: List[Dict] = []
    if not args.skip_training:
        train_run_name = f"{args.run_prefix}_train_{timestamp}"
        train_checkpoint_tag = args.checkpoint_tag.strip() or train_run_name
        train_cfg = _prepare_phase_config(
            base_config,
            phase="training",
            runs=args.train_runs,
            run_name=train_run_name,
            total_laps=total_laps,
            checkpoint_tag=train_checkpoint_tag,
        )
        print(f"[evaluate_candidate] Training phase: runs={args.train_runs}, seed={args.train_seed}, run_name={train_run_name}")
        train_log_path, train_records = _run_phase(train_cfg, seed=args.train_seed, verbose=args.verbose)

    eval_seeds = _parse_seeds(args.eval_seeds, args.eval_seed)
    all_eval_records: List[Dict] = []
    eval_runs_meta: List[Dict] = []
    run_offset = 0

    for seed in eval_seeds:
        eval_run_name = f"{args.run_prefix}_eval_{timestamp}_s{seed}"
        eval_cfg = _prepare_phase_config(
            base_config,
            phase="evaluation",
            runs=args.eval_runs,
            run_name=eval_run_name,
            total_laps=total_laps,
        )
        print(f"[evaluate_candidate] Evaluation phase: runs={args.eval_runs}, seed={seed}, run_name={eval_run_name}")
        eval_log_path, eval_records = _run_phase(eval_cfg, seed=seed, verbose=args.verbose)

        remapped_records = _remap_run_numbers(eval_records, run_offset=run_offset)
        all_eval_records.extend(remapped_records)
        eval_runs_meta.append(
            {
                "seed": int(seed),
                "runs": int(args.eval_runs),
                "eval_log_path": str(eval_log_path),
            }
        )
        run_offset += int(args.eval_runs)

    eval_metrics = _compute_eval_metrics(
        all_eval_records,
        dqn_driver_names=dqn_driver_names,
        baseline_driver_names=baseline_driver_names,
        random_driver_names=random_driver_names,
        total_laps=total_laps,
    )

    seed_level_win_rates: List[float] = []
    for run_info in eval_runs_meta:
        seed_records = _read_jsonl_records(Path(run_info["eval_log_path"]))
        seed_metrics = _compute_eval_metrics(
            seed_records,
            dqn_driver_names=dqn_driver_names,
            baseline_driver_names=baseline_driver_names,
            random_driver_names=random_driver_names,
            total_laps=total_laps,
        )
        seed_level_win_rates.append(float(seed_metrics["win_rate_vs_baseline"]["mean"]))

    eval_metrics["stability"]["win_rate_vs_baseline_variance_across_seeds"] = float(
        np.var(np.array(seed_level_win_rates, dtype=float), ddof=1)
    ) if len(seed_level_win_rates) > 1 else 0.0

    total_eval_runs = int(eval_metrics.get("num_eval_runs", 0))
    baseline_ci_low = float(eval_metrics["win_rate_vs_baseline"].get("ci95_low", 0.0))
    random_ci_low = float(eval_metrics["win_rate_vs_random"].get("ci95_low", 0.0)) if random_driver_names else 0.0

    no_clear_benefit = (
        total_eval_runs >= int(args.guardrail_runs)
        and baseline_ci_low <= 0.5
        and (not random_driver_names or random_ci_low <= 0.5)
    )

    eval_metrics["guardrail"] = {
        "name": "no_clear_performance_benefit_after_guardrail_runs",
        "guardrail_runs": int(args.guardrail_runs),
        "triggered": bool(no_clear_benefit),
        "reason": (
            "No statistically clear benefit over baseline (and random when present)."
            if no_clear_benefit
            else "Guardrail not triggered."
        ),
    }

    train_summary = {
        "num_train_runs": len(_group_by_run(train_records)),
        "train_log_path": str(train_log_path) if train_log_path is not None else None,
    }

    out_payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": str((ROOT / args.config).resolve()),
        "algorithm": {
            "family": "dqn",
            "name": algo_name,
            "options": algo_options,
        },
        "objective_name": "win_rate_vs_baseline",
        "objective_score": float(eval_metrics["primary_objective"]["score"]),
        "dqn_driver_names": dqn_driver_names,
        "baseline_driver_names": baseline_driver_names,
        "random_driver_names": random_driver_names,
        "phases": {
            "training": {
                "skipped": bool(args.skip_training),
                "runs": int(args.train_runs),
                "seed": int(args.train_seed),
                **train_summary,
            },
            "evaluation": {
                "runs_per_seed": int(args.eval_runs),
                "seeds": [int(s) for s in eval_seeds],
                "seed_runs": eval_runs_meta,
            },
        },
        "metrics": eval_metrics,
    }

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    primary_score = float(eval_metrics["primary_objective"]["score"])
    print(f"[evaluate_candidate] Primary objective (win_rate_vs_baseline): {primary_score:.6f}")
    print(f"[evaluate_candidate] Metrics written to: {out_path}")


if __name__ == "__main__":
    main()
