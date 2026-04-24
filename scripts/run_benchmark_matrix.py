import argparse
import copy
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_candidate.py"

STAGE_BUDGETS = {
    "A": {"train_runs": 150, "train_seeds": [11, 22, 33], "eval_runs": 100, "eval_seeds": [101, 202, 303]},
    "B": {"train_runs": 300, "train_seeds": [11, 22, 33, 44, 55], "eval_runs": 200, "eval_seeds": [101, 202, 303, 404, 505]},
}

DEFAULT_ALGOS = ["vanilla", "double", "dueling", "rainbow_lite"]
DEFAULT_PROTOCOL_FALLBACK = {
    "seed_sets": {
        "smoke": [101, 202, 303],
        "candidate": [101, 202, 303, 404, 505],
        "benchmark": [101, 202, 303, 404, 505],
    },
    "train_runs": {"low": 200, "medium": 200, "high": 200},
    "eval_runs": {"low": 200, "medium": 200, "high": 200},
    "comparison_matrix": {"algorithms": DEFAULT_ALGOS},
}
DEFAULT_BENCHMARK_CONTRACT = {
    "lock_enabled": True,
    "immutable_comparison_matrix": True,
    "immutable_seed_sets": True,
    "immutable_budgets": True,
    "reproducibility_reruns": 2,
    "tracks": {
        "low_primary": {
            "enabled": True,
            "complexity": "low",
            "stochasticity_levels": ["s0"],
            "is_primary_ranking_track": True,
        },
        "low_robustness": {
            "enabled": True,
            "complexity": "low",
            "stochasticity_levels": ["s1", "s2"],
            "is_primary_ranking_track": False,
        },
    },
    "promotion_gate": {
        "require_fairness_audit": True,
        "require_track_a_ci_stability": True,
        "require_reproducibility_consistency": True,
        "require_track_b_robustness_evidence": True,
    },
}


def _ci95(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0, "ci95_low": 0.0, "ci95_high": 0.0}
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    margin = 1.96 * (std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std, "n": int(len(arr)), "ci95_low": mean - margin, "ci95_high": mean + margin}


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_protocol_contract(config: Dict) -> Dict:
    user_protocol = config.get("protocol", {})
    return _deep_merge(DEFAULT_PROTOCOL_FALLBACK, user_protocol if isinstance(user_protocol, dict) else {})


def _resolve_benchmark_contract(protocol_contract: Dict) -> Dict:
    user_contract = protocol_contract.get("benchmark_contract", {})
    return _deep_merge(
        DEFAULT_BENCHMARK_CONTRACT,
        user_contract if isinstance(user_contract, dict) else {},
    )


def _validate_benchmark_contract(contract: Dict) -> None:
    tracks = contract.get("tracks", {})
    if not isinstance(tracks, dict) or not tracks:
        raise ValueError("protocol.benchmark_contract.tracks must be a non-empty object")
    primary_count = 0
    for track_name, track_cfg in tracks.items():
        if not isinstance(track_cfg, dict):
            raise TypeError(f"protocol.benchmark_contract.tracks.{track_name} must be an object")
        levels = track_cfg.get("stochasticity_levels", [])
        if not isinstance(levels, list) or not levels:
            raise ValueError(
                f"protocol.benchmark_contract.tracks.{track_name}.stochasticity_levels must be a non-empty array"
            )
        if bool(track_cfg.get("is_primary_ranking_track", False)):
            primary_count += 1
    if primary_count != 1:
        raise ValueError("Exactly one benchmark track must set is_primary_ranking_track=true")


def _validate_base_config_contract(config: Dict) -> None:
    required_top = ["reward", "feedback", "protocol", "stochasticity", "dqn_params"]
    for key in required_top:
        if key not in config or not isinstance(config.get(key), dict):
            raise ValueError(f"config.{key} must exist and be an object")

    required_components = {"outcome", "persistent_position", "tactical", "pace", "tyre_pit", "penalty"}
    reward_cfg = config["reward"]
    weights = reward_cfg.get("weights", {})
    if not isinstance(weights, dict) or set(weights.keys()) != required_components:
        raise ValueError("config.reward.weights must contain exactly: outcome,persistent_position,tactical,pace,tyre_pit,penalty")

    normalization_low = reward_cfg.get("normalization", {}).get("low", {})
    if not isinstance(normalization_low, dict) or set(normalization_low.keys()) != required_components:
        raise ValueError("config.reward.normalization.low must define all reward components")

    activation_low = reward_cfg.get("component_activation_by_complexity", {}).get("low", {})
    if not isinstance(activation_low, dict) or set(activation_low.keys()) != required_components:
        raise ValueError("config.reward.component_activation_by_complexity.low must define all reward components")

    feedback_low = config["feedback"].get("features_by_complexity", {}).get("low", [])
    if not isinstance(feedback_low, list) or not feedback_low:
        raise ValueError("config.feedback.features_by_complexity.low must be a non-empty array")

    levels = config["stochasticity"].get("levels", {})
    if not isinstance(levels, dict):
        raise ValueError("config.stochasticity.levels must be an object")
    for level in ("s0", "s1", "s2"):
        if level not in levels:
            raise ValueError("config.stochasticity.levels must include s0, s1, and s2")

    algorithms = config["protocol"].get("comparison_matrix", {}).get("algorithms", [])
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("config.protocol.comparison_matrix.algorithms must be a non-empty array")
    unknown_algorithms = [a for a in algorithms if str(a).strip().lower() not in DEFAULT_ALGOS]
    if unknown_algorithms:
        raise ValueError(f"Unsupported algorithm(s) in protocol comparison matrix: {unknown_algorithms}")


def _default_algo_options(algo: str) -> Dict:
    if algo == "rainbow_lite":
        return {"n_step": 3, "per_alpha": 0.6, "per_beta_start": 0.4, "per_beta_frames": 100000}
    return {}


def _build_trial_record(payload: Dict, trial_id: str, algo: str, train_seed: int) -> Dict:
    metrics = payload.get("metrics", {})
    reward_diag = metrics.get("reward_diagnostics", {})
    reward_means = reward_diag.get("reward_component_mean_per_driver_run_weighted", {})
    reward_means_flat: Dict[str, float] = {}
    if isinstance(reward_means, dict):
        for component, info in reward_means.items():
            if isinstance(info, dict):
                reward_means_flat[component] = float(info.get("mean", 0.0))
    parity_ratio = (
        metrics.get("fairness_diagnostics", {})
        .get("starting_position_exposure", {})
        .get("imbalance_ratio", 0.0)
    )
    return {
        "trial_id": trial_id,
        "algo": algo,
        "train_seed": int(train_seed),
        "objective_score": float(payload.get("objective_score", 0.0)),
        "avg_pos_delta": float(
            metrics.get("race_quality", {})
            .get("avg_position_delta_non_dqn_minus_dqn", {})
            .get("mean", 0.0)
        ),
        "overtake_success": float(
            metrics.get("tactical", {})
            .get("overtake_success_rate", {})
            .get("rate", 0.0)
        ),
        "dnf_rate": float(metrics.get("stability", {}).get("dnf_rate_dqn", {}).get("rate", 0.0)),
        "win_rate_variance_across_eval_seeds": float(
            metrics.get("stability", {}).get("win_rate_vs_baseline_variance_across_seeds", 0.0)
        ),
        "reward_component_means_weighted": reward_means_flat,
        "starting_position_imbalance_ratio": float(parity_ratio),
        "phases": payload.get("phases", {}),
    }


def _aggregate_rows(trial_records: List[Dict], algos: List[str]) -> List[Dict]:
    grouped = defaultdict(list)
    for record in trial_records:
        grouped[record["algo"]].append(record)

    vanilla_ci = _ci95([r["objective_score"] for r in grouped["vanilla"]])
    rows = []
    for algo in algos:
        records = grouped.get(algo, [])
        objective_ci = _ci95([r["objective_score"] for r in records])
        avg_pos_delta = float(np.mean([r["avg_pos_delta"] for r in records])) if records else 0.0
        overtake_success = float(np.mean([r["overtake_success"] for r in records])) if records else 0.0
        dnf_rate = float(np.mean([r["dnf_rate"] for r in records])) if records else 0.0
        seed_variance = float(np.mean([r["win_rate_variance_across_eval_seeds"] for r in records])) if records else 0.0
        start_parity_imbalance = float(np.mean([r["starting_position_imbalance_ratio"] for r in records])) if records else 0.0
        if algo == "vanilla":
            relation = "control"
        elif objective_ci["ci95_low"] > vanilla_ci["ci95_high"]:
            relation = "better"
        elif objective_ci["ci95_high"] < vanilla_ci["ci95_low"]:
            relation = "worse"
        else:
            relation = "overlap"
        rows.append(
            {
                "algo": algo,
                "objective": objective_ci,
                "vs_vanilla": relation,
                "avg_pos_delta": avg_pos_delta,
                "overtake_success": overtake_success,
                "dnf_rate": dnf_rate,
                "seed_variance": seed_variance,
                "starting_position_imbalance_ratio": start_parity_imbalance,
                "num_trials": len(records),
            }
        )
    return rows


def _compute_ranking(rows: List[Dict]) -> Dict:
    better = [row for row in rows if row.get("vs_vanilla") == "better"]
    if not better:
        overlap = [row for row in rows if row.get("algo") != "vanilla" and row.get("vs_vanilla") == "overlap"]
        if overlap:
            return {"classification": "overlap", "winner_algo": None}
        return {"classification": "inconclusive", "winner_algo": None}
    winner = max(better, key=lambda row: float(row.get("objective", {}).get("mean", 0.0)))
    return {"classification": "winner", "winner_algo": winner["algo"]}


def _audit_fairness(
    trial_records: List[Dict],
    train_runs: int,
    eval_runs: int,
    eval_seeds: List[int],
    complexity_profile: str,
    stochasticity_level: str,
) -> Dict:
    violations: List[str] = []
    for rec in trial_records:
        phases = rec.get("phases", {})
        train = phases.get("training", {})
        evaluation = phases.get("evaluation", {})
        if int(train.get("runs", -1)) != train_runs:
            violations.append(f"{rec['trial_id']}: training runs mismatch")
        if int(train.get("seed", -1)) != int(rec["train_seed"]):
            violations.append(f"{rec['trial_id']}: training seed mismatch")
        if int(evaluation.get("runs_per_seed", -1)) != eval_runs:
            violations.append(f"{rec['trial_id']}: eval runs mismatch")
        if [int(s) for s in evaluation.get("seeds", [])] != eval_seeds:
            violations.append(f"{rec['trial_id']}: eval seeds mismatch")
        if str(evaluation.get("complexity_profile", "")) != complexity_profile:
            violations.append(f"{rec['trial_id']}: eval complexity profile mismatch")
        if str(evaluation.get("stochasticity_level", "")) != stochasticity_level:
            violations.append(f"{rec['trial_id']}: eval stochasticity level mismatch")
    return {"passed": len(violations) == 0, "violations": violations}


def _render_markdown(summary: Dict) -> str:
    lines: List[str] = []
    lines.append(f"# Benchmark Summary ({summary['stage']})")
    lines.append("")
    lines.append(f"- Created at: {summary['created_at']}")
    lines.append(f"- Base config: `{summary['base_config']}`")
    lines.append(f"- Fairness audit (all cells): **{summary['overall_fairness']['passed']}**")
    lines.append(f"- Promotion gate passed: **{summary['promotion_gate']['passed']}**")
    lines.append("")
    for cell in summary.get("cells", []):
        lines.append(f"## {cell['track']} | {cell['stochasticity_level']} | rerun {cell['rerun_index']}")
        lines.append(f"- Fairness passed: **{cell['fairness']['passed']}**")
        lines.append("| Algo | Objective Mean | Objective CI95 | vs Vanilla | Avg Pos Delta | Overtake Success | DNF Rate | Seed Variance | Start Imbalance |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
        for row in cell.get("aggregate", []):
            ci = row["objective"]
            ci_text = f"[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]"
            lines.append(
                f"| {row['algo']} | {ci['mean']:.4f} | {ci_text} | {row['vs_vanilla']} | "
                f"{row['avg_pos_delta']:.4f} | {row['overtake_success']:.4f} | {row['dnf_rate']:.4f} | "
                f"{row['seed_variance']:.6f} | {row['starting_position_imbalance_ratio']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _compute_robustness_analysis(cell_summaries: List[Dict], algos: List[str]) -> Dict:
    by_level = {}
    for cell in cell_summaries:
        if int(cell.get("rerun_index", 0)) != 0:
            continue
        level = str(cell.get("stochasticity_level", ""))
        by_level[level] = cell

    if "s0" not in by_level or "s1" not in by_level or "s2" not in by_level:
        return {
            "available": False,
            "missing_levels": [lvl for lvl in ("s0", "s1", "s2") if lvl not in by_level],
            "per_algo": {},
        }

    def _trials_for(level: str, algo: str) -> List[Dict]:
        return [r for r in by_level[level].get("trials", []) if r.get("algo") == algo]

    per_algo: Dict[str, Dict] = {}
    for algo in algos:
        s0_trials = _trials_for("s0", algo)
        s1_trials = _trials_for("s1", algo)
        s2_trials = _trials_for("s2", algo)

        def _mean_key(trials: List[Dict], key: str) -> float:
            vals = [float(t.get(key, 0.0)) for t in trials]
            return float(np.mean(vals)) if vals else 0.0

        s0_obj = _mean_key(s0_trials, "objective_score")
        s1_obj = _mean_key(s1_trials, "objective_score")
        s2_obj = _mean_key(s2_trials, "objective_score")
        s0_var = _mean_key(s0_trials, "win_rate_variance_across_eval_seeds")
        s1_var = _mean_key(s1_trials, "win_rate_variance_across_eval_seeds")
        s2_var = _mean_key(s2_trials, "win_rate_variance_across_eval_seeds")
        s0_parity = _mean_key(s0_trials, "starting_position_imbalance_ratio")
        s1_parity = _mean_key(s1_trials, "starting_position_imbalance_ratio")
        s2_parity = _mean_key(s2_trials, "starting_position_imbalance_ratio")

        components = sorted(
            set().union(
                *[
                    set((trial.get("reward_component_means_weighted", {}) or {}).keys())
                    for trial in (s0_trials + s1_trials + s2_trials)
                ]
            )
        )

        def _comp_mean(trials: List[Dict], component: str) -> float:
            if not trials:
                return 0.0
            vals = [
                float((trial.get("reward_component_means_weighted", {}) or {}).get(component, 0.0))
                for trial in trials
            ]
            return float(np.mean(vals)) if vals else 0.0

        reward_delta_s1: List[float] = []
        reward_delta_s2: List[float] = []
        for component in components:
            c0 = _comp_mean(s0_trials, component)
            c1 = _comp_mean(s1_trials, component)
            c2 = _comp_mean(s2_trials, component)
            reward_delta_s1.append(abs(c1 - c0))
            reward_delta_s2.append(abs(c2 - c0))

        per_algo[algo] = {
            "objective_mean": {"s0": s0_obj, "s1": s1_obj, "s2": s2_obj},
            "objective_drop_from_s0": {"s1": s1_obj - s0_obj, "s2": s2_obj - s0_obj},
            "seed_variance_trend": {"s0": s0_var, "s1": s1_var, "s2": s2_var},
            "reward_component_stability_avg_abs_delta_from_s0": {
                "s1": float(np.mean(reward_delta_s1)) if reward_delta_s1 else 0.0,
                "s2": float(np.mean(reward_delta_s2)) if reward_delta_s2 else 0.0,
            },
            "starting_position_parity_imbalance_ratio": {
                "s0": s0_parity,
                "s1": s1_parity,
                "s2": s2_parity,
            },
        }

    return {"available": True, "missing_levels": [], "per_algo": per_algo}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standardized low-complexity benchmark tracks for DQN variants.")
    parser.add_argument("--config", default="config.json", help="Base config path.")
    parser.add_argument(
        "--stage",
        choices=["A", "B", "smoke", "candidate", "benchmark"],
        default="candidate",
        help="Benchmark stage budget preset. 'A/B' are legacy aliases.",
    )
    parser.add_argument(
        "--track",
        choices=["all", "low_primary", "low_robustness"],
        default="all",
        help="Run one benchmark track or all enabled tracks.",
    )
    parser.add_argument("--algos", default="", help="Optional comma-separated algo override.")
    parser.add_argument("--out-dir", default="metrics/benchmarks", help="Output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for evaluator.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-trial JSON outputs.")
    parser.add_argument("--train-runs", type=int, default=None, help="Optional train-runs override.")
    parser.add_argument("--eval-runs", type=int, default=None, help="Optional eval-runs override.")
    parser.add_argument("--train-seeds", default="", help="Optional comma-separated train seeds override.")
    parser.add_argument("--eval-seeds", default="", help="Optional comma-separated eval seeds override.")
    parser.add_argument("--repro-reruns", type=int, default=None, help="Optional reproducibility reruns override.")
    args = parser.parse_args()

    stage_alias = {"A": "smoke", "B": "benchmark"}
    stage_name = stage_alias.get(args.stage, args.stage)
    fallback_stage_key = "B" if stage_name == "benchmark" else "A"
    legacy_stage_cfg = STAGE_BUDGETS.get(args.stage, STAGE_BUDGETS.get(fallback_stage_key))

    base_config_path = (ROOT / args.config).resolve()
    base_config = _load_json(base_config_path)
    _validate_base_config_contract(base_config)

    protocol_contract = _resolve_protocol_contract(base_config)
    benchmark_contract = _resolve_benchmark_contract(protocol_contract)
    _validate_benchmark_contract(benchmark_contract)

    default_algos = protocol_contract.get("comparison_matrix", {}).get("algorithms", DEFAULT_ALGOS)
    if not isinstance(default_algos, list) or not default_algos:
        default_algos = DEFAULT_ALGOS

    lock_enabled = bool(benchmark_contract.get("lock_enabled", True))
    if lock_enabled and bool(benchmark_contract.get("immutable_comparison_matrix", True)) and args.algos.strip():
        raise ValueError("Benchmark contract lock enabled: --algos override is disabled.")
    if lock_enabled and bool(benchmark_contract.get("immutable_seed_sets", True)):
        if args.train_seeds.strip() or args.eval_seeds.strip():
            raise ValueError("Benchmark contract lock enabled: seed overrides are disabled.")
    if lock_enabled and bool(benchmark_contract.get("immutable_budgets", True)):
        if args.train_runs is not None or args.eval_runs is not None:
            raise ValueError("Benchmark contract lock enabled: run-budget overrides are disabled.")

    protocol_seed_set = protocol_contract.get("seed_sets", {}).get(stage_name, [])
    if not isinstance(protocol_seed_set, list) or not protocol_seed_set:
        protocol_seed_set = []
    protocol_train_runs = int(protocol_contract.get("train_runs", {}).get("low", DEFAULT_PROTOCOL_FALLBACK["train_runs"]["low"]))
    protocol_eval_runs = int(protocol_contract.get("eval_runs", {}).get("low", DEFAULT_PROTOCOL_FALLBACK["eval_runs"]["low"]))
    train_runs = int(args.train_runs if args.train_runs is not None else (protocol_train_runs if protocol_train_runs > 0 else int(legacy_stage_cfg["train_runs"])))
    eval_runs = int(args.eval_runs if args.eval_runs is not None else (protocol_eval_runs if protocol_eval_runs > 0 else int(legacy_stage_cfg["eval_runs"])))
    default_train_seeds = [int(s) for s in (protocol_seed_set or legacy_stage_cfg["train_seeds"])]
    default_eval_seeds = [int(s) for s in (protocol_seed_set or legacy_stage_cfg["eval_seeds"])]
    train_seeds = [int(s.strip()) for s in args.train_seeds.split(",") if s.strip()] if args.train_seeds.strip() else default_train_seeds
    eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()] if args.eval_seeds.strip() else default_eval_seeds
    eval_seed_csv = ",".join(str(s) for s in eval_seeds)

    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    if not algos:
        algos = [str(a).strip().lower() for a in default_algos if str(a).strip()]
    invalid_algos = [a for a in algos if a not in DEFAULT_ALGOS]
    if invalid_algos:
        raise ValueError(f"Unsupported algos: {invalid_algos}. Allowed: {DEFAULT_ALGOS}")
    if "vanilla" not in algos:
        raise ValueError("Please include 'vanilla' as control arm.")

    repro_reruns = int(args.repro_reruns if args.repro_reruns is not None else int(benchmark_contract.get("reproducibility_reruns", 2)))
    repro_reruns = max(1, repro_reruns)

    tracks_cfg = benchmark_contract.get("tracks", {})
    if args.track == "all":
        selected_tracks = [name for name, cfg in tracks_cfg.items() if bool(cfg.get("enabled", True))]
    else:
        selected_tracks = [args.track]
        if args.track not in tracks_cfg:
            raise ValueError(f"Requested track '{args.track}' missing in benchmark contract.")
        if not bool(tracks_cfg[args.track].get("enabled", True)):
            raise ValueError(f"Requested track '{args.track}' is disabled.")
    if not selected_tracks:
        raise ValueError("No benchmark tracks selected.")

    primary_track_name = next(
        (name for name, cfg in tracks_cfg.items() if bool(cfg.get("is_primary_ranking_track", False))),
        "low_primary",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (ROOT / args.out_dir / f"stage_{args.stage}_{timestamp}").resolve()
    config_dir = out_dir / "configs"
    raw_dir = out_dir / "raw"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_trial_records: List[Dict] = []
    cell_summaries: List[Dict] = []

    for track_name in selected_tracks:
        track_cfg = tracks_cfg[track_name]
        complexity_profile = str(track_cfg.get("complexity", "low")).strip().lower() or "low"
        levels = [str(level).strip() for level in track_cfg.get("stochasticity_levels", []) if str(level).strip()]

        for rerun_index in range(repro_reruns):
            for stochasticity_level in levels:
                trial_records: List[Dict] = []
                for algo in algos:
                    for train_seed in train_seeds:
                        trial_id = f"{args.stage}_{track_name}_{stochasticity_level}_r{rerun_index}_{algo}_ts{train_seed}"
                        trial_cfg = copy.deepcopy(base_config)
                        trial_cfg.setdefault("dqn_params", {})
                        trial_cfg["dqn_params"]["algo"] = algo
                        trial_cfg["dqn_params"]["algo_options"] = _default_algo_options(algo)
                        trial_cfg.setdefault("complexity", {})["active_profile"] = complexity_profile
                        trial_cfg.setdefault("stochasticity", {})["active_level"] = stochasticity_level

                        cfg_path = config_dir / track_name / stochasticity_level / f"rerun_{rerun_index}" / f"{trial_id}.json"
                        out_path = raw_dir / track_name / stochasticity_level / f"rerun_{rerun_index}" / f"{trial_id}.json"
                        _write_json(cfg_path, trial_cfg)

                        if out_path.exists() and args.skip_existing:
                            print(f"[benchmark] Reusing existing output: {out_path}")
                        else:
                            cmd = [
                                args.python, str(EVAL_SCRIPT),
                                "--config", str(cfg_path),
                                "--train-runs", str(train_runs),
                                "--train-seed", str(train_seed),
                                "--eval-runs", str(eval_runs),
                                "--eval-seeds", eval_seed_csv,
                                "--run-prefix", f"bench_{trial_id}",
                                "--out", str(out_path),
                                "--complexity-profile", complexity_profile,
                                "--stochasticity-level", stochasticity_level,
                            ]
                            print(f"[benchmark] Running {trial_id}")
                            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                            if result.returncode != 0:
                                print(result.stdout)
                                print(result.stderr)
                                raise RuntimeError(f"Benchmark trial failed: {trial_id}")

                        payload = _load_json(out_path)
                        record = _build_trial_record(payload=payload, trial_id=trial_id, algo=algo, train_seed=train_seed)
                        trial_records.append(record)
                        all_trial_records.append({**record, "track": track_name, "stochasticity_level": stochasticity_level, "complexity_profile": complexity_profile, "rerun_index": rerun_index})

                fairness = _audit_fairness(
                    trial_records=trial_records,
                    train_runs=train_runs,
                    eval_runs=eval_runs,
                    eval_seeds=eval_seeds,
                    complexity_profile=complexity_profile,
                    stochasticity_level=stochasticity_level,
                )
                aggregate_rows = _aggregate_rows(trial_records, algos=algos)
                ranking = _compute_ranking(aggregate_rows)
                cell_summaries.append(
                    {
                        "track": track_name,
                        "complexity_profile": complexity_profile,
                        "stochasticity_level": stochasticity_level,
                        "rerun_index": rerun_index,
                        "effective_budget": {
                            "train_runs": train_runs,
                            "train_seeds": train_seeds,
                            "eval_runs": eval_runs,
                            "eval_seeds": eval_seeds,
                        },
                        "aggregate": aggregate_rows,
                        "ranking": ranking,
                        "fairness": fairness,
                        "trials": trial_records,
                    }
                )

    overall_fairness_passed = all(bool(cell.get("fairness", {}).get("passed", False)) for cell in cell_summaries)
    overall_fairness_violations: List[str] = []
    for cell in cell_summaries:
        for violation in cell.get("fairness", {}).get("violations", []):
            overall_fairness_violations.append(f"{cell['track']}|{cell['stochasticity_level']}|r{cell['rerun_index']}: {violation}")

    primary_cells = sorted(
        [cell for cell in cell_summaries if cell.get("track") == primary_track_name and str(cell.get("stochasticity_level")) == "s0"],
        key=lambda cell: int(cell.get("rerun_index", 0)),
    )
    if primary_cells:
        first = primary_cells[0].get("ranking", {})
        stable = all(
            cell.get("ranking", {}).get("classification") == first.get("classification")
            and cell.get("ranking", {}).get("winner_algo") == first.get("winner_algo")
            for cell in primary_cells[1:]
        )
        primary_consistent = bool(stable)
    else:
        primary_consistent = False

    reproducibility = {
        "required_reruns": repro_reruns,
        "observed_primary_cells": len(primary_cells),
        "track_a_ci_stability": primary_consistent,
        "consistent": primary_consistent,
        "ranking_classes": [cell.get("ranking", {}).get("classification", "inconclusive") for cell in primary_cells],
        "winner_algorithms": [cell.get("ranking", {}).get("winner_algo") for cell in primary_cells],
    }

    robustness_analysis = _compute_robustness_analysis(
        cell_summaries,
        algos=algos,
    )

    gate_cfg = benchmark_contract.get("promotion_gate", {})
    gate_checks = {
        "fairness_audit_pass": overall_fairness_passed,
        "track_a_ci_stability": bool(reproducibility.get("track_a_ci_stability", False)),
        "reproducibility_consistency": bool(reproducibility.get("consistent", False)),
        "track_b_robustness_evidence": bool(robustness_analysis.get("available", False)),
    }
    required_map = {
        "fairness_audit_pass": bool(gate_cfg.get("require_fairness_audit", True)),
        "track_a_ci_stability": bool(gate_cfg.get("require_track_a_ci_stability", True)),
        "reproducibility_consistency": bool(gate_cfg.get("require_reproducibility_consistency", True)),
        "track_b_robustness_evidence": bool(gate_cfg.get("require_track_b_robustness_evidence", True)),
    }
    gate_passed = all((not required_map[key]) or gate_checks.get(key, False) for key in required_map.keys())

    summary = {
        "created_at": datetime.now().isoformat(),
        "stage": args.stage,
        "stage_name": stage_name,
        "base_config": str(base_config_path),
        "protocol": protocol_contract,
        "benchmark_contract": benchmark_contract,
        "effective_budget": {
            "train_runs": train_runs,
            "train_seeds": train_seeds,
            "eval_runs": eval_runs,
            "eval_seeds": eval_seeds,
            "algorithms": algos,
            "tracks": selected_tracks,
            "reproducibility_reruns": repro_reruns,
        },
        "cells": cell_summaries,
        "overall_fairness": {"passed": overall_fairness_passed, "violations": overall_fairness_violations, "mandatory": True},
        "reproducibility": reproducibility,
        "robustness_analysis": robustness_analysis,
        "promotion_gate": {"checks": gate_checks, "required": required_map, "passed": gate_passed},
        "trials": all_trial_records,
    }

    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    _write_json(summary_json, summary)
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")

    print(f"[benchmark] Summary JSON: {summary_json}")
    print(f"[benchmark] Summary Markdown: {summary_md}")
    print(f"[benchmark] Fairness audit passed: {summary['overall_fairness']['passed']}")
    print(f"[benchmark] Promotion gate passed: {summary['promotion_gate']['passed']}")


if __name__ == "__main__":
    main()
