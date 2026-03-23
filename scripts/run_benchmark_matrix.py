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
    "A": {
        "train_runs": 150,
        "train_seeds": [11, 22, 33],
        "eval_runs": 100,
        "eval_seeds": [101, 202, 303],
    },
    "B": {
        "train_runs": 300,
        "train_seeds": [11, 22, 33, 44, 55],
        "eval_runs": 200,
        "eval_seeds": [101, 202, 303, 404, 505],
    },
}

DEFAULT_ALGOS = ["vanilla", "double", "dueling", "rainbow_lite"]


def _ci95(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0, "ci95_low": 0.0, "ci95_high": 0.0}
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    margin = 1.96 * (std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "n": int(len(arr)),
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _default_algo_options(algo: str) -> Dict:
    if algo == "rainbow_lite":
        return {
            "n_step": 3,
            "per_alpha": 0.6,
            "per_beta_start": 0.4,
            "per_beta_frames": 100000,
        }
    return {}


def _render_markdown(summary: Dict) -> str:
    lines = []
    lines.append(f"# Benchmark Summary ({summary['stage']})")
    lines.append("")
    lines.append(f"- Created at: {summary['created_at']}")
    lines.append(f"- Base config: `{summary['base_config']}`")
    lines.append(f"- Train runs: {summary['effective_budget']['train_runs']}")
    lines.append(f"- Train seeds: {summary['effective_budget']['train_seeds']}")
    lines.append(f"- Eval runs per seed: {summary['effective_budget']['eval_runs']}")
    lines.append(f"- Eval seeds: {summary['effective_budget']['eval_seeds']}")
    if summary["effective_budget"].get("complexity_profile"):
        lines.append(f"- Eval complexity profile: {summary['effective_budget']['complexity_profile']}")
    lines.append("")
    lines.append("| Algo | Objective Mean | Objective CI95 | vs Vanilla | Avg Pos Delta | Overtake Success | DNF Rate |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for row in summary["aggregate"]:
        ci = row["objective"]
        ci_text = f"[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]"
        lines.append(
            f"| {row['algo']} | {ci['mean']:.4f} | {ci_text} | {row['vs_vanilla']} | "
            f"{row['avg_pos_delta']:.4f} | {row['overtake_success']:.4f} | {row['dnf_rate']:.4f} |"
        )
    lines.append("")
    lines.append(f"- Fairness audit passed: **{summary['fairness']['passed']}**")
    if summary["fairness"]["violations"]:
        lines.append("- Fairness violations:")
        for violation in summary["fairness"]["violations"]:
            lines.append(f"  - {violation}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run fair benchmark matrix for DQN variants.")
    parser.add_argument("--config", default="config.json", help="Base config path.")
    parser.add_argument("--stage", choices=["A", "B"], default="A", help="Benchmark stage budget.")
    parser.add_argument(
        "--algos",
        default=",".join(DEFAULT_ALGOS),
        help="Comma-separated algo list (subset of vanilla,double,dueling,rainbow_lite).",
    )
    parser.add_argument(
        "--out-dir",
        default="metrics/benchmarks",
        help="Directory for generated configs and results.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for running evaluator.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-trial JSON outputs.")
    parser.add_argument("--train-runs", type=int, default=None, help="Optional override for train runs.")
    parser.add_argument(
        "--train-seeds",
        default="",
        help="Optional comma-separated train seeds override.",
    )
    parser.add_argument("--eval-runs", type=int, default=None, help="Optional override for eval runs per seed.")
    parser.add_argument(
        "--eval-seeds",
        default="",
        help="Optional comma-separated eval seeds override.",
    )
    parser.add_argument(
        "--complexity-profile",
        default="",
        help="Optional complexity profile override. Only 'low' is implemented.",
    )
    args = parser.parse_args()

    stage_cfg = STAGE_BUDGETS[args.stage]
    train_runs = int(args.train_runs if args.train_runs is not None else stage_cfg["train_runs"])
    train_seeds = (
        [int(s.strip()) for s in args.train_seeds.split(",") if s.strip()]
        if args.train_seeds.strip()
        else [int(s) for s in stage_cfg["train_seeds"]]
    )
    eval_runs = int(args.eval_runs if args.eval_runs is not None else stage_cfg["eval_runs"])
    eval_seeds = (
        [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()]
        if args.eval_seeds.strip()
        else [int(s) for s in stage_cfg["eval_seeds"]]
    )
    eval_seed_csv = ",".join(str(s) for s in eval_seeds)
    complexity_profile = args.complexity_profile.strip().lower() if args.complexity_profile.strip() else ""

    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    invalid = [a for a in algos if a not in DEFAULT_ALGOS]
    if invalid:
        raise ValueError(f"Unsupported algos: {invalid}. Allowed: {DEFAULT_ALGOS}")
    if "vanilla" not in algos:
        raise ValueError("Please include 'vanilla' to keep the control arm in the benchmark.")

    base_config_path = (ROOT / args.config).resolve()
    base_config = _load_json(base_config_path)
    if "dqn_params" not in base_config or not isinstance(base_config["dqn_params"], dict):
        raise ValueError("Base config must include a valid 'dqn_params' object.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (ROOT / args.out_dir / f"stage_{args.stage}_{timestamp}").resolve()
    config_dir = out_dir / "configs"
    raw_dir = out_dir / "raw"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    trial_records = []

    for algo in algos:
        for train_seed in train_seeds:
            trial_id = f"{args.stage}_{algo}_ts{train_seed}"
            trial_cfg = copy.deepcopy(base_config)
            trial_cfg.setdefault("dqn_params", {})
            trial_cfg["dqn_params"]["algo"] = algo
            trial_cfg["dqn_params"]["algo_options"] = _default_algo_options(algo)

            cfg_path = config_dir / f"{trial_id}.json"
            out_path = raw_dir / f"{trial_id}.json"
            _write_json(cfg_path, trial_cfg)

            if out_path.exists() and args.skip_existing:
                print(f"[benchmark] Reusing existing output: {out_path}")
            else:
                cmd = [
                    args.python,
                    str(EVAL_SCRIPT),
                    "--config",
                    str(cfg_path),
                    "--train-runs",
                    str(train_runs),
                    "--train-seed",
                    str(train_seed),
                    "--eval-runs",
                    str(eval_runs),
                    "--eval-seeds",
                    eval_seed_csv,
                    "--run-prefix",
                    f"bench_{trial_id}",
                    "--out",
                    str(out_path),
                ]
                if complexity_profile:
                    cmd.extend(["--complexity-profile", complexity_profile])
                print(f"[benchmark] Running {trial_id}")
                result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if result.returncode != 0:
                    print(result.stdout)
                    print(result.stderr)
                    raise RuntimeError(f"Benchmark trial failed: {trial_id}")

            payload = _load_json(out_path)
            metrics = payload.get("metrics", {})
            trial_records.append(
                {
                    "trial_id": trial_id,
                    "algo": algo,
                    "train_seed": train_seed,
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
                    "dnf_rate": float(
                        metrics.get("stability", {})
                        .get("dnf_rate_dqn", {})
                        .get("rate", 0.0)
                    ),
                    "phases": payload.get("phases", {}),
                }
            )

    grouped = defaultdict(list)
    for record in trial_records:
        grouped[record["algo"]].append(record)

    aggregate_rows = []
    vanilla_ci = _ci95([r["objective_score"] for r in grouped["vanilla"]])

    for algo in algos:
        rows = grouped[algo]
        objective_ci = _ci95([r["objective_score"] for r in rows])
        avg_pos_delta = float(np.mean([r["avg_pos_delta"] for r in rows])) if rows else 0.0
        overtake_success = float(np.mean([r["overtake_success"] for r in rows])) if rows else 0.0
        dnf_rate = float(np.mean([r["dnf_rate"] for r in rows])) if rows else 0.0

        if algo == "vanilla":
            relation = "control"
        elif objective_ci["ci95_low"] > vanilla_ci["ci95_high"]:
            relation = "better"
        elif objective_ci["ci95_high"] < vanilla_ci["ci95_low"]:
            relation = "worse"
        else:
            relation = "overlap"

        aggregate_rows.append(
            {
                "algo": algo,
                "objective": objective_ci,
                "vs_vanilla": relation,
                "avg_pos_delta": avg_pos_delta,
                "overtake_success": overtake_success,
                "dnf_rate": dnf_rate,
                "num_trials": len(rows),
            }
        )

    fairness_violations = []
    for rec in trial_records:
        phases = rec.get("phases", {})
        train = phases.get("training", {})
        evaluation = phases.get("evaluation", {})
        expected_seed = rec["train_seed"]
        if int(train.get("runs", -1)) != train_runs:
            fairness_violations.append(f"{rec['trial_id']}: training runs mismatch")
        if int(train.get("seed", -1)) != expected_seed:
            fairness_violations.append(f"{rec['trial_id']}: training seed mismatch")
        if int(evaluation.get("runs_per_seed", -1)) != eval_runs:
            fairness_violations.append(f"{rec['trial_id']}: eval runs mismatch")
        observed_eval_seeds = [int(s) for s in evaluation.get("seeds", [])]
        if observed_eval_seeds != eval_seeds:
            fairness_violations.append(f"{rec['trial_id']}: eval seeds mismatch")
        if complexity_profile:
            observed_profile = str(evaluation.get("complexity_profile", ""))
            if observed_profile != complexity_profile:
                fairness_violations.append(f"{rec['trial_id']}: eval complexity profile mismatch")

    summary = {
        "created_at": datetime.now().isoformat(),
        "stage": args.stage,
        "base_config": str(base_config_path),
        "budget": stage_cfg,
        "effective_budget": {
            "train_runs": train_runs,
            "train_seeds": train_seeds,
            "eval_runs": eval_runs,
            "eval_seeds": eval_seeds,
            "complexity_profile": complexity_profile,
        },
        "algorithms": algos,
        "aggregate": aggregate_rows,
        "trials": trial_records,
        "fairness": {
            "passed": len(fairness_violations) == 0,
            "violations": fairness_violations,
        },
    }

    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    _write_json(summary_json, summary)
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")

    print(f"[benchmark] Summary JSON: {summary_json}")
    print(f"[benchmark] Summary Markdown: {summary_md}")
    print(f"[benchmark] Fairness audit passed: {summary['fairness']['passed']}")


if __name__ == "__main__":
    main()
