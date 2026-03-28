"""
Cross-play evaluation script for Phase 3 MARL investigations.

Trains two concurrent DQN agents, then evaluates under three conditions:
  - standard:    agents play with their trained identities (normal round-robin starts)
  - a1_carries:  A1's trained model is loaded for BOTH agent slots (A2 is replaced)
  - a2_carries:  A2's trained model is loaded for BOTH agent slots (A1 is replaced)
  - swapped:     A1 loads A2's trained model and vice versa (position-swap test)

The "swapped" condition is the key diagnostic for the R1 hypothesis:
  If A2 has a genuine policy advantage, the swapped A2-model (now labelled A1) should
  still win more often. If the advantage was purely positional/training-artefact, the
  win rates should equalise when models are swapped across naming slots.

Usage:
  conda run -n f1StrategySim python scripts/evaluate_crossplay.py \
    --config metrics/phase3/config_rainbow_marl.json \
    --train-runs 500 --eval-runs 150 \
    --train-seed 101 \
    --stochasticity-level s2 \
    --out metrics/phase3/crossplay_rainbow_s2_s101.json
"""

import argparse
import copy
import io
import json
import os
import random
import shutil
import sys
from contextlib import redirect_stdout
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

from src.simulator import RaceSimulator, init_simulator
from src.states import init_race_state
from src.track import load_track
from runtime_profiles import resolve_complexity_profile


# ---------------------------------------------------------------------------
# Utilities (replicated from evaluate_marl.py to keep script self-contained)
# ---------------------------------------------------------------------------

def _disable_visualisations() -> None:
    def _noop(*args, **kwargs):
        return None
    RaceSimulator._visualise_results = _noop
    RaceSimulator._visualise_agent_learning = _noop


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: List[float]) -> float:
    return float(np.std(np.array(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def _ci95(values: List[float]) -> Dict:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = _mean(values)
    std = _std(values)
    margin = 1.96 * (std / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n, "ci95_low": mean - margin, "ci95_high": mean + margin}


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _group_by_run(records: List[Dict]) -> Dict[int, Dict[str, Dict]]:
    grouped: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for record in records:
        run_number = int(record.get("run_number", -1))
        driver_name = record.get("driver_name")
        if run_number >= 0 and driver_name:
            grouped[run_number][driver_name] = record.get("data", {})
    return grouped


def _compute_win_rate(records: List[Dict], agent1_name: str, agent2_name: str) -> Dict:
    """Compute A1 win rate and positional advantage from race records."""
    run_data = _group_by_run(records)
    win_flags: List[float] = []
    pos_deltas: List[float] = []
    run_indices = sorted(run_data.keys())
    n = len(run_indices)
    third = max(1, n // 3)
    early_wins, late_wins = [], []

    for i, run_idx in enumerate(run_indices):
        d1 = run_data[run_idx].get(agent1_name)
        d2 = run_data[run_idx].get(agent2_name)
        if not d1 or not d2:
            continue
        p1 = float(d1.get("position", 999))
        p2 = float(d2.get("position", 999))
        win = 1.0 if p1 < p2 else 0.0
        win_flags.append(win)
        pos_deltas.append(p2 - p1)
        if i < third:
            early_wins.append(win)
        if i >= n - third:
            late_wins.append(win)

    return {
        "win_rate": _ci95(win_flags),
        "positional_advantage": _ci95(pos_deltas),
        "non_stationarity_drift": _mean(late_wins) - _mean(early_wins),
    }


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def _prepare_cfg(base_config: Dict, phase: str, runs: int, run_name: str,
                 stoch: str, total_laps: int) -> Dict:
    cfg = copy.deepcopy(base_config)
    sim = cfg.setdefault("simulator", {})
    sim["method"] = "batch"
    sim["runs"] = int(runs)
    sim["run_name"] = run_name
    sim["visualise_from_run_name"] = ""
    sim["agent_mode"] = phase
    cfg.setdefault("race_settings", {})["total_laps"] = int(total_laps)
    cfg.setdefault("stochasticity", {})["active_level"] = str(stoch).strip()
    cfg["agent_review_mode"] = False
    cfg["debugMode"] = False
    return cfg


def _run_phase(cfg: Dict, seed: int) -> tuple:
    _set_seed(seed)
    with redirect_stdout(io.StringIO()):
        track = load_track(cfg)
        race_state = init_race_state(cfg, track)
        simulator = init_simulator(race_state, cfg, track)
    log_path = Path(simulator.race_results_log_path)
    records = _read_jsonl(log_path)
    return log_path, records


# ---------------------------------------------------------------------------
# Model file management for cross-play swap
# ---------------------------------------------------------------------------

def _model_path(algo: str, slot: str) -> Path:
    """Return the path for DQN_A{slot}_{algo}_DQN_trained.pth."""
    return ROOT / "models" / f"DQN_A{slot}_{algo}_DQN_trained.pth"


def _swap_models(algo: str) -> None:
    """
    Swap model files between A1 and A2 slots.
    Saves originals as .swap_backup, writes swapped versions,
    restores originals via restore_models().
    """
    p1 = _model_path(algo, "1")
    p2 = _model_path(algo, "2")
    p1_bak = p1.with_suffix(".swap_backup")
    p2_bak = p2.with_suffix(".swap_backup")

    if not p1.exists() or not p2.exists():
        raise FileNotFoundError(
            f"Cannot swap: model files not found.\n  {p1}\n  {p2}"
        )

    shutil.copy2(str(p1), str(p1_bak))
    shutil.copy2(str(p2), str(p2_bak))

    # A1 slot now holds A2's model, A2 slot holds A1's model
    shutil.copy2(str(p2_bak), str(p1))
    shutil.copy2(str(p1_bak), str(p2))

    print(f"[crossplay] Models swapped: A1←A2's weights, A2←A1's weights")


def _restore_models(algo: str) -> None:
    """Restore original model files after a swap evaluation."""
    p1 = _model_path(algo, "1")
    p2 = _model_path(algo, "2")
    p1_bak = p1.with_suffix(".swap_backup")
    p2_bak = p2.with_suffix(".swap_backup")

    if p1_bak.exists():
        shutil.copy2(str(p1_bak), str(p1))
        p1_bak.unlink()
    if p2_bak.exists():
        shutil.copy2(str(p2_bak), str(p2))
        p2_bak.unlink()

    print(f"[crossplay] Models restored to original A1/A2 assignments")


def _copy_to_both_slots(algo: str, source_slot: str) -> None:
    """
    Load one agent's model into BOTH slots (for self-play baseline).
    source_slot: "1" or "2"
    """
    src = _model_path(algo, source_slot)
    dst_slot = "2" if source_slot == "1" else "1"
    dst = _model_path(algo, dst_slot)
    if not src.exists():
        raise FileNotFoundError(f"Cannot copy: {src} not found")
    dst_bak = dst.with_suffix(".selfplay_backup")
    shutil.copy2(str(dst), str(dst_bak))
    shutil.copy2(str(src), str(dst))
    print(f"[crossplay] Both slots loaded with A{source_slot}'s model (self-play)")


def _restore_selfplay_slot(algo: str, overwritten_slot: str) -> None:
    dst = _model_path(algo, overwritten_slot)
    dst_bak = dst.with_suffix(".selfplay_backup")
    if dst_bak.exists():
        shutil.copy2(str(dst_bak), str(dst))
        dst_bak.unlink()
    print(f"[crossplay] Slot A{overwritten_slot} restored from self-play backup")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-play position-swap evaluation for Phase 3 MARL."
    )
    parser.add_argument("--config", default="metrics/phase3/config_rainbow_marl.json")
    parser.add_argument("--train-runs", type=int, default=500)
    parser.add_argument("--eval-runs", type=int, default=150)
    parser.add_argument("--train-seed", type=int, default=101)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--stochasticity-level", default="s2")
    parser.add_argument("--complexity-profile", default="low_marl")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training; use existing model files.")
    parser.add_argument(
        "--modes", default="standard,swapped,a1_selfplay,a2_selfplay",
        help="Comma-separated evaluation modes to run. "
             "Options: standard, swapped, a1_selfplay, a2_selfplay"
    )
    parser.add_argument("--run-prefix", default="crossplay")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default="metrics/phase3/crossplay_latest.json")
    args = parser.parse_args()

    _disable_visualisations()

    config_path = (ROOT / args.config).resolve()
    base_config = _load_config(config_path)

    base_config.setdefault("complexity", {})["active_profile"] = args.complexity_profile

    # Determine algo name and agent names (same logic as evaluate_marl.py)
    dqn_cfg = base_config.get("dqn_params", {})
    algo_name = str(dqn_cfg.get("algo", "vanilla")).strip().lower() or "vanilla"

    agent1_name: Optional[str] = None
    agent2_name: Optional[str] = None
    dqn_seen = 0
    for c in base_config.get("competitors", []):
        if isinstance(c, dict) and str(c.get("agent", "")).lower() == "dqn":
            dqn_seen += 1
            if dqn_seen == 1:
                c["name"] = f"DQN_A1_{algo_name}"
                agent1_name = c["name"]
            elif dqn_seen == 2:
                c["name"] = f"DQN_A2_{algo_name}"
                agent2_name = c["name"]

    if not agent1_name or not agent2_name:
        raise ValueError("Config must have exactly two competitors with agent='dqn'.")

    stoch = args.stochasticity_level.strip() or "s2"
    base_config.setdefault("stochasticity", {})["active_level"] = stoch
    total_laps = int(base_config.get("race_settings", {}).get("total_laps", 5) or 5)
    eval_seed = args.eval_seed if args.eval_seed is not None else args.train_seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # -----------------------------------------------------------------------
    # Training phase
    # -----------------------------------------------------------------------
    train_log_path = None
    if not args.skip_training:
        train_cfg = _prepare_cfg(
            base_config, "training", args.train_runs,
            f"{args.run_prefix}_train_{timestamp}", stoch, total_laps,
        )
        print(
            f"[crossplay] Training: algo={algo_name}, runs={args.train_runs}, "
            f"seed={args.train_seed}, stoch={stoch}"
        )
        train_log_path, _ = _run_phase(train_cfg, args.train_seed)
        print(f"[crossplay] Training complete. Models saved to models/")
    else:
        print(f"[crossplay] --skip-training: using existing model files.")

    # Verify model files exist
    m1 = _model_path(algo_name, "1")
    m2 = _model_path(algo_name, "2")
    if not m1.exists() or not m2.exists():
        raise FileNotFoundError(
            f"Model files not found after training:\n  {m1}\n  {m2}\n"
            "Run without --skip-training or check the algo name."
        )

    # -----------------------------------------------------------------------
    # Evaluation phases
    # -----------------------------------------------------------------------
    results = {}

    def _run_eval(label: str) -> Dict:
        run_name = f"{args.run_prefix}_eval_{label}_{timestamp}"
        eval_cfg = _prepare_cfg(
            base_config, "evaluation", args.eval_runs, run_name, stoch, total_laps,
        )
        print(f"[crossplay] Evaluating mode='{label}', seed={eval_seed}, runs={args.eval_runs}")
        log_path, records = _run_phase(eval_cfg, eval_seed)
        metrics = _compute_win_rate(records, agent1_name, agent2_name)
        wr = metrics["win_rate"]
        print(
            f"[crossplay]   {agent1_name} win rate: {wr['mean']:.3f} "
            f"[{wr['ci95_low']:.3f}, {wr['ci95_high']:.3f}]  "
            f"drift={metrics['non_stationarity_drift']:+.3f}"
        )
        return {**metrics, "log_path": str(log_path), "n_runs": args.eval_runs}

    # Standard: A1 model → A1 slot, A2 model → A2 slot
    if "standard" in modes:
        results["standard"] = _run_eval("standard")

    # Swapped: A1 slot gets A2's model, A2 slot gets A1's model
    # Key diagnostic: if A2 model wins MORE when relabelled as A1, the training
    # asymmetry is policy-based not position-based.
    if "swapped" in modes:
        _swap_models(algo_name)
        try:
            results["swapped"] = _run_eval("swapped")
        finally:
            _restore_models(algo_name)

    # Self-play: A1's model plays against itself (both slots = A1's weights)
    if "a1_selfplay" in modes:
        _copy_to_both_slots(algo_name, "1")
        try:
            results["a1_selfplay"] = _run_eval("a1_selfplay")
        finally:
            _restore_selfplay_slot(algo_name, "2")

    # Self-play: A2's model plays against itself (both slots = A2's weights)
    if "a2_selfplay" in modes:
        _copy_to_both_slots(algo_name, "2")
        try:
            results["a2_selfplay"] = _run_eval("a2_selfplay")
        finally:
            _restore_selfplay_slot(algo_name, "1")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    interpretation = {}
    if "standard" in results and "swapped" in results:
        std_wr = results["standard"]["win_rate"]["mean"]
        swp_wr = results["swapped"]["win_rate"]["mean"]
        # In "swapped" mode, A1 slot holds A2's trained model.
        # If the original A2 model is genuinely better, it should win > 0.5 even
        # when relabelled as "A1" — i.e., swapped_wr should also be < 0.5 for
        # the A1 name (which now holds A2's model), meaning a lower value here
        # still indicates A2 policy dominance.
        # Policy advantage confirmed if: both std_wr < 0.5 AND swp_wr < 0.5
        # Position artefact if: std_wr < 0.5 but swp_wr >= 0.5 (win rate equalises)
        delta = swp_wr - std_wr
        if abs(delta) < 0.05:
            verdict = "policy_advantage_confirmed"
            detail = (
                "Win rate difference between standard and swapped conditions is <5pp. "
                "A2's advantage persists regardless of label assignment — "
                "the policy itself is stronger, not a training position artefact."
            )
        elif delta > 0.10:
            verdict = "position_artefact_likely"
            detail = (
                f"Swapped win rate ({swp_wr:.3f}) is substantially higher than "
                f"standard ({std_wr:.3f}). When A2's model is assigned the A1 slot, "
                "it wins more — consistent with positional/training-order bias rather "
                "than genuine policy superiority."
            )
        else:
            verdict = "ambiguous"
            detail = (
                f"Delta={delta:+.3f} — directional but not conclusive. "
                "Consider more seeds or extended training runs."
            )
        interpretation["policy_vs_position"] = {"verdict": verdict, "detail": detail}

    if "a1_selfplay" in results and "a2_selfplay" in results:
        # Self-play win rates should be ~0.5 (agent vs itself → random coin flip)
        a1sp = results["a1_selfplay"]["win_rate"]["mean"]
        a2sp = results["a2_selfplay"]["win_rate"]["mean"]
        interpretation["selfplay_sanity"] = {
            "a1_selfplay_wr": a1sp,
            "a2_selfplay_wr": a2sp,
            "note": (
                "Self-play win rates should be ~0.50 if evaluation is fair. "
                "Significant deviation indicates a positional bias in the evaluation protocol."
            ),
        }

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    out_payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": str(config_path),
        "phase": "phase3_crossplay",
        "algorithm": algo_name,
        "agent1_name": agent1_name,
        "agent2_name": agent2_name,
        "stochasticity_level": stoch,
        "training": {
            "skipped": bool(args.skip_training),
            "runs": int(args.train_runs),
            "seed": int(args.train_seed),
            "log_path": str(train_log_path) if train_log_path else None,
        },
        "evaluation": {
            "seed": int(eval_seed),
            "runs_per_mode": int(args.eval_runs),
            "modes_run": modes,
        },
        "results": results,
        "interpretation": interpretation,
    }

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"[crossplay] Results written to: {out_path}")

    if interpretation:
        for key, val in interpretation.items():
            if "verdict" in val:
                print(f"[crossplay] {key}: {val['verdict']} — {val['detail'][:80]}...")


if __name__ == "__main__":
    main()
