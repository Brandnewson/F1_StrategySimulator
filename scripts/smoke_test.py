#!/usr/bin/env python3
"""
Smoke test for F1 Strategy Simulator.

Runs a single 1-lap race in evaluation mode to verify the simulator pipeline
still works after a code change. Exits 0 on success, 1 on any failure.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --config config.json --seed 42
"""

import argparse
import copy
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: 1 run, 1 lap, evaluation mode.")
    parser.add_argument("--config", default="config.json", help="Path to base config JSON.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    parser.add_argument("--verbose", action="store_true", help="Print simulator output.")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"SMOKE FAIL: config not found at {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    # Override to minimal single-race, single-lap, evaluation only
    cfg = copy.deepcopy(base_config)
    cfg["debugMode"] = False
    cfg["agent_review_mode"] = False
    cfg.setdefault("simulator", {})
    cfg["simulator"]["method"] = "batch"
    cfg["simulator"]["runs"] = 1
    cfg["simulator"]["visualise_from_run_name"] = ""
    cfg["simulator"]["agent_mode"] = "evaluation"
    cfg["simulator"]["run_name"] = "smoke_test"
    cfg.setdefault("race_settings", {})
    cfg["race_settings"]["total_laps"] = 1

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    try:
        from src.simulator import RaceSimulator, init_simulator
        from src.states import init_race_state
        from src.track import load_track

        # Disable visualisations
        RaceSimulator._visualise_results = lambda *a, **k: None
        RaceSimulator._visualise_agent_learning = lambda *a, **k: None

        if args.verbose:
            track = load_track(cfg)
            race_state = init_race_state(cfg, track)
            simulator = init_simulator(race_state, cfg, track)
        else:
            with redirect_stdout(io.StringIO()):
                track = load_track(cfg)
                race_state = init_race_state(cfg, track)
                simulator = init_simulator(race_state, cfg, track)

        print("SMOKE OK: simulator ran 1 lap in evaluation mode without errors.")
        return 0

    except Exception:
        print("SMOKE FAIL: simulator raised an exception:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
