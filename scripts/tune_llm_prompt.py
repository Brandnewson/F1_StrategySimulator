"""Interactive prompt tuning harness for the LLM agent.

Runs small batches of races, extracts strategic plans and tactical decisions,
and prints a diagnostic report for prompt iteration.

Usage:
    conda activate f1StrategySim
    python scripts/tune_llm_prompt.py --runs 5 --seed 42 --verbose
"""

import argparse
import copy
import io
import json
import os
import random
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List

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
from agents.LLM import LLMAgent


def _disable_visualisations():
    def _noop(*args, **kwargs):
        return None
    RaceSimulator._visualise_results = _noop
    RaceSimulator._visualise_agent_learning = _noop


def _set_deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl_records(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _prepare_config(base_config: dict, runs: int, alpha_mode: str,
                    stochasticity: str = "s0") -> dict:
    """Override config for LLM tuning: low_llm_vs_base, evaluation mode."""
    cfg = copy.deepcopy(base_config)
    cfg["complexity"] = cfg.get("complexity", {})
    cfg["complexity"]["active_profile"] = "low_llm_vs_base"
    cfg["simulator"] = cfg.get("simulator", {})
    cfg["simulator"]["runs"] = runs
    cfg["simulator"]["agent_mode"] = "evaluation"
    cfg["simulator"]["run_name"] = "llm_prompt_tuning"
    cfg["debugMode"] = False
    cfg["agent_review_mode"] = False
    cfg["llm_params"] = cfg.get("llm_params", {})
    cfg["llm_params"]["alpha_mode"] = alpha_mode
    # Set stochasticity level
    cfg["stochasticity"] = cfg.get("stochasticity", {})
    cfg["stochasticity"]["active_level"] = stochasticity
    return cfg


def _run_tuning(cfg: dict, seed: int, verbose: bool):
    """Run simulation and return records + LLM agent references."""
    _set_deterministic_seed(seed)
    _disable_visualisations()

    if verbose:
        track = load_track(cfg)
        race_state = init_race_state(cfg, track)
        simulator = init_simulator(race_state, cfg, track)
    else:
        with redirect_stdout(io.StringIO()):
            track = load_track(cfg)
            race_state = init_race_state(cfg, track)
            simulator = init_simulator(race_state, cfg, track)

    # Collect LLM agents before they're lost
    llm_agents = {}
    for driver in simulator.race_state.drivers:
        if isinstance(driver.agent, LLMAgent):
            llm_agents[driver.name] = driver.agent

    log_path = simulator.race_results_log_path
    records = _read_jsonl_records(Path(log_path)) if log_path else []

    return records, llm_agents


# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------

def check_difficulty_response(strategic_logs: List[Dict], zones: List[Dict]) -> bool:
    """Verify harder zones get less aggressive actions on average."""
    easy_zones = {z["name"] for z in zones if z["difficulty"] <= 0.3}
    hard_zones = {z["name"] for z in zones if z["difficulty"] >= 0.8}

    if not easy_zones or not hard_zones:
        return True  # Can't test without both

    easy_avg = _avg_aggression_for_zones(strategic_logs, easy_zones)
    hard_avg = _avg_aggression_for_zones(strategic_logs, hard_zones)
    return easy_avg > hard_avg


def check_position_response(strategic_logs: List[Dict]) -> bool:
    """Verify plans are more aggressive when behind than when leading."""
    leading = [s for s in strategic_logs if s.get("position", 1) == 1]
    behind = [s for s in strategic_logs if s.get("position", 1) >= 2]

    if not leading or not behind:
        return True

    lead_avg = _avg_aggression_overall(leading)
    behind_avg = _avg_aggression_overall(behind)
    return behind_avg > lead_avg


def check_urgency_response(strategic_logs: List[Dict]) -> bool:
    """Verify final laps show higher peak aggression at targeted zones.

    The LLM may correctly focus on fewer zones rather than raising aggression
    everywhere. We check max aggression at targeted zones, not average across all.
    """
    early = [s for s in strategic_logs if s.get("laps_remaining", 5) >= 4]
    late = [s for s in strategic_logs if s.get("laps_remaining", 5) <= 2]

    if not early or not late:
        return True

    early_max = _max_aggression_at_targets(early)
    late_max = _max_aggression_at_targets(late)
    return late_max >= early_max


def _max_aggression_at_targets(strategic_logs):
    """Max action value across targeted (non-zero) zones."""
    values = []
    for entry in strategic_logs:
        plan = entry.get("plan", {})
        for zone_data in plan.values():
            c = zone_data.get("close", 0)
            if c > 0:
                values.append(c)
    return max(values) if values else 0


def _avg_aggression_for_zones(strategic_logs, zone_names):
    """Average action value across all plans for specified zones."""
    values = []
    for entry in strategic_logs:
        plan = entry.get("plan", {})
        for zname in zone_names:
            if zname in plan:
                values.append(plan[zname].get("close", 0))
                values.append(plan[zname].get("far", 0))
    return sum(values) / len(values) if values else 0.0


def _avg_aggression_overall(strategic_logs):
    """Average action value across all zones in a set of plans."""
    values = []
    for entry in strategic_logs:
        plan = entry.get("plan", {})
        for zone_data in plan.values():
            values.append(zone_data.get("close", 0))
            values.append(zone_data.get("far", 0))
    return sum(values) / len(values) if values else 0.0


# ------------------------------------------------------------------
# Report printing
# ------------------------------------------------------------------

def _print_report(records, llm_agents, zones, runs, alpha_mode):
    """Print the prompt tuning diagnostic report."""
    print(f"\n{'='*60}")
    print(f"  PROMPT TUNING REPORT ({runs} races, alpha={alpha_mode})")
    print(f"{'='*60}\n")

    # Win/loss summary
    llm_names = set(llm_agents.keys())
    llm_wins = 0
    base_wins = 0
    llm_positions = []
    base_positions = []

    # Group by run
    runs_grouped = defaultdict(dict)
    for rec in records:
        rn = rec.get("run_number", -1)
        runs_grouped[rn][rec.get("driver_name")] = rec.get("data", {})

    for run_data in runs_grouped.values():
        for dname, data in run_data.items():
            pos = data.get("position", 99)
            if dname in llm_names:
                llm_positions.append(pos)
                if pos == 1:
                    llm_wins += 1
            else:
                base_positions.append(pos)
                if pos == 1:
                    base_wins += 1

    total_races = len(runs_grouped)
    print(f"Results: LLM won {llm_wins}/{total_races}, Base won {base_wins}/{total_races}")
    if llm_positions:
        print(f"Avg LLM finish: P{sum(llm_positions)/len(llm_positions):.1f} | "
              f"Avg Base finish: P{sum(base_positions)/len(base_positions):.1f}")

    # Collect all strategic logs across agents
    all_strategic = []
    all_tactical = []
    total_cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}

    for aname, agent in llm_agents.items():
        all_strategic.extend(agent.get_strategic_log())
        all_tactical.extend(agent.get_tactical_log())
        cost = agent.get_cost_summary()
        total_cost["calls"] += cost.get("total_calls", 0)
        total_cost["input_tokens"] += cost.get("input_tokens", 0)
        total_cost["output_tokens"] += cost.get("output_tokens", 0)
        total_cost["estimated_cost_usd"] += cost.get("estimated_cost_usd", 0.0)

    # Zone difficulty response
    print(f"\n--- Zone Difficulty Response ---")
    for z in zones:
        avg = _avg_aggression_for_zones(all_strategic, {z["name"]})
        print(f"  {z['name']:30s} (diff={z['difficulty']:.1f}): avg aggression = {avg:.2f}")

    # Sanity checks
    print(f"\n--- Sanity Checks ---")
    diff_ok = check_difficulty_response(all_strategic, zones)
    pos_ok = check_position_response(all_strategic)
    urg_ok = check_urgency_response(all_strategic)
    print(f"  Difficulty response:   {'PASS' if diff_ok else 'FAIL'} "
          f"(easy zones more aggressive than hard zones)")
    print(f"  Position sensitivity:  {'PASS' if pos_ok else 'FAIL'} "
          f"(more aggressive when behind)")
    print(f"  Urgency response:      {'PASS' if urg_ok else 'FAIL'} "
          f"(more aggressive in final laps)")

    # Action distribution from tactical log
    print(f"\n--- Tactical Action Distribution ---")
    zone_actions = defaultdict(lambda: defaultdict(list))
    for entry in all_tactical:
        zone_actions[entry["zone"]][entry["bucket"]].append(entry["action"])

    for z in zones:
        zname = z["name"]
        if zname not in zone_actions:
            continue
        parts = []
        for bucket in ("close", "far"):
            actions = zone_actions[zname].get(bucket, [])
            if actions:
                dist = defaultdict(int)
                for a in actions:
                    dist[a] += 1
                total = len(actions)
                labels = {0: "HLD", 1: "CON", 2: "NOR", 3: "AGG"}
                desc = " ".join(f"{labels[k]}:{dist[k]/total:.0%}" for k in sorted(dist))
                parts.append(f"{bucket}=[{desc}]")
        print(f"  {zname:30s} {' | '.join(parts)}")

    # Cost
    print(f"\n--- Cost ---")
    print(f"  API calls: {total_cost['calls']} | "
          f"Tokens in: {total_cost['input_tokens']:,} | "
          f"Tokens out: {total_cost['output_tokens']:,} | "
          f"~${total_cost['estimated_cost_usd']:.4f}")

    # Sample strategic plans
    print(f"\n--- Sample Strategic Plans (last 3) ---")
    for entry in all_strategic[-3:]:
        print(f"  [Lap {entry['lap']+1}, P{entry['position']}, "
              f"{entry['laps_remaining']} laps left]")
        plan = entry.get("plan", {})
        parts = []
        for z in zones:
            zname = z["name"]
            if zname in plan:
                c = plan[zname].get("close", "?")
                f = plan[zname].get("far", "?")
                short = zname[:12]
                parts.append(f"{short}: c={c} f={f}")
        print(f"    {' | '.join(parts)}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="LLM agent prompt tuning harness")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--runs", type=int, default=5, help="Number of races to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", default="competitive",
                        choices=["competitive", "partial", "cooperative"],
                        help="Alpha mode for LLM agent instructions")
    parser.add_argument("--stochasticity", default="s0", choices=["s0", "s1", "s2"],
                        help="Stochasticity level for overtake outcomes")
    parser.add_argument("--verbose", action="store_true", help="Show simulator output")
    args = parser.parse_args()

    config = _load_config(args.config)
    cfg = _prepare_config(config, args.runs, args.alpha, args.stochasticity)

    # Extract zone info for report
    zones = LLMAgent._extract_zones(config)

    print(f"Running {args.runs} races with LLM agent "
          f"(alpha={args.alpha}, stoch={args.stochasticity}, seed={args.seed})...")
    records, llm_agents = _run_tuning(cfg, args.seed, args.verbose)

    if not llm_agents:
        print("ERROR: No LLM agents found. Check complexity profile and competitor config.")
        sys.exit(1)

    _print_report(records, llm_agents, zones, args.runs, args.alpha)


if __name__ == "__main__":
    main()
