"""Phase 9 — Formal evaluation of the LLM agent as a semantic reasoning baseline.

Runs the LLM agent (Haiku) against the Base agent across multiple seeds and
stochasticity levels, producing metrics in the same format as evaluate_candidate.py
for direct comparison with DQN Phase 2 results.

Usage:
    conda activate f1StrategySim
    python scripts/evaluate_llm_agent.py --eval-runs 150 --stochasticity-level s0
    python scripts/evaluate_llm_agent.py --eval-runs 150 --stochasticity-level s1
    python scripts/evaluate_llm_agent.py --eval-runs 150 --stochasticity-level s2
"""

import argparse
import copy
import io
import json
import os
import random
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))

from src.simulator import DEFAULT_PROTOCOL_CONTRACT, RaceSimulator, init_simulator
from src.states import init_race_state
from src.track import load_track
from agents.LLM import LLMAgent

# Reuse core utilities from evaluate_candidate
from evaluate_candidate import (
    _ci95,
    _compute_eval_metrics,
    _compute_win_flags_vs_group,
    _group_by_run,
    _parse_seeds,
    _prepare_phase_config,
    _read_jsonl_records,
    _remap_run_numbers,
    _set_deterministic_seed,
)

# Reuse sanity checks from tune_llm_prompt
from tune_llm_prompt import (
    check_difficulty_response,
    check_position_response,
    check_urgency_response,
)


def _disable_visualisations() -> None:
    def _noop(*args, **kwargs):
        return None
    RaceSimulator._visualise_results = _noop
    RaceSimulator._visualise_agent_learning = _noop


def _run_llm_eval_phase(
    cfg: Dict, seed: int, verbose: bool
) -> Tuple[Path, List[Dict], Dict[str, LLMAgent]]:
    """Run a single evaluation seed and return records + LLM agent references."""
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

    # Extract LLM agents BEFORE references are lost
    llm_agents = {}
    for driver in simulator.race_state.drivers:
        if isinstance(driver.agent, LLMAgent):
            llm_agents[driver.name] = driver.agent

    race_log_path = simulator.race_results_log_path
    if race_log_path is None:
        raise RuntimeError("Simulation phase did not produce race_results.jsonl")
    records = _read_jsonl_records(Path(race_log_path))
    return Path(race_log_path), records, llm_agents


def _collect_llm_diagnostics(
    all_agents: Dict[str, List[LLMAgent]],
) -> Dict[str, Any]:
    """Aggregate cost and log stats across all LLM agents and seeds."""
    total = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    strategic_count = 0
    tactical_count = 0

    for agent_list in all_agents.values():
        for agent in agent_list:
            cost = agent.get_cost_summary()
            total["calls"] += cost.get("total_calls", 0)
            total["input_tokens"] += cost.get("input_tokens", 0)
            total["output_tokens"] += cost.get("output_tokens", 0)
            strategic_count += len(agent.get_strategic_log())
            tactical_count += len(agent.get_tactical_log())

    est_cost = (
        total["input_tokens"] * 0.25 / 1_000_000
        + total["output_tokens"] * 1.25 / 1_000_000
    )

    return {
        "total_api_calls": total["calls"],
        "total_input_tokens": total["input_tokens"],
        "total_output_tokens": total["output_tokens"],
        "estimated_cost_usd": round(est_cost, 4),
        "strategic_plans_generated": strategic_count,
        "tactical_decisions_made": tactical_count,
    }


def _compute_strategic_coherence(
    all_agents: Dict[str, List[LLMAgent]],
    zones: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the 3 sanity checks on aggregated strategic logs."""
    all_strategic = []
    for agent_list in all_agents.values():
        for agent in agent_list:
            all_strategic.extend(agent.get_strategic_log())

    return {
        "difficulty_response": check_difficulty_response(all_strategic, zones),
        "position_sensitivity": check_position_response(all_strategic),
        "urgency_response": check_urgency_response(all_strategic),
        "total_plans_analysed": len(all_strategic),
    }


def _sample_strategic_plans(
    all_agents: Dict[str, List[LLMAgent]], n: int = 10
) -> List[Dict]:
    """Collect last N strategic plans for qualitative analysis."""
    all_plans = []
    for agent_list in all_agents.values():
        for agent in agent_list:
            all_plans.extend(agent.get_strategic_log())
    return all_plans[-n:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 9 — Evaluate LLM agent as semantic reasoning baseline."
    )
    parser.add_argument("--config", default="config.json", help="Base config path")
    parser.add_argument("--eval-runs", type=int, default=None,
                        help="Eval runs per seed (default: from protocol)")
    parser.add_argument("--eval-seeds", default="101,202,303,404,505",
                        help="Comma-separated eval seeds")
    parser.add_argument("--stochasticity-level", default="s0",
                        choices=["s0", "s1", "s2"],
                        help="Stochasticity level")
    parser.add_argument("--alpha", default="competitive",
                        choices=["competitive", "partial", "cooperative"],
                        help="LLM alpha instruction mode")
    parser.add_argument("--total-laps", type=int, default=None,
                        help="Override total laps (default: from config)")
    parser.add_argument("--out", default=None,
                        help="Output metrics JSON path (default: auto-generated)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show simulator output")
    args = parser.parse_args()

    # ---- Load & configure ----
    with open(args.config, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    # Force LLM complexity profile
    base_config.setdefault("complexity", {})["active_profile"] = "low_llm_vs_base"
    base_config.setdefault("llm_params", {})["alpha_mode"] = args.alpha

    # Resolve protocol
    protocol = copy.deepcopy(DEFAULT_PROTOCOL_CONTRACT)
    user_proto = base_config.get("protocol", {})
    if isinstance(user_proto, dict):
        protocol.update(user_proto)

    # Resolve eval params
    eval_runs = args.eval_runs
    if eval_runs is None:
        eval_runs_map = protocol.get("eval_runs", {})
        eval_runs = int(eval_runs_map.get("low", 150))
    eval_seeds = _parse_seeds(args.eval_seeds, 101)

    total_laps = args.total_laps
    if total_laps is None:
        total_laps = int(base_config.get("race_settings", {}).get("total_laps", 5))

    stoch_level = args.stochasticity_level

    # Identify driver names
    llm_driver_names = []
    baseline_driver_names = []
    for comp in base_config.get("competitors", []):
        agent = str(comp.get("agent", "")).strip().lower()
        name = comp.get("name", "")
        if agent == "llm":
            llm_driver_names.append(name)
        elif agent == "base":
            baseline_driver_names.append(name)

    # Output path
    out_path = args.out
    if out_path is None:
        os.makedirs(str(ROOT / "metrics" / "phase9"), exist_ok=True)
        out_path = str(ROOT / "metrics" / "phase9" / f"llm_{stoch_level}.json")

    # ---- Print plan ----
    print(f"Phase 9 — LLM Agent Evaluation")
    print(f"  Stochasticity: {stoch_level}")
    print(f"  Alpha mode:    {args.alpha}")
    print(f"  Eval runs:     {eval_runs} per seed")
    print(f"  Seeds:         {eval_seeds}")
    print(f"  Total races:   {eval_runs * len(eval_seeds)}")
    print(f"  Output:        {out_path}")
    est_calls = eval_runs * len(eval_seeds) * total_laps // 2  # ~half start P2
    est_cost = est_calls * 2500 * 0.25 / 1_000_000  # rough input token estimate
    print(f"  Est. API calls: ~{est_calls} (~${est_cost:.2f})")
    print()

    # ---- Run evaluation across seeds ----
    all_eval_records: List[Dict] = []
    all_llm_agents: Dict[str, List[LLMAgent]] = {}  # name → [agent_per_seed]
    eval_runs_meta: List[Dict] = []
    run_offset = 0

    for seed_idx, seed in enumerate(eval_seeds):
        print(f"  Seed {seed} ({seed_idx + 1}/{len(eval_seeds)})...", end=" ", flush=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"phase9_llm_{stoch_level}_{args.alpha}_s{seed}_{ts}"

        cfg = _prepare_phase_config(
            base_config,
            phase="evaluation",
            runs=eval_runs,
            run_name=run_name,
            total_laps=total_laps,
            stochasticity_level=stoch_level,
        )

        log_path, records, llm_agents = _run_llm_eval_phase(cfg, seed, args.verbose)

        # Collect LLM agents for diagnostics
        for name, agent in llm_agents.items():
            all_llm_agents.setdefault(name, []).append(agent)

        # Remap and accumulate
        records = _remap_run_numbers(records, run_offset)
        run_offset += eval_runs
        all_eval_records.extend(records)

        # Track metadata
        cost = sum(
            a.get_cost_summary().get("total_calls", 0) for a in llm_agents.values()
        )
        eval_runs_meta.append({
            "seed": seed,
            "runs": eval_runs,
            "stochasticity_level": stoch_level,
            "eval_log_path": str(log_path),
            "api_calls": cost,
        })

        print(f"done ({len(records)} records, {cost} API calls)")

    # ---- Compute metrics (reuses evaluate_candidate's core function) ----
    print("\nComputing metrics...")
    eval_metrics = _compute_eval_metrics(
        records=all_eval_records,
        dqn_driver_names=llm_driver_names,  # same filtering logic works
        baseline_driver_names=baseline_driver_names,
        random_driver_names=[],
        total_laps=total_laps,
    )

    # ---- Per-seed win rate variance ----
    seed_win_rates = []
    grouped = _group_by_run(all_eval_records)
    cumulative_offset = 0
    for meta in eval_runs_meta:
        seed_runs = {
            k: v for k, v in grouped.items()
            if cumulative_offset <= k < cumulative_offset + meta["runs"]
        }
        cumulative_offset += meta["runs"]
        if seed_runs:
            flags = _compute_win_flags_vs_group(seed_runs, llm_driver_names, baseline_driver_names)
            if flags:
                seed_win_rates.append(float(np.mean(flags)))

    if len(seed_win_rates) > 1:
        eval_metrics.setdefault("stability", {})["win_rate_vs_baseline_variance_across_seeds"] = float(
            np.var(seed_win_rates, ddof=1)
        )
        eval_metrics["stability"]["per_seed_win_rates"] = seed_win_rates

    # ---- LLM-specific diagnostics ----
    zones = LLMAgent._extract_zones(base_config)
    llm_diag = _collect_llm_diagnostics(all_llm_agents)
    coherence = _compute_strategic_coherence(all_llm_agents, zones)
    sample_plans = _sample_strategic_plans(all_llm_agents, n=10)

    # ---- Assemble output ----
    primary_score = eval_metrics.get("primary_objective", {}).get("score", 0.0)

    output = {
        "created_at": datetime.now().isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "protocol": protocol,
        "reward_contract": base_config.get("reward", {}),
        "feedback_contract": base_config.get("feedback", {}),
        "stochasticity_contract": base_config.get("stochasticity", {}),
        "algorithm": {
            "family": "llm",
            "name": "haiku",
            "model": base_config.get("llm_params", {}).get("model", "claude-haiku-4-5-20251001"),
            "alpha_mode": args.alpha,
        },
        "objective_name": f"win_rate_vs_baseline[low_llm_vs_base]",
        "objective_score": primary_score,
        "llm_driver_names": llm_driver_names,
        "baseline_driver_names": baseline_driver_names,
        "phases": {
            "training": {"skipped": True, "reason": "LLM agent requires no training"},
            "evaluation": {
                "runs_per_seed": eval_runs,
                "seeds": eval_seeds,
                "complexity_profile": "low_llm_vs_base",
                "stochasticity_level": stoch_level,
                "seed_runs": eval_runs_meta,
            },
        },
        "metrics": eval_metrics,
        "llm_diagnostics": llm_diag,
        "strategic_coherence": coherence,
        "strategic_plans_sample": sample_plans,
        "complexity": "low_llm_vs_base",
        "stochasticity": stoch_level,
    }

    # ---- Write output ----
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    # ---- Print summary ----
    wr = eval_metrics.get("win_rate_vs_baseline", {})
    pos = eval_metrics.get("race_quality", {}).get("avg_position_dqn", {})
    tac = eval_metrics.get("tactical", {})
    attempt_rate = tac.get("overtake_attempt_rate_per_driver_run", 0)
    success = tac.get("overtake_success_rate", {})

    print(f"\n{'='*60}")
    print(f"  PHASE 9 RESULTS — LLM vs Base ({stoch_level}, {args.alpha})")
    print(f"{'='*60}")
    print(f"  Win rate vs baseline:  {wr.get('mean', 0):.3f} "
          f"[{wr.get('ci95_low', 0):.3f}, {wr.get('ci95_high', 0):.3f}]")
    print(f"  Avg LLM position:      {pos.get('mean', 0):.2f} "
          f"[{pos.get('ci95_low', 0):.2f}, {pos.get('ci95_high', 0):.2f}]")
    print(f"  Overtake attempts/run: {attempt_rate:.2f}")
    print(f"  Overtake success rate: {success.get('rate', 0):.3f} "
          f"({success.get('succeeded', 0)}/{success.get('attempted', 0)})")
    print(f"  Cross-seed variance:   {eval_metrics.get('stability', {}).get('win_rate_vs_baseline_variance_across_seeds', 'N/A')}")
    print(f"  Coherence checks:      "
          f"diff={'PASS' if coherence.get('difficulty_response') else 'FAIL'} | "
          f"pos={'PASS' if coherence.get('position_sensitivity') else 'FAIL'} | "
          f"urg={'PASS' if coherence.get('urgency_response') else 'FAIL'}")
    print(f"  API cost:              ${llm_diag.get('estimated_cost_usd', 0):.4f} "
          f"({llm_diag.get('total_api_calls', 0)} calls)")
    print(f"\n  Metrics written to: {out_path}")

    # Zone behaviour summary
    zone_beh = eval_metrics.get("behavioral_diagnostics", {}).get("zone_behavior", {})
    if zone_beh:
        print(f"\n  Zone Behavior:")
        print(f"  {'Zone':<30s} {'Decisions':>9s} {'Attempts':>9s} {'Success':>9s} {'Rate':>7s}")
        for zid in sorted(zone_beh.keys()):
            z = zone_beh[zid]
            sr = z.get("success_rate", 0)
            print(f"  {z.get('zone_name', zid):<30s} "
                  f"{z.get('decisions', 0):>9d} "
                  f"{z.get('attempts', 0):>9d} "
                  f"{z.get('successes', 0):>9d} "
                  f"{sr:>7.3f}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
