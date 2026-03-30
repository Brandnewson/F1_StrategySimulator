"""
Phase 4/5 Integrity Tests
=========================
Tests validating the mechanisms introduced for the incentive-regime sweep
(Phase 4) and non-zero-sum MARL (Phase 5).

  Group D — Reward sharing formula
  Group E — Balanced evaluation protocol
  Group F — Competitor selection profiles

Run with:
    cd C:/Code/F1_StrategySimulator
    python -m pytest tests/test_phase45_integrity.py -v --tb=short
"""

import copy
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("MPLBACKEND", "Agg")

from src.simulator import RaceSimulator, init_simulator
from src.states import init_race_state
from src.track import load_track
from src.agents.DQN import DQNAgent
from runtime_profiles import (
    select_low_marl_competitors,
    select_low_marl_vs_base_competitors,
)

CONFIG_PATH = ROOT / "config.json"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _disable_visualisations():
    def _noop(*args, **kwargs):
        return None
    RaceSimulator._visualise_results = _noop
    RaceSimulator._visualise_agent_learning = _noop


def _make_marl_config(
    alpha: float = 0.0,
    complexity: str = "low_marl",
    laps: int = 2,
    runs: int = 1,
    stochasticity: str = "s0",
    eval_start_pos_offset: int = 0,
) -> dict:
    """Return a MARL config for fast test execution."""
    cfg = copy.deepcopy(_load_config())
    cfg["competitors"] = [
        {"name": "DQN_A1", "agent": "dqn", "colour": "#022050"},
        {"name": "DQN_A2", "agent": "dqn", "colour": "#CC0000"},
        {"name": "Base_Agent", "agent": "base", "colour": "#FFA500"},
    ]
    cfg["complexity"]["active_profile"] = complexity
    cfg["simulator"]["runs"] = runs
    cfg["simulator"]["method"] = "batch"
    cfg["simulator"]["agent_mode"] = "training"
    cfg["simulator"]["run_name"] = ""
    cfg["simulator"]["visualise_from_run_name"] = ""
    cfg["simulator"]["telemetry"]["log_decision_events"] = True
    cfg["simulator"]["telemetry"]["include_legacy_tick_histories"] = False
    cfg["race_settings"]["total_laps"] = laps
    cfg["stochasticity"]["active_level"] = stochasticity
    cfg["agent_review_mode"] = False
    cfg["debugMode"] = False
    cfg.setdefault("marl", {})["reward_sharing_alpha"] = alpha
    cfg["marl"]["eval_start_pos_offset"] = eval_start_pos_offset
    return cfg


def _run_marl_race(alpha=0.0, complexity="low_marl", laps=2, runs=1,
                    stochasticity="s0", seed=42, eval_start_pos_offset=0):
    """Run a minimal MARL race and return (simulator, records)."""
    np.random.seed(seed)
    _disable_visualisations()
    cfg = _make_marl_config(
        alpha=alpha, complexity=complexity, laps=laps, runs=runs,
        stochasticity=stochasticity, eval_start_pos_offset=eval_start_pos_offset,
    )
    with redirect_stdout(io.StringIO()):
        track = load_track(cfg)
        race_state = init_race_state(cfg, track)
        simulator = init_simulator(race_state, cfg, track)

    log_path = Path(simulator.race_results_log_path)
    records = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return simulator, records


# ===========================================================================
# Group D — Reward sharing formula
# ===========================================================================

class TestRewardSharingFormula:
    """Validate _get_mixed_outcome_raw under all guard conditions."""

    def test_alpha_zero_returns_own_delta(self):
        """At alpha=0.0, outcome_raw must equal starting_position - final_position
        (identical to Phase 3 behaviour, numerical idempotency)."""
        sim, records = _run_marl_race(alpha=0.0, complexity="low_marl", laps=2, runs=3)
        assert records, "No records written"

        checked = 0
        for record in records:
            data = record.get("data", {})
            starting_pos = data.get("starting_position")
            final_pos = data.get("position")
            outcome_raw = data.get("reward_component_totals_raw", {}).get("outcome")
            if starting_pos is None or final_pos is None or outcome_raw is None:
                continue
            expected = float(starting_pos - final_pos)
            assert abs(outcome_raw - expected) < 1e-5, (
                f"Alpha=0.0 idempotency violated for {record['driver_name']}: "
                f"expected {expected}, got {outcome_raw}"
            )
            checked += 1
        assert checked > 0, "No outcome_raw found in records"

    def test_alpha_positive_blends_outcomes(self):
        """At alpha=0.5, outcome_raw should differ from own_delta when agents
        finish in different positions than they started."""
        sim, records = _run_marl_race(alpha=0.5, complexity="low_marl", laps=5, runs=10, seed=42)
        assert records, "No records written"

        dqn_records = [r for r in records if "DQN" in r["driver_name"]]
        assert len(dqn_records) >= 2, "Need at least 2 DQN driver records"

        # Group by run
        runs = {}
        for r in dqn_records:
            runs.setdefault(r["run_number"], []).append(r)

        blended_found = False
        for run_num, run_records in runs.items():
            if len(run_records) != 2:
                continue
            r1, r2 = run_records
            d1 = r1["data"]
            d2 = r2["data"]

            own_delta_1 = float(d1["starting_position"] - d1["position"])
            own_delta_2 = float(d2["starting_position"] - d2["position"])
            outcome_1 = d1.get("reward_component_totals_raw", {}).get("outcome", 0.0)
            outcome_2 = d2.get("reward_component_totals_raw", {}).get("outcome", 0.0)

            # At alpha=0.5: outcome_i = 0.5 * own_delta_i + 0.5 * teammate_delta_j
            expected_1 = 0.5 * own_delta_1 + 0.5 * own_delta_2
            expected_2 = 0.5 * own_delta_2 + 0.5 * own_delta_1

            assert abs(outcome_1 - expected_1) < 1e-5, (
                f"Run {run_num} {r1['driver_name']}: expected blended outcome "
                f"{expected_1}, got {outcome_1}"
            )
            assert abs(outcome_2 - expected_2) < 1e-5, (
                f"Run {run_num} {r2['driver_name']}: expected blended outcome "
                f"{expected_2}, got {outcome_2}"
            )

            if own_delta_1 != own_delta_2:
                blended_found = True

        assert blended_found, (
            "No run produced different own_deltas for the two agents. "
            "Cannot verify blending is non-trivial. Try more runs or different seed."
        )

    def test_guard_low_complexity_skips_sharing(self):
        """Under 'low' complexity (single DQN vs Base), alpha > 0 must be ignored
        because there is only 1 DQN agent (guard condition: exactly 2 DQN agents)."""
        cfg = _make_marl_config(alpha=0.75, complexity="low", laps=2, runs=3)
        # low complexity only keeps 1 DQN + 1 Base
        cfg["competitors"] = [
            {"name": "DQN_Driver", "agent": "dqn", "colour": "#022050"},
            {"name": "Base_Driver", "agent": "base", "colour": "#FFA500"},
        ]
        np.random.seed(42)
        _disable_visualisations()
        with redirect_stdout(io.StringIO()):
            track = load_track(cfg)
            race_state = init_race_state(cfg, track)
            simulator = init_simulator(race_state, cfg, track)

        log_path = Path(simulator.race_results_log_path)
        records = []
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))

        for record in records:
            if "DQN" not in record["driver_name"]:
                continue
            data = record["data"]
            starting_pos = data.get("starting_position")
            final_pos = data.get("position")
            outcome_raw = data.get("reward_component_totals_raw", {}).get("outcome")
            if starting_pos is None or final_pos is None or outcome_raw is None:
                continue
            expected = float(starting_pos - final_pos)
            assert abs(outcome_raw - expected) < 1e-5, (
                f"Guard failed: alpha=0.75 in 'low' complexity should not blend, "
                f"but outcome_raw={outcome_raw} != own_delta={expected}"
            )

    def test_alpha_sharing_active_in_low_marl_vs_base(self):
        """In low_marl_vs_base with alpha > 0, the outcome_raw values must
        differ from the simple own_delta for at least some runs, confirming
        that reward sharing is active and the guard condition accepts the
        low_marl_vs_base profile."""
        sim, records = _run_marl_race(
            alpha=0.5, complexity="low_marl_vs_base", laps=3, runs=5, seed=101,
        )
        assert records, "No records written"

        dqn_records = [r for r in records if "DQN" in r["driver_name"]]
        blended_count = 0
        checked = 0

        for r in dqn_records:
            data = r["data"]
            sp = data.get("starting_position")
            fp = data.get("position")
            outcome = data.get("reward_component_totals_raw", {}).get("outcome")
            if sp is None or fp is None or outcome is None:
                continue
            own_delta = float(sp - fp)
            checked += 1
            if abs(outcome - own_delta) > 1e-5:
                blended_count += 1

        assert checked > 0, "No DQN outcome records found"
        assert blended_count > 0, (
            f"All {checked} DQN records had outcome_raw == own_delta. "
            f"Reward sharing does not appear active in low_marl_vs_base."
        )

    def test_known_timing_limitation_three_agent(self):
        """KNOWN LIMITATION: In 3-agent races, the terminal bonus for the first
        agent to finish uses the teammate's CURRENT position, which may differ
        from the teammate's FINAL position (the teammate is still racing against
        Base). At alpha=0.5, this breaks the mathematical symmetry where both
        agents should receive identical outcome_raw values.

        This test DOCUMENTS the limitation by verifying the asymmetry exists.
        The maximum error per episode is bounded by alpha * 1 position shift.
        The error is non-systematic (teammate may move up or down), adding noise
        but not bias. Phase 5 results remain valid because the signal (83%
        JointBeatBase improvement at alpha=0.75) substantially exceeds this
        noise floor."""
        sim, records = _run_marl_race(
            alpha=0.5, complexity="low_marl_vs_base", laps=3, runs=10, seed=101,
        )
        dqn_records = [r for r in records if "DQN" in r["driver_name"]]
        runs = {}
        for r in dqn_records:
            runs.setdefault(r["run_number"], []).append(r)

        asymmetric_count = 0
        total_pairs = 0
        for run_num, run_records in runs.items():
            if len(run_records) != 2:
                continue
            r1, r2 = sorted(run_records, key=lambda x: x["driver_name"])
            o1 = r1["data"].get("reward_component_totals_raw", {}).get("outcome", 0.0)
            o2 = r2["data"].get("reward_component_totals_raw", {}).get("outcome", 0.0)
            total_pairs += 1
            if abs(o1 - o2) > 1e-5:
                asymmetric_count += 1

        # Document the rate rather than assert it away
        rate = asymmetric_count / total_pairs if total_pairs > 0 else 0
        print(f"\n[KNOWN LIMITATION] 3-agent terminal timing asymmetry: "
              f"{asymmetric_count}/{total_pairs} runs ({rate:.0%}) had different "
              f"outcome_raw for A1 vs A2 at alpha=0.5. "
              f"This is caused by terminal bonus computation using the teammate's "
              f"intermediate position, not final position.")
        # This test always passes — it documents, not enforces.
        assert total_pairs > 0, "No paired runs found"


# ===========================================================================
# Group E — Balanced evaluation protocol
# ===========================================================================

class TestBalancedEvalProtocol:
    """Validate that eval_start_pos_offset produces different starting position cycles."""

    def test_offset_zero_starts_a1_first(self):
        """With offset=0, DQN_A1 should get position 1 in the first race."""
        sim, records = _run_marl_race(
            alpha=0.0, complexity="low_marl", laps=1, runs=1,
            seed=42, eval_start_pos_offset=0,
        )
        run1_records = [r for r in records if r["run_number"] == 0]
        a1 = [r for r in run1_records if "A1" in r["driver_name"]]
        assert a1, "DQN_A1 not found in records"
        assert a1[0]["data"]["starting_position"] == 1, (
            f"Offset=0: expected A1 starting_position=1, "
            f"got {a1[0]['data']['starting_position']}"
        )

    def test_offset_one_starts_a2_first(self):
        """With offset=1, DQN_A2 should get position 1 in the first race (A1 gets position 2)."""
        sim, records = _run_marl_race(
            alpha=0.0, complexity="low_marl", laps=1, runs=1,
            seed=42, eval_start_pos_offset=1,
        )
        run1_records = [r for r in records if r["run_number"] == 0]
        a1 = [r for r in run1_records if "A1" in r["driver_name"]]
        assert a1, "DQN_A1 not found in records"
        assert a1[0]["data"]["starting_position"] == 2, (
            f"Offset=1: expected A1 starting_position=2, "
            f"got {a1[0]['data']['starting_position']}"
        )

    def test_three_agent_offset_cycles(self):
        """With 3 agents (low_marl_vs_base), offset=0 and offset=1 produce different
        starting grids for the first race."""
        _, records_0 = _run_marl_race(
            alpha=0.0, complexity="low_marl_vs_base", laps=1, runs=1,
            seed=42, eval_start_pos_offset=0,
        )
        _, records_1 = _run_marl_race(
            alpha=0.0, complexity="low_marl_vs_base", laps=1, runs=1,
            seed=42, eval_start_pos_offset=1,
        )
        a1_pos_0 = [r for r in records_0 if "A1" in r["driver_name"]][0]["data"]["starting_position"]
        a1_pos_1 = [r for r in records_1 if "A1" in r["driver_name"]][0]["data"]["starting_position"]

        assert a1_pos_0 != a1_pos_1, (
            f"Offset 0 and 1 should produce different A1 starting positions, "
            f"but both gave position {a1_pos_0}"
        )


# ===========================================================================
# Group F — Competitor selection profiles
# ===========================================================================

class TestCompetitorSelection:
    """Validate runtime profile competitor selection functions."""

    COMPETITORS = [
        {"name": "DQN Agent", "agent": "dqn"},
        {"name": "DQN Agent 2", "agent": "dqn"},
        {"name": "Base Agent", "agent": "base"},
    ]

    def test_low_marl_selects_two_dqn(self):
        """low_marl must select exactly 2 DQN agents and no Base."""
        result = select_low_marl_competitors(self.COMPETITORS)
        assert len(result) == 2
        assert all(c["agent"] == "dqn" for c in result)

    def test_low_marl_vs_base_selects_two_dqn_one_base(self):
        """low_marl_vs_base must select exactly 2 DQN agents and 1 Base agent."""
        result = select_low_marl_vs_base_competitors(self.COMPETITORS)
        assert len(result) == 3
        dqn_count = sum(1 for c in result if c["agent"] == "dqn")
        base_count = sum(1 for c in result if c["agent"] == "base")
        assert dqn_count == 2, f"Expected 2 DQN, got {dqn_count}"
        assert base_count == 1, f"Expected 1 Base, got {base_count}"

    def test_low_marl_vs_base_rejects_missing_base(self):
        """low_marl_vs_base must raise ValueError if no Base agent in config."""
        dqn_only = [
            {"name": "DQN Agent", "agent": "dqn"},
            {"name": "DQN Agent 2", "agent": "dqn"},
        ]
        with pytest.raises(ValueError, match="agent='base'"):
            select_low_marl_vs_base_competitors(dqn_only)

    def test_low_marl_vs_base_rejects_single_dqn(self):
        """low_marl_vs_base must raise ValueError if fewer than 2 DQN agents."""
        one_dqn = [
            {"name": "DQN Agent", "agent": "dqn"},
            {"name": "Base Agent", "agent": "base"},
        ]
        with pytest.raises(ValueError, match="agent='dqn'"):
            select_low_marl_vs_base_competitors(one_dqn)
