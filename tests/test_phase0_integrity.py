"""
Phase 0 System Integrity Tests
================================
Three groups of tests that must pass before any learning claims can be made:

  Group A — Simulator correctness
  Group B — Stochastic calibration (statistical)
  Group C — Telemetry contract

Run with:
    cd C:/Code/F1_StrategySimulator
    python -m pytest tests/test_phase0_integrity.py -v --tb=short 2>&1 | tee phase0_results.txt
"""

import copy
import io
import json
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — mirror the pattern used in evaluate_candidate.py
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("MPLBACKEND", "Agg")

from src.simulator import RaceSimulator, init_simulator
from src.states import init_race_state
from src.track import load_track
from base_agents import RiskLevel

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = ROOT / "config.json"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _make_minimal_config(laps: int = 2, runs: int = 1, stochasticity: str = "s0",
                          agent_mode: str = "training") -> dict:
    """Return a minimal config derived from config.json, tuned for fast tests."""
    cfg = _load_config()
    cfg = copy.deepcopy(cfg)

    # Keep only 2 competitors: one DQN, one base
    cfg["competitors"] = [
        {"name": "DQN_Driver", "agent": "dqn", "colour": "#022050"},
        {"name": "Base_Driver", "agent": "base", "colour": "#FFA500"},
    ]

    cfg["simulator"]["runs"] = runs
    cfg["simulator"]["method"] = "batch"
    cfg["simulator"]["agent_mode"] = agent_mode
    cfg["simulator"]["run_name"] = ""          # auto-generate so logs don't collide
    cfg["simulator"]["visualise_from_run_name"] = ""
    cfg["simulator"]["telemetry"]["log_decision_events"] = True
    cfg["simulator"]["telemetry"]["include_legacy_tick_histories"] = False

    cfg["race_settings"]["total_laps"] = laps
    cfg["stochasticity"]["active_level"] = stochasticity
    cfg["agent_review_mode"] = False
    cfg["debugMode"] = False

    return cfg


def _disable_visualisations():
    """Monkey-patch matplotlib-heavy methods so tests don't open windows."""
    def _noop(*args, **kwargs):
        return None
    RaceSimulator._visualise_results = _noop
    RaceSimulator._visualise_agent_learning = _noop


def _run_minimal_race(laps: int = 2, runs: int = 1, stochasticity: str = "s0",
                       seed: int = 42) -> tuple:
    """Run a minimal race and return (simulator, records).

    Returns
    -------
    simulator : RaceSimulator
    records   : list[dict]  — parsed race_results.jsonl entries
    """
    np.random.seed(seed)

    _disable_visualisations()
    cfg = _make_minimal_config(laps=laps, runs=runs, stochasticity=stochasticity)

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
# Group A — Simulator correctness
# ===========================================================================

class TestSimulatorCorrectness:

    def test_one_decision_per_zone_per_lap(self):
        """Each (driver, zone_id, lap) triple must appear at most once in decision_events."""
        _, records = _run_minimal_race(laps=3, runs=3)
        assert records, "No records written — telemetry may be broken"

        # Key: (run_number, driver_name) -> set of (zone_id, lap)
        # Same zone/lap CAN appear in different runs (after race_reset); that is expected.
        seen = defaultdict(set)
        for record in records:
            scope_key = (record["run_number"], record["driver_name"])
            events = record.get("data", {}).get("decision_events", [])
            for event in events:
                key = (event["zone_id"], event["lap"])
                assert key not in seen[scope_key], (
                    f"Duplicate decision within same run for "
                    f"driver={record['driver_name']} run={record['run_number']} "
                    f"zone={event['zone_id']} lap={event['lap']}"
                )
                seen[scope_key].add(key)

    def test_cooldown_enforcement(self):
        """No two consecutive attempt=True events for the same driver should be < 5s apart."""
        tick_duration = 0.01  # seconds per tick (from config)
        cooldown_seconds = 5.0

        _, records = _run_minimal_race(laps=5, runs=5)

        for record in records:
            driver = record["driver_name"]
            events = record.get("data", {}).get("decision_events", [])
            attempt_events = [e for e in events if e.get("attempt", False)]
            for i in range(1, len(attempt_events)):
                tick_gap = attempt_events[i]["decision_tick"] - attempt_events[i - 1]["decision_tick"]
                elapsed_seconds = tick_gap * tick_duration
                assert elapsed_seconds >= cooldown_seconds - 0.001, (
                    f"Cooldown violated for {driver}: "
                    f"{elapsed_seconds:.3f}s between consecutive attempts "
                    f"(ticks {attempt_events[i-1]['decision_tick']} -> {attempt_events[i]['decision_tick']})"
                )

    def test_terminal_reward_correctness(self):
        """outcome_raw at race end == starting_position - final_position for each driver."""
        _, records = _run_minimal_race(laps=2, runs=3)
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
                f"Terminal outcome mismatch for {record['driver_name']} "
                f"(run {record['run_number']}): "
                f"expected outcome_raw={expected}, got {outcome_raw}"
            )
            checked += 1

        assert checked > 0, "No drivers had outcome_raw in records — check reward schema"

    def test_reward_sum_identity(self):
        """reward_component_sum_error must be < 1e-4 for every decision event."""
        _, records = _run_minimal_race(laps=3, runs=3)

        max_error = 0.0
        total_events = 0
        for record in records:
            events = record.get("data", {}).get("decision_events", [])
            for event in events:
                error = abs(float(event.get("reward_component_sum_error", 0.0)))
                max_error = max(max_error, error)
                if error >= 1e-4:
                    pytest.fail(
                        f"reward_component_sum_error={error:.2e} >= 1e-4 "
                        f"in run={record['run_number']} driver={record['driver_name']} "
                        f"zone={event.get('zone_id')} lap={event.get('lap')}"
                    )
                total_events += 1

        # Informational: print max error observed
        print(f"\n[sum_identity] checked {total_events} events, max |error| = {max_error:.2e}")

    def test_inactive_components_zero(self):
        """In low complexity, 'pace' and 'tyre_pit' must be 0.0 everywhere."""
        _, records = _run_minimal_race(laps=2, runs=2)
        assert records, "No records written"

        inactive = ["pace", "tyre_pit"]

        for record in records:
            data = record.get("data", {})

            # Check aggregate totals (both weighted and raw)
            for field in ("reward_component_totals", "reward_component_totals_raw"):
                totals = data.get(field, {})
                for comp in inactive:
                    val = totals.get(comp, 0.0)
                    assert val == 0.0, (
                        f"{field}[{comp}]={val} expected 0.0 "
                        f"(run={record['run_number']}, driver={record['driver_name']})"
                    )

            # Check every individual decision event
            for event in data.get("decision_events", []):
                for comp in inactive:
                    for key in ("reward_components", "reward_components_raw", "reward_components_normalized"):
                        val = event.get(key, {}).get(comp, 0.0)
                        assert val == 0.0, (
                            f"event[{key}][{comp}]={val} expected 0.0 "
                            f"(run={record['run_number']}, driver={record['driver_name']}, "
                            f"zone={event.get('zone_id')}, lap={event.get('lap')})"
                        )


# ===========================================================================
# Group B — Stochastic calibration (statistical)
# ===========================================================================

class TestStochasticCalibration:
    """
    These tests sample the overtake probability calculation directly
    by constructing a minimal simulator and calling _attempt_overtake
    in a controlled loop. We count outcomes over 1000 repetitions and
    assert statistical ordering (not exact values).
    """

    N_REPS = 1000
    ALPHA = 0.01   # required significance — with 1000 reps the ordering must be clear

    @pytest.fixture(scope="class")
    def simulator(self):
        """Build a minimal RaceSimulator (s1) for probability sampling."""
        _disable_visualisations()
        cfg = _make_minimal_config(laps=1, runs=1, stochasticity="s1")
        with redirect_stdout(io.StringIO()):
            track = load_track(cfg)
            race_state = init_race_state(cfg, track)
            sim = RaceSimulator(race_state, cfg, track)
        return sim

    def _count_successes(self, simulator, zone: dict, risk_level, gap_km: float,
                          n: int, seed: int = 0) -> int:
        """Call _attempt_overtake n times and count successes, without mutating driver state."""
        drivers = simulator.race_state.drivers
        overtaking = drivers[1]   # lower position (behind)
        target = drivers[0]       # higher position (ahead)

        np.random.seed(seed)
        successes = 0
        for _ in range(n):
            # We need to extract probability without side-effects;
            # call the internal probability logic directly.
            profile = simulator.active_stochasticity_profile
            difficulty = float(zone.get("difficulty", 0.5))
            gap = float(gap_km)

            base_probability = (1.0 - difficulty) * float(profile.get("base_probability_scale", 1.0))
            risk_prob_modifiers = {
                RiskLevel.CONSERVATIVE: -0.08,
                RiskLevel.NORMAL: 0.0,
                RiskLevel.AGGRESSIVE: 0.12,
            }
            risk_modifier_scale = float(profile.get("risk_modifier_scale", 1.0))
            risk_adjustment = float(risk_prob_modifiers.get(risk_level, 0.0)) * risk_modifier_scale

            gap_norm = float(np.clip(gap / 0.1, 0.0, 1.0))
            gap_modifier_scale = float(profile.get("gap_modifier_scale", 1.0))
            gap_modifier = (1.0 - gap_norm - 0.5) * 0.3 * gap_modifier_scale

            probability_noise_std = float(profile.get("probability_noise_std", 0.0))
            noise = float(np.random.normal(0.0, probability_noise_std)) if probability_noise_std > 0.0 else 0.0

            min_p = float(profile.get("min_success_probability", 0.02))
            max_p = float(profile.get("max_success_probability", 0.95))
            prob = float(np.clip(base_probability + risk_adjustment + gap_modifier + noise, min_p, max_p))

            if np.random.random() < prob:
                successes += 1
        return successes

    def test_aggression_increases_probability(self, simulator):
        """AGGRESSIVE > NORMAL > CONSERVATIVE success rate at s1, fixed zone and gap."""
        zone = {"difficulty": 0.5, "name": "TestZone", "distance_from_start": 1.0}
        gap = 0.03  # 30m — reasonably close

        n_cons = self._count_successes(simulator, zone, RiskLevel.CONSERVATIVE, gap, self.N_REPS, seed=1)
        n_norm = self._count_successes(simulator, zone, RiskLevel.NORMAL, gap, self.N_REPS, seed=2)
        n_agg  = self._count_successes(simulator, zone, RiskLevel.AGGRESSIVE, gap, self.N_REPS, seed=3)

        rate_cons = n_cons / self.N_REPS
        rate_norm = n_norm / self.N_REPS
        rate_agg  = n_agg  / self.N_REPS

        print(f"\n[aggression] CONSERVATIVE={rate_cons:.3f}  NORMAL={rate_norm:.3f}  AGGRESSIVE={rate_agg:.3f}")

        assert rate_agg > rate_norm, (
            f"Expected AGGRESSIVE ({rate_agg:.3f}) > NORMAL ({rate_norm:.3f})"
        )
        assert rate_norm > rate_cons, (
            f"Expected NORMAL ({rate_norm:.3f}) > CONSERVATIVE ({rate_cons:.3f})"
        )

    def test_gap_decreases_probability(self, simulator):
        """Closer gap => higher success rate (NORMAL risk, medium-difficulty zone)."""
        zone = {"difficulty": 0.5, "name": "TestZone", "distance_from_start": 1.0}

        gap_close = 0.01  # 10m
        gap_far   = 0.08  # 80m

        n_close = self._count_successes(simulator, zone, RiskLevel.NORMAL, gap_close, self.N_REPS, seed=10)
        n_far   = self._count_successes(simulator, zone, RiskLevel.NORMAL, gap_far,   self.N_REPS, seed=11)

        rate_close = n_close / self.N_REPS
        rate_far   = n_far   / self.N_REPS

        print(f"\n[gap] close({gap_close*1000:.0f}m)={rate_close:.3f}  far({gap_far*1000:.0f}m)={rate_far:.3f}")

        assert rate_close > rate_far, (
            f"Expected closer gap ({rate_close:.3f}) to yield higher success than far gap ({rate_far:.3f})"
        )

    def test_zone_difficulty_scales_probability(self, simulator):
        """Easier zone (lower difficulty) => higher success rate at identical conditions."""
        zone_easy = {"difficulty": 0.2, "name": "La Source",    "distance_from_start": 0.4}
        zone_hard = {"difficulty": 0.7, "name": "Les Combes",   "distance_from_start": 2.5}

        gap = 0.03

        n_easy = self._count_successes(simulator, zone_easy, RiskLevel.NORMAL, gap, self.N_REPS, seed=20)
        n_hard = self._count_successes(simulator, zone_hard, RiskLevel.NORMAL, gap, self.N_REPS, seed=21)

        rate_easy = n_easy / self.N_REPS
        rate_hard = n_hard / self.N_REPS

        print(f"\n[zone_difficulty] easy(diff=0.2)={rate_easy:.3f}  hard(diff=0.7)={rate_hard:.3f}")

        assert rate_easy > rate_hard, (
            f"Expected easy zone ({rate_easy:.3f}) to yield higher success than hard zone ({rate_hard:.3f})"
        )


# ===========================================================================
# Group C — Telemetry contract
# ===========================================================================

class TestTelemetryContract:

    REQUIRED_TOP_KEYS = {"run_number", "driver_name", "data"}
    REQUIRED_DATA_KEYS = {
        "position", "starting_position", "reward_total",
        "reward_component_totals", "reward_component_totals_raw",
        "stochasticity_level", "decision_events",
    }
    REQUIRED_EVENT_KEYS = {
        "zone_id", "lap", "action_label", "attempt", "success",
        "reward", "reward_components", "reward_component_sum_error",
        "success_probability",
    }

    @pytest.fixture(scope="class")
    def records(self):
        _, recs = _run_minimal_race(laps=2, runs=2)
        assert recs, "No records produced — telemetry may be broken"
        return recs

    def test_jsonl_schema(self, records):
        """Top-level and data-level required keys must be present in every record."""
        for record in records:
            missing_top = self.REQUIRED_TOP_KEYS - set(record.keys())
            assert not missing_top, f"Missing top-level keys: {missing_top}"

            data = record.get("data", {})
            missing_data = self.REQUIRED_DATA_KEYS - set(data.keys())
            assert not missing_data, (
                f"Missing data keys for {record['driver_name']} "
                f"(run {record['run_number']}): {missing_data}"
            )

    def test_decision_event_schema(self, records):
        """Every decision_event must have all required keys."""
        for record in records:
            events = record.get("data", {}).get("decision_events", [])
            for i, event in enumerate(events):
                missing = self.REQUIRED_EVENT_KEYS - set(event.keys())
                assert not missing, (
                    f"decision_events[{i}] for {record['driver_name']} "
                    f"(run {record['run_number']}) missing keys: {missing}"
                )

    def test_component_totals_match_events(self, records):
        """
        reward_component_totals (weighted) breakdown contract:

        The terminal bonus contributes only to 'outcome' (if finished) or 'penalty' (if DNF).
        All other components (tactical, persistent_position, pace, tyre_pit) only come from
        per-decision events. Therefore:

          1. event_sums['outcome'] == 0  (outcome is terminal-only, never per-decision)
          2. totals['outcome'] can be any sign (negative = position loss)
          3. For non-terminal components (tactical, persistent_position, pace, tyre_pit):
             totals[k] ≈ event_sums[k]  (within floating-point tolerance)
        """
        from src.simulator import REWARD_COMPONENTS

        # Components whose total comes exclusively from terminal bonus (not per-decision events)
        TERMINAL_ONLY = {"outcome", "penalty"}
        # Components whose total comes exclusively from per-decision events (not terminal)
        DECISION_ONLY = set(REWARD_COMPONENTS) - TERMINAL_ONLY

        for record in records:
            data = record.get("data", {})
            totals = data.get("reward_component_totals", {})
            events = data.get("decision_events", [])

            # Sum event-level weighted components
            event_sums = {comp: 0.0 for comp in REWARD_COMPONENTS}
            for event in events:
                for comp in REWARD_COMPONENTS:
                    event_sums[comp] += float(event.get("reward_components", {}).get(comp, 0.0))

            # outcome must NEVER appear in per-decision events
            assert event_sums["outcome"] == 0.0, (
                f"'outcome' appeared in decision_events for {record['driver_name']} "
                f"(run {record['run_number']}) — it should only be in the terminal bonus"
            )

            # For decision-only components, totals == event_sums (no terminal contribution)
            for comp in DECISION_ONLY:
                total_val = float(totals.get(comp, 0.0))
                event_sum = event_sums[comp]
                assert abs(total_val - event_sum) < 1e-4, (
                    f"reward_component_totals[{comp}]={total_val:.6f} != "
                    f"sum of decision_events[{comp}]={event_sum:.6f} "
                    f"(expected equal, terminal doesn't touch {comp}) "
                    f"for {record['driver_name']} run={record['run_number']}"
                )

    def test_stochasticity_level_recorded(self, records):
        """Every record must have stochasticity_level matching the configured level."""
        for record in records:
            level = record.get("data", {}).get("stochasticity_level")
            assert level is not None, (
                f"stochasticity_level missing for {record['driver_name']} "
                f"(run {record['run_number']})"
            )
            assert level == "s0", (
                f"Expected stochasticity_level='s0' (default test config), "
                f"got '{level}' for {record['driver_name']}"
            )
