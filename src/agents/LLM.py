"""LLM-based strategic agent using Anthropic's Haiku model.

Two-tier architecture:
  1. Strategic layer (1 API call per lap): produces a conditional action table
     mapping each overtaking zone to actions based on gap buckets.
  2. Tactical layer (no API call): deterministic lookup from the action table.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv

from base_agents import BaseAgent, DriverAction, RiskLevel
from feedback import create_driver_feedback

# Action index mapping — matches DQNAgent.ACTION_SPACE exactly
ACTION_MAP: List[Tuple[RiskLevel, bool]] = [
    (RiskLevel.NORMAL, False),       # 0 = HOLD
    (RiskLevel.CONSERVATIVE, True),  # 1 = ATTEMPT_CONSERVATIVE
    (RiskLevel.NORMAL, True),        # 2 = ATTEMPT_NORMAL
    (RiskLevel.AGGRESSIVE, True),    # 3 = ATTEMPT_AGGRESSIVE
]
ACTION_LABELS = ["HOLD", "CONSERVATIVE", "NORMAL", "AGGRESSIVE"]

# Default conservative plan used as fallback
DEFAULT_ACTION = 0  # HOLD


class LLMAgent(BaseAgent):
    """LLM-based F1 strategy agent with per-lap planning."""

    def __init__(
        self,
        config: dict,
        name: str = "LLMAgent",
        alpha_mode: str = "competitive",
    ):
        super().__init__(name=name)
        self.config = config

        # Load API key from .env
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found. Ensure .env file exists in project root."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

        # LLM config
        llm_params = config.get("llm_params", {})
        self.model = str(llm_params.get("model", "claude-haiku-4-5-20251001"))
        self.max_retries = int(llm_params.get("max_retries", 1))
        self.temperature = float(llm_params.get("temperature", 0.0))
        self.alpha_mode = alpha_mode

        # Extract zone info from track config
        self._zone_list = self._extract_zones(config)

        # Build sanitized key ↔ real name mappings for tool schema
        self._key_to_name: Dict[str, str] = {}
        self._name_to_key: Dict[str, str] = {}
        for z in self._zone_list:
            key = self._sanitize_key(z["name"])
            self._key_to_name[key] = z["name"]
            self._name_to_key[z["name"]] = key

        # Build prompts and tool schema
        self._system_prompt = self._build_system_prompt()
        self._tool_schema = self._build_strategic_tool()

        # State — reset per race
        self.zone_plan: Dict[str, Dict[str, int]] = {}
        self._current_lap = -1
        self._last_lap_events: List[Dict[str, Any]] = []

        # Logging — accumulates across races
        self.strategic_log: List[Dict[str, Any]] = []
        self.tactical_log: List[Dict[str, Any]] = []
        self._cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    # ------------------------------------------------------------------
    # Zone extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_key(name: str) -> str:
        """Convert zone name to API-safe property key (alphanumeric + underscores)."""
        import re
        key = re.sub(r"[^a-zA-Z0-9]", "_", name.strip())
        key = re.sub(r"_+", "_", key).strip("_")
        return key[:64]

    @staticmethod
    def _extract_zones(config: dict) -> List[Dict[str, Any]]:
        """Pull zone names, difficulties, and distances from track config."""
        track_cfg = config.get("track", {})
        oz_cfg = track_cfg.get("overtakingZones", {})
        zones = []
        for key in sorted(oz_cfg.keys()):
            z = oz_cfg[key]
            zones.append({
                "name": str(z.get("name", key)).strip(),
                "difficulty": float(z.get("difficulty", 0.5)),
                "distance": float(z.get("distance_from_start", 0.0)),
            })
        return zones

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        # Build zone table with distances and gap-to-next
        zone_lines = []
        for i, z in enumerate(self._zone_list):
            key = self._name_to_key[z["name"]]
            gap_str = ""
            if i < len(self._zone_list) - 1:
                gap = self._zone_list[i + 1]["distance"] - z["distance"]
                gap_str = f"  (gap to next: {gap:.1f}km)"
            zone_lines.append(
                f"  {key}: {z['name']} | dist={z['distance']:.1f}km | "
                f"difficulty={z['difficulty']:.1f}{gap_str}"
            )
        zone_desc = "\n".join(zone_lines)

        alpha_instruction = {
            "competitive": (
                "Your sole objective is to finish in the best possible position. "
                "You have no teammates — every other car is a rival."
            ),
            "partial": (
                "You are racing with a teammate. Balance finishing as high as "
                "possible yourself with helping your teammate finish well. "
                "Avoid blocking or aggressively fighting your teammate."
            ),
            "cooperative": (
                "You and your teammate share the same goal: maximise the team's "
                "combined finishing positions. Your teammate's result matters "
                "equally to your own. Coordinate to beat the other competitors."
            ),
        }.get(self.alpha_mode, "Maximise your own finishing position.")

        return f"""\
You are an expert Formula 1 race strategist managing a car at Spa-Francorchamps.

TRACK ZONES (in lap order):
{zone_desc}

ACTIONS (integer codes):
  0 = HOLD — do not attempt an overtake
  1 = CONSERVATIVE — cautious attempt. ~15% success bonus. On failure: lose 0.02km.
  2 = NORMAL — standard attempt. ~25% success bonus. On failure: lose 0.05km.
  3 = AGGRESSIVE — bold attempt. ~40% success bonus. On failure: lose 0.08km.

SUCCESS PROBABILITY:
  Base probability ≈ (1 - difficulty). Risk level adds a bonus on top.
  - La Source (difficulty 0.2): ~80% base. NORMAL close ≈ 90%+ success.
  - Difficulty 0.7 zones: ~30% base. Even AGGRESSIVE close ≈ 55% success.
  - Raidillon (difficulty 0.9): ~10% base. Attempts here are almost always wasted.
  "far" gap reduces success by roughly half compared to "close".

FAILURE CONSEQUENCES:
  A failed attempt costs distance. AGGRESSIVE failure loses 0.08km — enough to \
create a gap the opponent can exploit. Multiple failed aggressive attempts in \
one lap can lose you the race. Use AGGRESSIVE sparingly and only where success \
probability justifies the risk.

COOLDOWN:
  After any attempt (success or failure), there is a cooldown period before you \
can attempt again. Zones that are close together on track cannot both be attempted \
in the same pass. Look at the gap-to-next distances:
  - Campus→Stavelot: only 0.3km apart — you can only attempt at ONE of these.
  - Stavelot→Blanchimont: 1.0km — enough gap to attempt both.
  - Choose the zone in each cluster that gives you the best risk/reward.

STRATEGIC PRINCIPLES:
  - La Source is your best overtaking opportunity (low difficulty, first zone each lap).
  - Raidillon is almost never worth attempting — the risk/reward is terrible.
  - Among the 0.7-difficulty zones, prefer zones with more run-off to the next zone.
  - When leading (P1), HOLD everywhere — you cannot overtake if nobody is ahead.
  - When behind with few laps left, take calculated risks at your best zones, \
not blind aggression everywhere.
  - Adapt based on what worked last lap. If an attempt failed, consider a different \
zone or lower risk next time.

GAP BUCKETS:
  "close" = car ahead within 0.05km (genuine overtaking opportunity)
  "far" = car ahead 0.05–0.1km (marginal, success roughly halved)

OBJECTIVE: {alpha_instruction}

Each lap you receive your position, laps remaining, and last lap results. \
Produce an action plan for every zone. Think strategically about which specific \
zones to target rather than applying the same plan to all similar-difficulty zones."""

    def _build_strategic_tool(self) -> dict:
        """Build the tool_use schema that forces structured zone plan output."""
        zone_properties = {}
        required_zones = []
        for z in self._zone_list:
            key = self._name_to_key[z["name"]]
            zone_properties[key] = {
                "type": "object",
                "properties": {
                    "close": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": f"Action at {z['name']} when car ahead within 0.05km",
                    },
                    "far": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": f"Action at {z['name']} when car ahead 0.05-0.1km",
                    },
                },
                "required": ["close", "far"],
            }
            required_zones.append(key)

        return {
            "name": "set_lap_plan",
            "description": "Set the overtaking action plan for each zone this lap.",
            "input_schema": {
                "type": "object",
                "properties": zone_properties,
                "required": required_zones,
            },
        }

    def _build_user_message(self, driver, race_state) -> str:
        """Build the per-lap user prompt with current race context."""
        total_laps = self._get_total_laps(race_state)
        current_lap = int(getattr(driver, "completed_laps", 0))
        laps_remaining = max(0, total_laps - current_lap)
        position = int(getattr(driver, "position", 1))
        num_drivers = len(getattr(race_state, "drivers", []))

        msg = (
            f"Lap {current_lap + 1} of {total_laps}. "
            f"You are P{position} of {num_drivers}. "
            f"{laps_remaining} laps remaining."
        )

        # Position context
        if position == 1:
            msg += " You are LEADING — you only need to defend, not overtake."
        elif laps_remaining <= 2 and position > 1:
            msg += " URGENT — few laps left and you need to gain position."
        elif position > 1:
            msg += " You need to overtake the car ahead."

        # Include last lap feedback if available
        if self._last_lap_events:
            attempts = [e for e in self._last_lap_events if e.get("action", 0) > 0]
            successes = [e for e in attempts if e.get("success")]
            fails = [e for e in attempts if not e.get("success")]

            msg += f"\n\nLast lap: {len(attempts)} attempts, "
            msg += f"{len(successes)} succeeded, {len(fails)} failed."

            if attempts:
                msg += "\nDetails:"
                for evt in self._last_lap_events:
                    if evt.get("action", 0) == 0:
                        continue  # Skip HOLDs in feedback
                    outcome = "SUCCESS" if evt.get("success") else "FAILED"
                    action_name = ACTION_LABELS[evt.get("action", 0)]
                    zone = evt.get("zone", "?")
                    gap_bucket = evt.get("bucket", "?")
                    msg += f"\n  {zone} ({gap_bucket}): {action_name} → {outcome}"

                if fails:
                    msg += "\nConsider adjusting risk at zones where you failed."

        return msg

    @staticmethod
    def _get_total_laps(race_state) -> int:
        cfg = getattr(race_state, "config", {})
        if isinstance(cfg, dict):
            return int(cfg.get("race_settings", {}).get("total_laps", 5) or 5)
        return 5

    # ------------------------------------------------------------------
    # Strategic call (1 per lap)
    # ------------------------------------------------------------------

    def _update_strategy(self, driver, race_state) -> None:
        """Call Haiku to get a new zone plan for this lap."""
        # Pull last-lap outcomes from the driver's decision event history
        current_lap = int(getattr(driver, "completed_laps", 0))
        prev_lap = current_lap - 1
        self._last_lap_events = []
        if prev_lap >= 0 and hasattr(driver, "decision_events"):
            for evt in driver.decision_events:
                if evt.get("decision_lap") == prev_lap:
                    # Map simulator labels like "ATTEMPT_NORMAL" → "NORMAL"
                    raw_label = evt.get("action_label", "HOLD")
                    short_label = raw_label.replace("ATTEMPT_", "") if raw_label != "HOLD" else "HOLD"
                    action_idx = ACTION_LABELS.index(short_label) if short_label in ACTION_LABELS else 0
                    self._last_lap_events.append({
                        "zone": evt.get("zone_name", "?"),
                        "bucket": "close" if evt.get("gap_to_ahead_km", 1.0) < 0.05 else "far",
                        "action": action_idx,
                        "action_label": short_label,
                        "success": evt.get("success", False),
                    })

        user_msg = self._build_user_message(driver, race_state)

        for attempt in range(1 + self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=self.temperature,
                    system=self._system_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                    tools=[self._tool_schema],
                    tool_choice={"type": "tool", "name": "set_lap_plan"},
                )

                # Track cost
                usage = response.usage
                self._cost["calls"] += 1
                self._cost["input_tokens"] += usage.input_tokens
                self._cost["output_tokens"] += usage.output_tokens

                # Parse tool call response
                plan = self._parse_tool_response(response)
                if plan is not None:
                    self.zone_plan = plan
                    self._log_strategic(driver, race_state, plan, user_msg)
                    return

            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(0.5)
                    continue
                print(f"[LLMAgent] API error after {attempt + 1} attempts: {e}")

        # Fallback: keep previous plan or use default conservative
        if not self.zone_plan:
            self.zone_plan = self._default_plan()

    def _parse_tool_response(self, response) -> Optional[Dict[str, Dict[str, int]]]:
        """Extract zone plan from Haiku's tool_use response (sanitized keys → real names)."""
        for block in response.content:
            if block.type == "tool_use" and block.name == "set_lap_plan":
                raw = block.input
                plan = {}
                for z in self._zone_list:
                    zname = z["name"]
                    key = self._name_to_key[zname]
                    zone_data = raw.get(key, {})
                    plan[zname] = {
                        "close": max(0, min(3, int(zone_data.get("close", DEFAULT_ACTION)))),
                        "far": max(0, min(3, int(zone_data.get("far", DEFAULT_ACTION)))),
                    }
                return plan
        return None

    def _default_plan(self) -> Dict[str, Dict[str, int]]:
        """Conservative fallback plan — attempt at easy zones only."""
        plan = {}
        for z in self._zone_list:
            if z["difficulty"] <= 0.3:
                plan[z["name"]] = {"close": 2, "far": 0}
            else:
                plan[z["name"]] = {"close": 0, "far": 0}
        return plan

    # ------------------------------------------------------------------
    # Tactical execution (no API call)
    # ------------------------------------------------------------------

    def get_action(self, driver, race_state, upcoming_zone=None, **kwargs) -> DriverAction:
        """Look up action from the current zone plan."""
        current_lap = int(getattr(driver, "completed_laps", 0))

        # Detect new race: lap counter went backwards
        if current_lap < self._current_lap:
            self._reset_for_race()

        # New lap → call strategic layer
        if current_lap != self._current_lap:
            self._update_strategy(driver, race_state)
            if current_lap > self._current_lap and self._current_lap >= 0:
                # Clear last lap events for the new cycle
                self._last_lap_events = []
            self._current_lap = current_lap

        # Determine gap bucket
        feedback = create_driver_feedback(driver, race_state, upcoming_zone)
        has_car = feedback.feature_values.get("has_car_ahead", 0.0)
        gap_norm = feedback.feature_values.get("gap_to_ahead_norm", 1.0)

        if has_car < 0.5:
            bucket = "no_car"
        elif gap_norm < 0.5:
            bucket = "close"
        else:
            bucket = "far"

        # Look up zone name
        zone_name = "unknown"
        if upcoming_zone is not None:
            zone_name = str(upcoming_zone.get("name", "unknown")).strip()

        # Resolve action
        if bucket == "no_car":
            action_idx = DEFAULT_ACTION  # HOLD
        else:
            zone_entry = self.zone_plan.get(zone_name, {})
            action_idx = zone_entry.get(bucket, DEFAULT_ACTION)
            action_idx = max(0, min(3, action_idx))

        risk, attempt = ACTION_MAP[action_idx]
        action = DriverAction(risk_level=risk, attempt_overtake=attempt)

        # Log tactical decision
        self._log_tactical(zone_name, bucket, action_idx, current_lap)

        return action

    def get_last_decision_context(self) -> None:
        """Required by simulator interface. LLM agent has no replay buffer."""
        return None

    # ------------------------------------------------------------------
    # Race reset (self-detecting)
    # ------------------------------------------------------------------

    def _reset_for_race(self) -> None:
        """Clear per-race state. Logs and cost accumulate across races."""
        self.zone_plan = {}
        self._current_lap = -1
        self._last_lap_events = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_strategic(self, driver, race_state, plan, user_msg) -> None:
        entry = {
            "race_tick": int(getattr(race_state, "current_tick", 0)),
            "lap": int(getattr(driver, "completed_laps", 0)),
            "position": int(getattr(driver, "position", 0)),
            "laps_remaining": self._get_total_laps(race_state) - int(getattr(driver, "completed_laps", 0)),
            "plan": {k: dict(v) for k, v in plan.items()},
            "user_message": user_msg,
        }
        self.strategic_log.append(entry)

    def _log_tactical(self, zone_name, bucket, action_idx, lap) -> None:
        entry = {
            "lap": lap,
            "zone": zone_name,
            "bucket": bucket,
            "action": action_idx,
            "action_label": ACTION_LABELS[action_idx],
        }
        self.tactical_log.append(entry)
        # Also track for last-lap feedback to the strategic layer
        self._last_lap_events.append(entry)

    def get_strategic_log(self) -> List[Dict[str, Any]]:
        return list(self.strategic_log)

    def get_tactical_log(self) -> List[Dict[str, Any]]:
        return list(self.tactical_log)

    def get_cost_summary(self) -> Dict[str, Any]:
        # Haiku pricing (approximate): $0.25/MTok input, $1.25/MTok output
        est_cost = (
            self._cost["input_tokens"] * 0.25 / 1_000_000
            + self._cost["output_tokens"] * 1.25 / 1_000_000
        )
        return {
            "total_calls": self._cost["calls"],
            "input_tokens": self._cost["input_tokens"],
            "output_tokens": self._cost["output_tokens"],
            "estimated_cost_usd": round(est_cost, 4),
        }
