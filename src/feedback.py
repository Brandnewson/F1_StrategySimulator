"""Config-driven observation model for strategic decision feedback."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

DECISION_RANGE_KM = 0.2
MAX_GAP_FOR_DECISION_KM = 0.1

DEFAULT_FEEDBACK_CONTRACT: Dict[str, object] = {
    "schema_version": "v2_config_driven",
    "active_profile": "default",
    "features_by_complexity": {
        # Low-complexity default keeps tactical features and adds race context.
        "low": [
            "zone_distance_norm",
            "gap_to_ahead_norm",
            "zone_difficulty",
            "has_car_ahead",
            "current_position_norm",
            "laps_remaining_norm",
        ],
        # Future-ready placeholders. Not all are currently populated.
        "medium": [
            "zone_distance_norm",
            "gap_to_ahead_norm",
            "zone_difficulty",
            "has_car_ahead",
            "current_position_norm",
            "laps_remaining_norm",
            "gap_to_behind_norm",
            "nearest_rival_gap_norm",
            "traffic_density_norm",
        ],
        "high": [
            "zone_distance_norm",
            "gap_to_ahead_norm",
            "zone_difficulty",
            "has_car_ahead",
            "current_position_norm",
            "laps_remaining_norm",
            "gap_to_behind_norm",
            "nearest_rival_gap_norm",
            "traffic_density_norm",
            "tyre_age_norm",
            "tyre_compound_code",
            "degradation_estimate_norm",
            "pit_loss_estimate_norm",
            "laps_since_last_stop_norm",
            "weather_code",
            "safety_flag",
            "rivals_pit_summary_norm",
        ],
    },
}

SUPPORTED_FEEDBACK_FEATURES = {
    "zone_distance_norm",
    "gap_to_ahead_norm",
    "zone_difficulty",
    "has_car_ahead",
    "current_position_norm",
    "laps_remaining_norm",
    "gap_to_behind_norm",
    "nearest_rival_gap_norm",
    "traffic_density_norm",
    "tyre_age_norm",
    "tyre_compound_code",
    "degradation_estimate_norm",
    "pit_loss_estimate_norm",
    "laps_since_last_stop_norm",
    "weather_code",
    "safety_flag",
    "rivals_pit_summary_norm",
}


def _deep_merge(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_complexity_name(config: Optional[Dict], complexity: Optional[str]) -> str:
    if complexity:
        return str(complexity).strip().lower() or "low"
    if isinstance(config, dict):
        return str(config.get("complexity", {}).get("active_profile", "low")).strip().lower() or "low"
    return "low"


def resolve_feedback_contract(
    config: Optional[Dict], complexity: Optional[str] = None
) -> Tuple[str, List[str], Dict[str, object]]:
    """Resolve active feedback schema and feature list from config."""
    cfg = config if isinstance(config, dict) else {}
    user_feedback = cfg.get("feedback", {})
    merged = _deep_merge(
        DEFAULT_FEEDBACK_CONTRACT,
        user_feedback if isinstance(user_feedback, dict) else {},
    )
    complexity_name = _resolve_complexity_name(cfg, complexity)
    by_complexity = merged.get("features_by_complexity", {})
    if not isinstance(by_complexity, dict):
        raise TypeError("feedback.features_by_complexity must be an object")
    features = by_complexity.get(complexity_name) or by_complexity.get("low") or []
    if not isinstance(features, list) or not features:
        raise ValueError(f"No feedback features configured for complexity '{complexity_name}'")

    normalized_features = [str(f).strip() for f in features if str(f).strip()]
    unknown = [f for f in normalized_features if f not in SUPPORTED_FEEDBACK_FEATURES]
    if unknown:
        raise ValueError(
            f"Unsupported feedback features for complexity '{complexity_name}': {unknown}. "
            f"Supported features: {sorted(SUPPORTED_FEEDBACK_FEATURES)}"
        )
    return str(merged.get("schema_version", "v2_config_driven")), normalized_features, merged


@dataclass
class DriverFeedback:
    """Decision-time observation passed into RL agents."""

    feature_values: Dict[str, float]
    active_features: List[str]

    def to_vector(self, normalize: bool = True) -> np.ndarray:
        """Return feature vector in configured feature order."""
        _ = normalize  # Features are pre-normalized in this schema.
        values = [float(self.feature_values.get(name, 0.0)) for name in self.active_features]
        return np.array(values, dtype=np.float32)

    @staticmethod
    def get_state_dim(config: Optional[Dict] = None, complexity: Optional[str] = None) -> int:
        _, features, _ = resolve_feedback_contract(config, complexity=complexity)
        return len(features)


def _find_driver_ahead_on_track(driver, race_state):
    """Return nearest driver ahead and track gap in km."""
    driver_total = driver.completed_laps * race_state.track_distance + driver.current_distance
    closest = None
    min_gap = None
    for other in race_state.drivers:
        if other is driver:
            continue
        other_total = other.completed_laps * race_state.track_distance + other.current_distance
        gap = other_total - driver_total
        if gap <= 0:
            continue
        if min_gap is None or gap < min_gap:
            min_gap = gap
            closest = other
    return closest, float(min_gap) if min_gap is not None else None


def _find_driver_behind_on_track(driver, race_state):
    """Return nearest driver behind and track gap in km."""
    driver_total = driver.completed_laps * race_state.track_distance + driver.current_distance
    closest = None
    min_gap = None
    for other in race_state.drivers:
        if other is driver:
            continue
        other_total = other.completed_laps * race_state.track_distance + other.current_distance
        gap = driver_total - other_total
        if gap <= 0:
            continue
        if min_gap is None or gap < min_gap:
            min_gap = gap
            closest = other
    return closest, float(min_gap) if min_gap is not None else None


def _safe_total_laps(race_state) -> int:
    cfg = race_state.config if isinstance(getattr(race_state, "config", {}), dict) else {}
    total_laps = int(cfg.get("race_settings", {}).get("total_laps", 0) or 0)
    if total_laps <= 0:
        # Fallback for malformed config; keeps normalization stable.
        total_laps = max(1, int(getattr(race_state, "elapsed_time", 0) // 1) + 1)
    return total_laps


def _weather_code_from_config(config: Dict[str, object]) -> float:
    weather = str(config.get("race_settings", {}).get("weather", "dry")).strip().lower()
    mapping = {"dry": 0.0, "mixed": 0.5, "wet": 1.0}
    return float(mapping.get(weather, 0.0))


def create_driver_feedback(
    driver,
    race_state,
    upcoming_zone,
    config: Optional[Dict] = None,
    complexity: Optional[str] = None,
) -> DriverFeedback:
    """Build config-driven decision feedback for one driver."""
    cfg = config if isinstance(config, dict) else getattr(race_state, "config", {})
    _, active_features, _ = resolve_feedback_contract(cfg, complexity=complexity)

    zone_distance_km = DECISION_RANGE_KM
    zone_difficulty = 0.5
    if upcoming_zone is not None:
        zone_dist_from_start = float(upcoming_zone.get("distance_from_start", 0.0))
        zone_distance_km = zone_dist_from_start - driver.current_distance
        if zone_distance_km < 0:
            zone_distance_km += race_state.track_distance
        zone_difficulty = float(upcoming_zone.get("difficulty", 0.5))

    _, gap_to_ahead_km = _find_driver_ahead_on_track(driver, race_state)
    _, gap_to_behind_km = _find_driver_behind_on_track(driver, race_state)
    has_car_ahead = gap_to_ahead_km is not None

    num_drivers = max(1, len(getattr(race_state, "drivers", [])))
    denom_positions = max(1, num_drivers - 1)
    current_position_norm = float(np.clip((float(getattr(driver, "position", num_drivers)) - 1.0) / denom_positions, 0.0, 1.0))

    total_laps = _safe_total_laps(race_state)
    laps_remaining = max(0, total_laps - int(getattr(driver, "completed_laps", 0)))
    laps_remaining_norm = float(np.clip(laps_remaining / float(max(1, total_laps)), 0.0, 1.0))

    gap_ahead_for_norm = float(gap_to_ahead_km if gap_to_ahead_km is not None else MAX_GAP_FOR_DECISION_KM)
    gap_behind_for_norm = float(gap_to_behind_km if gap_to_behind_km is not None else MAX_GAP_FOR_DECISION_KM)
    nearest_rival_gap_km = min(gap_ahead_for_norm, gap_behind_for_norm)
    traffic_density_norm = float(np.clip(1.0 - (nearest_rival_gap_km / MAX_GAP_FOR_DECISION_KM), 0.0, 1.0))

    feature_values: Dict[str, float] = {
        "zone_distance_norm": float(np.clip(zone_distance_km / DECISION_RANGE_KM, 0.0, 1.0)),
        "gap_to_ahead_norm": float(np.clip(gap_ahead_for_norm / MAX_GAP_FOR_DECISION_KM, 0.0, 1.0)),
        "zone_difficulty": float(np.clip(zone_difficulty, 0.0, 1.0)),
        "has_car_ahead": 1.0 if has_car_ahead else 0.0,
        "current_position_norm": current_position_norm,
        "laps_remaining_norm": laps_remaining_norm,
        # Future-ready placeholders below
        "gap_to_behind_norm": float(np.clip(gap_behind_for_norm / MAX_GAP_FOR_DECISION_KM, 0.0, 1.0)),
        "nearest_rival_gap_norm": float(np.clip(nearest_rival_gap_km / MAX_GAP_FOR_DECISION_KM, 0.0, 1.0)),
        "traffic_density_norm": traffic_density_norm,
        "tyre_age_norm": float(np.clip(float(getattr(driver, "tyre_age", 0)) / float(max(1, total_laps)), 0.0, 1.0)),
        "tyre_compound_code": 0.5,  # Placeholder encoding; low complexity keeps tyre controls inactive.
        "degradation_estimate_norm": 0.0,
        "pit_loss_estimate_norm": 0.0,
        "laps_since_last_stop_norm": 0.0,
        "weather_code": _weather_code_from_config(cfg if isinstance(cfg, dict) else {}),
        "safety_flag": 0.0,
        "rivals_pit_summary_norm": 0.0,
    }

    return DriverFeedback(feature_values=feature_values, active_features=active_features)
