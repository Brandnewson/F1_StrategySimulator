"""Minimal observation model for overtaking decisions."""

from dataclasses import dataclass

import numpy as np

DECISION_RANGE_KM = 0.2
MAX_GAP_FOR_DECISION_KM = 0.1


@dataclass
class DriverFeedback:
    """Decision-time observation passed into RL agents."""

    zone_distance_km: float
    gap_to_ahead_km: float
    zone_difficulty: float
    has_car_ahead: bool

    def to_vector(self, normalize: bool = True) -> np.ndarray:
        """Return [zone_distance_norm, gap_to_ahead_norm, zone_difficulty, has_car_ahead]."""
        if normalize:
            zone_distance_norm = float(np.clip(self.zone_distance_km / DECISION_RANGE_KM, 0.0, 1.0))
            gap_to_ahead_norm = float(np.clip(self.gap_to_ahead_km / MAX_GAP_FOR_DECISION_KM, 0.0, 1.0))
        else:
            zone_distance_norm = float(self.zone_distance_km)
            gap_to_ahead_norm = float(self.gap_to_ahead_km)

        features = [
            zone_distance_norm,
            gap_to_ahead_norm,
            float(np.clip(self.zone_difficulty, 0.0, 1.0)),
            1.0 if self.has_car_ahead else 0.0,
        ]
        return np.array(features, dtype=np.float32)

    @staticmethod
    def get_state_dim() -> int:
        return 4


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


def create_driver_feedback(driver, race_state, upcoming_zone) -> DriverFeedback:
    """Build minimal decision feedback for one driver."""
    zone_distance_km = DECISION_RANGE_KM
    zone_difficulty = 0.5
    if upcoming_zone is not None:
        zone_dist_from_start = float(upcoming_zone.get("distance_from_start", 0.0))
        zone_distance_km = zone_dist_from_start - driver.current_distance
        if zone_distance_km < 0:
            zone_distance_km += race_state.track_distance
        zone_difficulty = float(upcoming_zone.get("difficulty", 0.5))

    _, gap_to_ahead_km = _find_driver_ahead_on_track(driver, race_state)
    has_car_ahead = gap_to_ahead_km is not None

    return DriverFeedback(
        zone_distance_km=float(zone_distance_km),
        gap_to_ahead_km=float(gap_to_ahead_km if gap_to_ahead_km is not None else MAX_GAP_FOR_DECISION_KM),
        zone_difficulty=float(zone_difficulty),
        has_car_ahead=bool(has_car_ahead),
    )
