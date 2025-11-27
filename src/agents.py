from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple
import numpy as np


class RiskLevel(Enum):
    CONSERVATIVE = 0
    NORMAL = 1
    AGGRESSIVE = 2


@dataclass
class DriverAction:
    risk_level: RiskLevel
    attempt_overtake: bool


# Decision point: 0.1 km before overtaking point
DECISION_DISTANCE_KM = 0.1


def find_upcoming_zone_for_driver(race_state, driver, lookahead: float = DECISION_DISTANCE_KM):
    """Find the first overtaking zone ahead of the driver within `lookahead` km.

    Returns the zone dict or None.
    """
    if not hasattr(race_state, "overtaking_zones"):
        return None

    cur_dist = getattr(driver, "current_distance", None)
    if cur_dist is None:
        return None

    # Find the nearest zone ahead within lookahead
    for zone in race_state.overtaking_zones:
        try:
            zdist = float(zone.get("distance_from_start", 0.0))
        except Exception:
            continue

        if 0.0 < (zdist - cur_dist) <= lookahead:
            return zone

    return None


def default_policy(driver, race_state) -> Tuple[RiskLevel, bool]:
    """Default heuristic policy that selects a risk level and whether to attempt an overtake.

    This is intentionally simple: it (a) finds an upcoming overtaking zone, (b) computes
    a small probability distribution over risk levels influenced by zone difficulty and gap,
    and (c) returns a sampled risk level and a probabilistic overtake decision.

    RL policies can replace this by passing a custom callable to BaseAgent.
    """
    zone = find_upcoming_zone_for_driver(race_state, driver)

    # default to normal behaviour when no zone
    if zone is None:
        return RiskLevel.NORMAL, False

    difficulty = float(zone.get("difficulty", 0.5))
    gap = getattr(driver, "gap_to_ahead", None)

    # Build a simple preference for risk levels (conservative -> aggressive)
    # easier zones favour aggressive choices, harder zones favour conservative
    # base preferences: [cons, norm, aggr]
    ease = 1.0 - difficulty
    pref_cons = max(0.05, 0.6 * (1.0 - ease))
    pref_norm = max(0.1, 0.6 * 0.5)
    pref_aggr = max(0.05, 0.6 * ease)

    # If gap is very small, favour aggressive (to overtake)
    if gap is not None:
        if gap < 0.6:
            pref_aggr *= 1.6
            pref_cons *= 0.6
        elif gap > 2.0:
            pref_cons *= 1.4
            pref_aggr *= 0.6

    prefs = np.array([pref_cons, pref_norm, pref_aggr], dtype=float)
    prefs = np.clip(prefs, 1e-3, None)
    probs = prefs / prefs.sum()

    chosen = np.random.choice([RiskLevel.CONSERVATIVE, RiskLevel.NORMAL, RiskLevel.AGGRESSIVE], p=probs)

    # Determine overtake attempt probability based on chosen risk and zone difficulty
    if chosen == RiskLevel.CONSERVATIVE:
        base_prob = 0.35 * (1.0 - difficulty)
    elif chosen == RiskLevel.NORMAL:
        base_prob = 0.6 * (1.0 - difficulty)
    else:  # AGGRESSIVE
        base_prob = 0.9 * (1.0 - 0.5 * difficulty)

    # gap modifies attempt chance
    if gap is None:
        attempt_prob = base_prob * 0.4
    else:
        if gap < 0.5:
            attempt_prob = min(0.99, base_prob * 1.6)
        elif gap < 1.5:
            attempt_prob = base_prob * 1.0
        else:
            attempt_prob = base_prob * 0.3

    attempt = np.random.random() < attempt_prob

    return chosen, bool(attempt)


class BaseAgent:
    """Flexible agent that chooses among risk levels and whether to attempt overtakes.

    By default the agent uses `default_policy`. Supply a custom `policy` callable with the
    signature `(driver, race_state) -> (RiskLevel, bool)` to override behaviour (useful for RL).
    """

    def __init__(self, name: str = "BaseAgent", policy: Optional[Callable] = None):
        self.name = name
        self.policy = policy if policy is not None else default_policy

    def set_policy(self, policy: Callable):
        """Set a new policy callable for this agent."""
        self.policy = policy

    def get_action(self, driver, race_state) -> DriverAction:
        """Compute action using the configured policy."""
        risk, attempt = self.policy(driver, race_state)
        # Ensure risk is a RiskLevel and attempt is bool
        if not isinstance(risk, RiskLevel):
            # allow numeric or string inputs
            try:
                risk = RiskLevel(risk)
            except Exception:
                risk = RiskLevel.NORMAL
        return DriverAction(risk_level=risk, attempt_overtake=bool(attempt))


# Small convenience factory functions
def make_random_agent(name: str = "RandomAgent") -> BaseAgent:
    """Agent with a purely random policy over risk levels + small overtake chance."""

    def rand_policy(driver, race_state):
        risk = np.random.choice(list(RiskLevel))
        attempt = np.random.random() < 0.3
        return risk, attempt

    return BaseAgent(name=name, policy=rand_policy)


def make_param_agent(name: str, cons_weight: float, norm_weight: float, aggr_weight: float) -> BaseAgent:
    """Create an agent that deterministically prefers risk levels according to weights.

    The agent still uses heuristics for attempt probability based on chosen risk and zone difficulty.
    This is useful for testing or as a deterministic policy baseline.
    """
    weights = np.array([cons_weight, norm_weight, aggr_weight], dtype=float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0:
        weights = np.array([0.33, 0.34, 0.33])
    probs = weights / weights.sum()

    def param_policy(driver, race_state):
        chosen = np.random.choice(list(RiskLevel), p=probs)
        # reuse default attempt logic for compatibility
        zone = find_upcoming_zone_for_driver(race_state, driver)
        if zone is None:
            return chosen, False
        difficulty = float(zone.get("difficulty"))
        gap = getattr(driver, "gap_to_ahead", None)
        # map chosen -> base_prob
        if chosen == RiskLevel.CONSERVATIVE:
            base_prob = 0.25 * (1.0 - difficulty)
        elif chosen == RiskLevel.NORMAL:
            base_prob = 0.6 * (1.0 - difficulty)
        else:
            base_prob = 0.9 * (1.0 - 0.5 * difficulty)
        if gap is None:
            attempt_prob = base_prob * 0.4
        else:
            if gap < 0.6:
                attempt_prob = min(0.99, base_prob * 1.6)
            elif gap < 1.8:
                attempt_prob = base_prob
            else:
                attempt_prob = base_prob * 0.3
        attempt = np.random.random() < attempt_prob
        return chosen, bool(attempt)

    return BaseAgent(name=name, policy=param_policy)
