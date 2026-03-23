"""Runtime profile resolution for simulator complexity."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple


DEFAULT_COMPLEXITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "low": {
        "implemented": True,
        "description": "Single DQN driver against one Base driver.",
    },
    "medium": {
        "implemented": False,
        "description": "Multiple competitors in one race.",
    },
    "high": {
        "implemented": False,
        "description": "Tyre dynamics and pit stops.",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_complexity_profile(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Resolve active simulator complexity.

    Medium and high are intentionally unimplemented for now.
    Any unimplemented selection falls back to low.
    """
    complexity_cfg = config.get("complexity", {}) if isinstance(config, dict) else {}
    user_profiles = complexity_cfg.get("profiles", {})
    merged_profiles = _deep_merge(
        DEFAULT_COMPLEXITY_PROFILES,
        user_profiles if isinstance(user_profiles, dict) else {},
    )

    requested = str(complexity_cfg.get("active_profile", "low")).strip().lower() or "low"
    if requested not in merged_profiles:
        requested = "low"

    profile = _deep_merge(DEFAULT_COMPLEXITY_PROFILES["low"], merged_profiles.get(requested, {}))
    if not bool(profile.get("implemented", False)):
        requested = "low"
        profile = _deep_merge(DEFAULT_COMPLEXITY_PROFILES["low"], merged_profiles.get("low", {}))

    return requested, profile, merged_profiles


def resolve_compound_name(raw_name: Any, compounds: Dict[str, Dict[str, Any]], default_compound: str = "medium") -> str:
    """Return a supported compound name.

    This helper is kept for future high-complexity tyre work.
    """
    name = str(raw_name).strip().lower() if raw_name is not None else ""
    if name in compounds:
        return name
    return str(default_compound).strip().lower() or "medium"


def select_low_complexity_competitors(competitors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select one DQN competitor and one Base competitor for low complexity mode."""
    dqn_competitor = None
    base_competitor = None
    for comp in competitors:
        agent = str(comp.get("agent", "")).strip().lower()
        if agent == "dqn" and dqn_competitor is None:
            dqn_competitor = deepcopy(comp)
        if agent == "base" and base_competitor is None:
            base_competitor = deepcopy(comp)
        if dqn_competitor is not None and base_competitor is not None:
            break

    if dqn_competitor is None:
        raise ValueError("Low complexity mode requires at least one competitor with agent='dqn'.")
    if base_competitor is None:
        raise ValueError("Low complexity mode requires at least one competitor with agent='base'.")

    return [dqn_competitor, base_competitor]

