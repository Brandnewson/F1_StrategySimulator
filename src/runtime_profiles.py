"""Runtime profile resolution for simulator complexity."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple


DEFAULT_COMPLEXITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "low": {
        "implemented": True,
        "description": "Single DQN driver against one Base driver.",
    },
    "low_marl": {
        "implemented": True,
        "description": "Two DQN drivers competing against each other (independent learner MARL).",
    },
    "low_marl_vs_base": {
        "implemented": True,
        "description": "Two DQN drivers + one Base adversary (non-zero-sum MARL).",
    },
    "low_marl_3dqn_vs_base": {
        "implemented": True,
        "description": "Three DQN drivers + one Base adversary (3-vs-1 non-zero-sum MARL).",
    },
    "low_marl_teams": {
        "implemented": True,
        "description": "Two teams of 2 DQN drivers + one Base adversary (team-based MARL).",
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


def select_low_marl_competitors(competitors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select exactly two DQN competitors for low-complexity MARL mode."""
    dqn_competitors = []
    for comp in competitors:
        agent = str(comp.get("agent", "")).strip().lower()
        if agent == "dqn":
            dqn_competitors.append(deepcopy(comp))
        if len(dqn_competitors) == 2:
            break

    if len(dqn_competitors) < 2:
        raise ValueError(
            f"low_marl complexity mode requires at least two competitors with agent='dqn'. "
            f"Found {len(dqn_competitors)}."
        )
    return dqn_competitors


def select_low_marl_vs_base_competitors(competitors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select exactly two DQN competitors and one Base competitor for non-zero-sum MARL mode."""
    dqn_competitors = []
    base_competitor = None
    for comp in competitors:
        agent = str(comp.get("agent", "")).strip().lower()
        if agent == "dqn" and len(dqn_competitors) < 2:
            dqn_competitors.append(deepcopy(comp))
        if agent == "base" and base_competitor is None:
            base_competitor = deepcopy(comp)

    if len(dqn_competitors) < 2:
        raise ValueError(
            f"low_marl_vs_base complexity mode requires at least two competitors with agent='dqn'. "
            f"Found {len(dqn_competitors)}."
        )
    if base_competitor is None:
        raise ValueError(
            "low_marl_vs_base complexity mode requires exactly one competitor with agent='base'."
        )
    return dqn_competitors + [base_competitor]


def select_low_marl_3dqn_vs_base_competitors(competitors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select exactly three DQN competitors and one Base competitor for 3-vs-1 non-zero-sum MARL."""
    dqn_competitors: List[Dict[str, Any]] = []
    base_competitor = None
    for comp in competitors:
        agent = str(comp.get("agent", "")).strip().lower()
        if agent == "dqn" and len(dqn_competitors) < 3:
            dqn_competitors.append(deepcopy(comp))
        if agent == "base" and base_competitor is None:
            base_competitor = deepcopy(comp)

    if len(dqn_competitors) < 3:
        raise ValueError(
            f"low_marl_3dqn_vs_base complexity mode requires at least three competitors with agent='dqn'. "
            f"Found {len(dqn_competitors)}."
        )
    if base_competitor is None:
        raise ValueError(
            "low_marl_3dqn_vs_base complexity mode requires exactly one competitor with agent='base'."
        )
    return dqn_competitors + [base_competitor]


def select_low_marl_teams_competitors(competitors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select 4 DQN competitors (2 teams of 2) + 1 Base for team-based MARL mode."""
    from collections import defaultdict
    teams: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    base_competitor = None
    for comp in competitors:
        agent = str(comp.get("agent", "")).strip().lower()
        if agent == "dqn":
            tid = comp.get("team_id")
            if tid is not None:
                teams[str(tid)].append(deepcopy(comp))
        elif agent == "base" and base_competitor is None:
            base_competitor = deepcopy(comp)

    if len(teams) != 2:
        raise ValueError(
            f"low_marl_teams requires exactly 2 teams with team_id set, "
            f"found {len(teams)}: {list(teams.keys())}"
        )
    for tid, members in teams.items():
        if len(members) < 2:
            raise ValueError(
                f"low_marl_teams: team '{tid}' needs at least 2 DQN agents, "
                f"found {len(members)}."
            )
    if base_competitor is None:
        raise ValueError(
            "low_marl_teams requires exactly one competitor with agent='base'."
        )

    result = []
    for tid in sorted(teams.keys()):
        result.extend(teams[tid][:2])
    result.append(base_competitor)
    return result
