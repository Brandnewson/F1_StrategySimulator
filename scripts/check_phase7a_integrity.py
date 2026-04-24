"""Phase 7A integrity check — verify experiment metrics are complete and consistent."""
import json
import glob
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def main():
    files = sorted(glob.glob("metrics/phase7a/3dqn_*.json"))
    print(f"=== Phase 7A Integrity Check ({len(files)} trials) ===\n")

    issues = []
    all_data = []

    for f in files:
        d = json.load(open(f))
        fname = Path(f).name
        all_data.append((fname, d))

        # 1. Phase label
        if d.get("phase") != "phase7_3dqn_vs_base":
            issues.append(f"{fname}: wrong phase label: {d.get('phase')}")

        # 2. Complexity profile
        if d.get("complexity_profile") != "low_marl_3dqn_vs_base":
            issues.append(f"{fname}: wrong complexity: {d.get('complexity_profile')}")

        # 3. Algorithm
        if d.get("algorithm") != "rainbow_lite":
            issues.append(f"{fname}: wrong algo: {d.get('algorithm')}")

        # 4. Alpha matches filename
        expected_alpha = 0.75 if "_a075_" in fname else 0.0
        actual_alpha = d.get("reward_sharing_alpha", -1)
        if abs(actual_alpha - expected_alpha) > 1e-6:
            issues.append(f"{fname}: alpha={actual_alpha}, expected {expected_alpha}")

        # 5. Three DQN agents present
        dqn_names = d.get("all_dqn_names", [])
        if len(dqn_names) < 3:
            issues.append(f"{fname}: expected 3 DQN agents, got {len(dqn_names)}")

        # 6. agent3_name populated
        if not d.get("agent3_name"):
            issues.append(f"{fname}: agent3_name missing")

        # 7. Training not skipped, correct budget
        train = d.get("phases", {}).get("training", {})
        if train.get("skipped"):
            issues.append(f"{fname}: training was skipped!")
        if train.get("runs") != 500:
            issues.append(f"{fname}: train runs={train.get('runs')}, expected 500")

        # 8. Eval runs and balanced positions
        ev = d.get("phases", {}).get("evaluation", {})
        if ev.get("runs_per_seed") != 150:
            issues.append(f"{fname}: eval runs={ev.get('runs_per_seed')}, expected 150")
        if not ev.get("balanced_positions"):
            issues.append(f"{fname}: balanced positions not set")

        # 9. DNF rates
        metrics = d.get("metrics", {})
        for agent_key in ["agent1", "agent2"]:
            agent_m = metrics.get(agent_key, {})
            dnf = agent_m.get("dnf_rate", 0)
            if dnf > 0:
                issues.append(f"{fname}: {agent_key} DNF rate={dnf}")

        # 10. vs_base_metrics has all 3 agents
        vb = d.get("vs_base_metrics", {})
        for i in range(1, 4):
            key = f"dqn_a{i}_beats_base_rate"
            if key not in vb:
                issues.append(f"{fname}: missing {key} in vs_base_metrics")

        n_races = vb.get("n_races", 0)
        if n_races != 150:
            issues.append(f"{fname}: vs_base n_races={n_races}, expected 150")

    # --- Completeness ---
    a000_files = [f for f, _ in all_data if "_a000_" in f]
    a075_files = [f for f, _ in all_data if "_a075_" in f]
    print("Trial Completeness:")
    print(f"  Expected: 18 (2 alphas x 3 seeds x 3 stoch)")
    print(f"  Found:    {len(files)} (alpha=0.00: {len(a000_files)}, alpha=0.75: {len(a075_files)})")
    if len(a075_files) == 0:
        print("  WARNING:  alpha=0.75 trials not yet run")
    print()

    # --- Seed/stoch coverage per alpha ---
    for alpha_label, subset in [("alpha=0.00", a000_files), ("alpha=0.75", a075_files)]:
        if not subset:
            continue
        seeds_seen = set()
        stochs_seen = set()
        for fname in subset:
            d = dict(all_data)[fname] if False else [dd for ff, dd in all_data if ff == fname][0]
            seeds_seen.add(d.get("phases", {}).get("training", {}).get("seed"))
            stochs_seen.add(d.get("stochasticity_level"))
        print(f"{alpha_label} coverage:")
        print(f"  Seeds: {sorted(seeds_seen)} (expect [101, 202, 303])")
        print(f'  Stoch: {sorted(stochs_seen)} (expect ["s0", "s1", "s2"])')
        print()

    # --- Per-trial metrics table ---
    print("Per-trial metrics:")
    hdr = f"{'File':<30} {'WR_a1':>7} {'a1_base':>8} {'a2_base':>8} {'a3_base':>8} {'joint':>7} {'z_diff':>7} {'r_diff':>7}"
    print(hdr)
    print("-" * len(hdr))
    for fname, d in all_data:
        m = d.get("metrics", {})
        vb = d.get("vs_base_metrics", {})
        sd = m.get("strategy_differentiation", {})
        wr = m.get("win_rate_a1_vs_a2", {}).get("mean", 0)
        a1b = vb.get("dqn_a1_beats_base_rate", {}).get("mean", 0)
        a2b = vb.get("dqn_a2_beats_base_rate", {}).get("mean", 0)
        a3b = vb.get("dqn_a3_beats_base_rate", {}).get("mean", 0)
        jb = vb.get("joint_dqn_beat_base_rate", {}).get("mean", 0)
        zd = sd.get("zone_differentiation_index", 0)
        rd = sd.get("risk_differentiation_index", 0)
        print(f"{fname:<30} {wr:>7.3f} {a1b:>8.3f} {a2b:>8.3f} {a3b:>8.3f} {jb:>7.3f} {zd:>7.3f} {rd:>7.3f}")
    print()

    # --- Aggregates by stochasticity ---
    for alpha_label, prefix in [("alpha=0.00", "_a000_"), ("alpha=0.75", "_a075_")]:
        subset = [(f, d) for f, d in all_data if prefix in f]
        if not subset:
            continue
        print(f"Aggregates by stochasticity ({alpha_label}):")
        by_stoch = defaultdict(list)
        for fname, d in subset:
            s = d.get("stochasticity_level")
            vb = d.get("vs_base_metrics", {})
            by_stoch[s].append(vb.get("joint_dqn_beat_base_rate", {}).get("mean", 0))
        for s in sorted(by_stoch.keys()):
            vals = by_stoch[s]
            seeds_str = ", ".join(f"{v:.3f}" for v in vals)
            print(f"  {s}: joint_beat_base mean={np.mean(vals):.3f} seeds=[{seeds_str}]")
        print()

    # --- Verdict ---
    if issues:
        print(f"ISSUES FOUND ({len(issues)}):")
        for i in issues:
            print(f"  ! {i}")
        sys.exit(1)
    else:
        print("No integrity issues found in available trials.")


if __name__ == "__main__":
    main()
