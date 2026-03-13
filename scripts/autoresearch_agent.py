#!/usr/bin/env python3
"""
Autonomous research agent for F1 Strategy Simulator.

Uses Claude API to suggest code changes, runs evaluations, and iteratively improves
the DQN agent's win_rate_vs_baseline metric.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python scripts/autoresearch_agent.py --branch autoresearch/mar10 --max-iterations 50

Requirements:
    - ANTHROPIC_API_KEY env var set
    - git configured with user.name and user.email
    - f1StrategySim conda environment activated
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not found. Install with: pip install anthropic")
    sys.exit(1)

def load_env_file(env_path: Path = None) -> None:
    """Load .env file into environment variables."""
    if env_path is None:
        env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SCRIPT = ROOT / "scripts" / "evaluate_candidate.py"
PROGRAM_MD = ROOT / "tools" / "autoresearch" / "program.md"
RESULTS_TSV = ROOT / "results.tsv"
METRICS_DIR = ROOT / "metrics"
REPORT_MD = ROOT / "AUTORESEARCH_REPORT.md"


def shell(cmd: str, check: bool = True) -> str:
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            print(f"Command failed: {cmd}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(f"Command failed with exit code {result.returncode}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {cmd}")
        raise


def read_file(path: Path) -> str:
    """Read file content."""
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: Path, content: str) -> None:
    """Write file content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def validate_candidate_edit(file_path: Path) -> Tuple[bool, str]:
    """Validate edited file syntax before commit."""
    suffix = file_path.suffix.lower()
    if suffix == ".py":
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, (result.stderr or result.stdout or "Python syntax validation failed").strip()
        return True, ""

    if suffix == ".json":
        try:
            json.loads(read_file(file_path))
            return True, ""
        except Exception as e:
            return False, f"JSON validation failed: {e}"

    return True, ""


def read_results_tsv() -> Dict[str, dict]:
    """Read results.tsv into dict."""
    results = {}
    if not RESULTS_TSV.exists():
        return results
    with open(RESULTS_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader and reader.fieldnames:
            for row in reader:
                if row.get("commit"):
                    results[row["commit"]] = row
    return results


def write_results_tsv(results: Dict[str, dict]) -> None:
    """Write results.tsv from dict."""
    fieldnames = ["commit", "objective_score", "phase", "status", "description"]
    with open(RESULTS_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for commit, row in sorted(results.items()):
            writer.writerow(row)


def get_current_commit() -> str:
    """Get current git commit hash (short)."""
    return shell("git rev-parse --short HEAD")


def git_commit(message: str) -> str:
    """Create a git commit."""
    shell(f'git add -A && git commit -m "{message}"')
    return get_current_commit()


def git_reset_hard() -> None:
    """Reset to previous commit."""
    shell("git reset --hard HEAD~1")


def run_evaluator_quick(
    eval_runs: int = 50,
    eval_seeds: str = "101,202,303",
    total_laps: Optional[int] = None,
) -> Optional[float]:
    """Run quick evaluation phase. Returns objective_score or None if crashed."""
    METRICS_DIR.mkdir(exist_ok=True)
    out_file = METRICS_DIR / "candidate_eval.json"
    total_laps_flag = f" --total-laps {int(total_laps)}" if total_laps is not None else ""
    cmd = (
        f'C:/Users/brans/miniconda3/envs/f1StrategySim/python.exe {EVALUATOR_SCRIPT} '
        f'--skip-training --eval-runs {eval_runs} --eval-seeds "{eval_seeds}" '
        f'--out {out_file}{total_laps_flag}'
    )
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Evaluator failed: {result.stderr}")
            return None
        if not out_file.exists():
            print(f"Metrics file not created: {out_file}")
            return None
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        score = data.get("objective_score")
        return float(score) if score is not None else None
    except Exception as e:
        print(f"Evaluator exception: {e}")
        return None


def run_evaluator_train(
    train_runs: int = 20,
    train_seed: int = 1337,
    total_laps: Optional[int] = None,
    checkpoint_tag: str = "",
) -> Optional[float]:
    """Run lightweight training + evaluation phase. Returns objective_score or None if crashed."""
    METRICS_DIR.mkdir(exist_ok=True)
    out_file = METRICS_DIR / "candidate_train_eval.json"
    total_laps_flag = f" --total-laps {int(total_laps)}" if total_laps is not None else ""
    checkpoint_flag = f" --checkpoint-tag {checkpoint_tag}" if checkpoint_tag else ""
    cmd = (
        f'C:/Users/brans/miniconda3/envs/f1StrategySim/python.exe {EVALUATOR_SCRIPT} '
        f'--train-runs {train_runs} --train-seed {train_seed} --eval-runs 1 --eval-seeds "{train_seed}" '
        f'--out {out_file}{total_laps_flag}{checkpoint_flag}'
    )
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Training evaluator failed: {result.stderr}")
            return None
        if not out_file.exists():
            print(f"Training metrics file not created: {out_file}")
            return None
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        score = data.get("objective_score")
        return float(score) if score is not None else None
    except Exception as e:
        print(f"Training evaluator exception: {e}")
        return None


def write_autoresearch_report(
    branch: str,
    max_iterations: int,
    final_best: float,
    iteration_rows: List[Dict[str, str]],
) -> None:
    created_at = datetime.now().isoformat()
    lines: List[str] = []
    lines.append("# AutoResearch Report")
    lines.append("")
    lines.append(f"- Created at: {created_at}")
    lines.append(f"- Branch: {branch}")
    lines.append(f"- Planned iterations: {max_iterations}")
    lines.append(f"- Completed iterations: {len(iteration_rows)}")
    lines.append(f"- Final best objective_score: {final_best:.6f}")
    lines.append("")
    lines.append("## Iteration Summary")
    lines.append("")
    lines.append("| Iteration | Commit | File | Status | Objective Score | Reasoning | Checkpoint Tag |")
    lines.append("|---:|---|---|---|---:|---|---|")
    for row in iteration_rows:
        iteration = row.get("iteration", "")
        commit = row.get("commit", "")
        file_name = row.get("file", "")
        status = row.get("status", "")
        score = row.get("score", "")
        reasoning = row.get("reasoning", "").replace("|", "/")
        checkpoint_tag = row.get("checkpoint_tag", "")
        lines.append(
            f"| {iteration} | {commit} | {file_name} | {status} | {score} | {reasoning} | {checkpoint_tag} |"
        )
    lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def initialize_repo(branch: str) -> None:
    """Initialize git branch and results.tsv."""
    print(f"[init] Creating branch {branch}...")
    shell(f"git checkout -b {branch}", check=False)  # ignore if already exists
    
    print(f"[init] Initializing results.tsv...")
    if not RESULTS_TSV.exists():
        write_results_tsv({})
    
    print(f"[init] Verifying evaluator works...")
    score = run_evaluator_quick(eval_runs=1, eval_seeds="101", total_laps=1)
    if score is None:
        print("ERROR: Evaluator failed on test run. Fix and retry.")
        sys.exit(1)
    print(f"[init] Baseline test score: {score:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous research agent for F1 Strategy Simulator."
    )
    parser.add_argument(
        "--branch",
        default="autoresearch/mar10",
        help="Git branch name (default: autoresearch/mar10)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum iterations to run (default: 50)",
    )
    parser.add_argument(
        "--eval-runs",
        type=int,
        default=50,
        help="Eval runs per iteration (default: 50)",
    )
    parser.add_argument(
        "--eval-seeds",
        default="101,202,303",
        help="Comma-separated eval seeds (default: 101,202,303)",
    )
    parser.add_argument(
        "--train-runs",
        type=int,
        default=20,
        help="Training runs per iteration (default: 20)",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=1337,
        help="Training seed per iteration (default: 1337)",
    )
    parser.add_argument(
        "--smoke-laps",
        type=int,
        default=1,
        help="Laps for smoke test evaluation (default: 1)",
    )
    parser.add_argument(
        "--eval-laps",
        type=int,
        default=10,
        help="Laps for quick eval phase (default: 10)",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.02,
        help="Minimum improvement to promote to full phase (default: 0.02)",
    )
    args = parser.parse_args()

    # Load .env file if it exists
    load_env_file()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it one of these ways:")
        print('  1. Create .env file: cp .env.example .env && edit .env')
        print('  2. PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."')
        print('  3. Permanent: [Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")')
        sys.exit(1)

    # Read program.md for context
    if not PROGRAM_MD.exists():
        print(f"ERROR: program.md not found at {PROGRAM_MD}")
        sys.exit(1)
    program_context = read_file(PROGRAM_MD)

    # Initialize
    initialize_repo(args.branch)

    # Read baseline
    results = read_results_tsv()
    best_baseline = max(
        (float(r.get("objective_score", 0)) for r in results.values()),
        default=0.0,
    )

    client = anthropic.Anthropic(api_key=api_key)

    print(f"\n{'='*60}")
    print(f"Starting autoresearch loop")
    print(f"Branch: {args.branch}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Best baseline: {best_baseline:.6f}")
    print(f"{'='*60}\n")

    iteration_rows: List[Dict[str, str]] = []

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n[{iteration}] ===== ITERATION {iteration} / {args.max_iterations} =====")

        # Read current code
        dqn_code = read_file(ROOT / "src" / "agents" / "DQN.py")
        simulator_code = read_file(ROOT / "src" / "simulator.py")
        config = read_file(ROOT / "config.json")

        # Ask Claude what to try
        prompt = f"""
{program_context}

=== CURRENT STATE ===
Current best objective_score: {best_baseline:.6f}
Iteration: {iteration} / {args.max_iterations}

=== CURRENT CODE SNIPPETS ===

**src/agents/DQN.py** (first 100 lines):
{dqn_code[:3000]}

**src/simulator.py** (reward shaping region, ~50 lines):
{simulator_code[2000:4000] if len(simulator_code) > 2000 else simulator_code}

**config.json**:
{config}

=== YOUR TASK ===
Suggest ONE specific, focused code change to improve win_rate_vs_baseline.

RESPOND with ONLY a JSON object (no markdown, no extra text):
{{
  "file": "src/agents/DQN.py",
  "old_text": "exact string to find (must match exactly)",
  "new_text": "replacement text",
  "reasoning": "brief 1-2 sentence explanation"
}}

The change should be small and testable. Focus on one hyperparameter or simple logic tweak.
"""

        print(f"[{iteration}] Asking Claude for suggestion...")
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        resp_text = response.content[0].text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', resp_text, re.DOTALL)
        if not json_match:
            print(f"[{iteration}] ERROR: Could not parse Claude response as JSON")
            print(f"Response was: {resp_text[:200]}")
            continue

        try:
            suggestion = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"[{iteration}] ERROR: JSON decode failed: {e}")
            continue

        file_path = ROOT / suggestion["file"]
        old_text = suggestion["old_text"]
        new_text = suggestion["new_text"]
        reasoning = suggestion["reasoning"]

        print(f"[{iteration}] File: {suggestion['file']}")
        print(f"[{iteration}] Reasoning: {reasoning}")
        print(f"[{iteration}] Applying change...")

        # Apply change
        try:
            content = read_file(file_path)
            if old_text not in content:
                print(f"[{iteration}] ERROR: old_text not found in file")
                print(f"Looking for: {old_text[:100]}")
                continue
            new_content = content.replace(old_text, new_text, 1)
            write_file(file_path, new_content)

            is_valid, validation_error = validate_candidate_edit(file_path)
            if not is_valid:
                print(f"[{iteration}] ERROR: Syntax gate failed for {suggestion['file']}")
                print(validation_error[:1000])
                write_file(file_path, content)
                continue
        except Exception as e:
            print(f"[{iteration}] ERROR: Could not apply change: {e}")
            continue

        # Commit change
        try:
            commit_msg = f"Attempt {iteration}: {reasoning[:60]}"
            git_commit(commit_msg)
            commit = get_current_commit()
            print(f"[{iteration}] Committed: {commit}")
        except Exception as e:
            print(f"[{iteration}] ERROR: Could not commit: {e}")
            continue

        checkpoint_tag = f"iter{iteration}_{commit}"

        # Run lightweight training
        print(f"[{iteration}] Running lightweight training ({args.train_runs} runs)...")
        train_score = run_evaluator_train(
            train_runs=args.train_runs,
            train_seed=args.train_seed,
            total_laps=args.eval_laps,
            checkpoint_tag=checkpoint_tag,
        )
        if train_score is None:
            print(f"[{iteration}] TRAINING FAILED: stopping immediately")
            results[commit] = {
                "commit": commit,
                "objective_score": "0.000000",
                "phase": "training",
                "status": "crash",
                "description": reasoning[:50],
            }
            write_results_tsv(results)
            iteration_rows.append(
                {
                    "iteration": str(iteration),
                    "commit": commit,
                    "file": suggestion["file"],
                    "status": "training_crash",
                    "score": "0.000000",
                    "reasoning": reasoning[:120],
                    "checkpoint_tag": checkpoint_tag,
                }
            )
            write_autoresearch_report(args.branch, args.max_iterations, best_baseline, iteration_rows)
            sys.exit(1)

        # Run smoke test (1 run x configurable laps)
        print(f"[{iteration}] Running smoke eval (1 run, {args.smoke_laps} laps)...")
        smoke_score = run_evaluator_quick(eval_runs=1, eval_seeds="101", total_laps=args.smoke_laps)
        if smoke_score is None:
            print(f"[{iteration}] SMOKE FAILED: stopping immediately")
            results[commit] = {
                "commit": commit,
                "objective_score": "0.000000",
                "phase": "smoke",
                "status": "crash",
                "description": reasoning[:50],
            }
            write_results_tsv(results)
            iteration_rows.append(
                {
                    "iteration": str(iteration),
                    "commit": commit,
                    "file": suggestion["file"],
                    "status": "smoke_crash",
                    "score": "0.000000",
                    "reasoning": reasoning[:120],
                    "checkpoint_tag": checkpoint_tag,
                }
            )
            write_autoresearch_report(args.branch, args.max_iterations, best_baseline, iteration_rows)
            sys.exit(1)

        # Run evaluation
        print(f"[{iteration}] Running quick eval ({args.eval_runs} runs, {args.eval_laps} laps)...")
        score = run_evaluator_quick(
            eval_runs=args.eval_runs,
            eval_seeds=args.eval_seeds,
            total_laps=args.eval_laps,
        )

        if score is None:
            print(f"[{iteration}] EVAL FAILED: stopping immediately")
            results[commit] = {
                "commit": commit,
                "objective_score": "0.000000",
                "phase": "quick",
                "status": "crash",
                "description": reasoning[:50],
            }
            write_results_tsv(results)
            iteration_rows.append(
                {
                    "iteration": str(iteration),
                    "commit": commit,
                    "file": suggestion["file"],
                    "status": "eval_crash",
                    "score": "0.000000",
                    "reasoning": reasoning[:120],
                    "checkpoint_tag": checkpoint_tag,
                }
            )
            write_autoresearch_report(args.branch, args.max_iterations, best_baseline, iteration_rows)
            sys.exit(1)

        print(f"[{iteration}] Score: {score:.6f}")

        # Decision logic
        improvement = score - best_baseline
        keep = improvement > -args.improvement_threshold

        if keep:
            status = "keep"
            print(f"[{iteration}] ✓ KEEP (improvement: {improvement:+.6f})")
            best_baseline = max(best_baseline, score)
        else:
            status = "discard"
            print(f"[{iteration}] ✗ DISCARD (decline: {improvement:+.6f})")
            git_reset_hard()

        results[commit] = {
            "commit": commit,
            "objective_score": f"{score:.6f}",
            "phase": "quick",
            "status": status,
            "description": reasoning[:50],
        }
        write_results_tsv(results)
        iteration_rows.append(
            {
                "iteration": str(iteration),
                "commit": commit,
                "file": suggestion["file"],
                "status": status,
                "score": f"{score:.6f}",
                "reasoning": reasoning[:120],
                "checkpoint_tag": checkpoint_tag,
            }
        )

        print(f"[{iteration}] Best so far: {best_baseline:.6f}")

    write_autoresearch_report(args.branch, args.max_iterations, best_baseline, iteration_rows)
    print(f"\n{'='*60}")
    print(f"Autoresearch complete. Results saved to {RESULTS_TSV}")
    print(f"Report written to {REPORT_MD}")
    print(f"Final best: {best_baseline:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
