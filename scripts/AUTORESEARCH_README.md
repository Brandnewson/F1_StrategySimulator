# Autoresearch Agent Setup & Usage

## Prerequisites

1. **Anthropic API Key**: Sign up at https://console.anthropic.com/ and get your API key
   
2. **Python Package**:
   ```bash
   pip install anthropic
   ```

3. **Git Setup** (if not already done):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```

## Quick Start

### 1. Set Environment Variable (PowerShell)
```powershell
# Option A: Use .env file (recommended)
cp .env.example .env
# Then edit .env with your API key

# Option B: Set in PowerShell (temporary session only)
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Option C: Set permanently (requires PowerShell restart)
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")
```

### 2. Run Initial Baseline Evaluation
Before starting autoresearch, establish a baseline:
```bash
conda activate f1StrategySim
cd C:\Code\F1_StrategySimulator
python scripts/evaluate_candidate.py --skip-training --eval-runs 50 --eval-seeds "101,202,303" --out metrics/baseline.json
```

Record the `objective_score` from the output—this will be your starting point.

### 3. Initialize Autoresearch Git Branch
```bash
git checkout main
git pull origin main
```

### 4. Start the Autoresearch Loop
```bash
python scripts/autoresearch_agent.py --branch autoresearch/mar10 --max-iterations 50
```

#### Optional Arguments:
- `--max-iterations N`: Run for N iterations (default: 50)
- `--eval-runs N`: Runs per quick-phase eval (default: 50)
- `--eval-seeds "S1,S2,S3"`: Commission-separated seed list (default: "101,202,303")
- `--improvement-threshold T`: Keep if score > baseline - T (default: 0.02, so ±2%)

### Example: Conservative, Long-Running Sweep
```bash
python scripts/autoresearch_agent.py \
  --branch autoresearch/sweep_v2 \
  --max-iterations 100 \
  --eval-runs 50 \
  --improvement-threshold 0.01
```

## What It Does

1. **Iteration Loop**: For each iteration:
   - Reads current DQN.py, simulator.py, config.json
   - Calls Claude Opus (via API) to suggest 1 focused code change
   - Applies the change to disk
   - Commits to git
   - Runs evaluator (50 runs × 3 seeds ≈ 5 minutes)
   - Parses `objective_score`
   - **Keep** if score ≥ baseline - 0.02
   - **Discard** (git reset) if score < baseline - 0.02
   - Updates `results.tsv`

2. **Results Tracking**: All iterations recorded in `results.tsv`:
   ```
   commit                 objective_score  phase  status   description
   a1b2c3d                0.532100         quick  keep     Increased learning rate
   e5f6g7h                0.489000         quick  discard  Reduced gamma
   ```

3. **Git History**: Each iteration is a commit; discarded runs get reset automatically.

## Monitoring Progress

### Watch Results in Real-Time
```bash
# PowerShell
while ($true) { Clear-Host; Get-Content results.tsv | Select-Object -Last 10; Start-Sleep -Seconds 10 }
```

### View Latest Eval Metrics
```bash
cat metrics/candidate_eval.json | jq '.metrics'
```

### Inspect a Specific Git Commit
```bash
git show <commit>
git diff <commit>~1 <commit>
```

## Troubleshooting

### `ERROR: ANTHROPIC_API_KEY environment variable not set`
```powershell
# Option A: Create and edit .env file (recommended)
cp .env.example .env
# Edit .env with your actual key

# Option B: Set in current session
$env:ANTHROPIC_API_KEY = "sk-ant-your-key"

# Then run agent
python scripts/autoresearch_agent.py --branch autoresearch/mar10
```

### `ERROR: Could not parse Claude response as JSON`
This is usually transient. The agent will skip that iteration and try again. If persistent:
- Check `ANTHROPIC_API_KEY` is valid
- Verify internet connection
- Try a smaller `--max-iterations` to test

### `ModuleNotFoundError: anthropic`
```bash
pip install anthropic
```

### Evaluator keeps crashing
1. Run manual test:
   ```bash
   python scripts/evaluate_candidate.py --skip-training --eval-runs 1 --eval-seeds "101" --out test.json
   ```
2. Check `src/agents/DQN.py` and `src/simulator.py` for syntax errors
3. If syntax error, fix and re-run agent (it will auto-recover)

### Want to Stop Early?
Press `Ctrl+C` in the terminal. The last change is already committed. You can:
- Resume later: `python scripts/autoresearch_agent.py --branch autoresearch/mar10 --max-iterations 100`
- Cherry-pick best: `git log --oneline | head -20`
- Revert to a good commit: `git reset --hard <good_commit>`

## Next Steps (After Loop Completes)

1. **Review Results**:
   ```bash
   tail -20 results.tsv
   ```

2. **Find Best Experiment**:
   ```bash
   cat results.tsv | sort -t$'\t' -k2 -rn | head -5
   ```

3. **Inspect Winning Changes**:
   ```bash
   git show <best_commit>
   ```

4. **Run Full Evaluation** (500 runs) on best:
   ```bash
   python scripts/evaluate_candidate.py --skip-training --eval-runs 500 --eval-seeds "101,202,303,404,505" --out metrics/final_eval.json
   ```

5. **Merge to Main** (if satisfied):
   ```bash
   git checkout main
   git merge autoresearch/mar10
   git push origin main
   ```

---

**Questions?** Review `tools/autoresearch/program.md` for the full research protocol and hyperparameter tuning ideas.
