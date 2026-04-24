# Autoresearch Setup - Quick Start (5 minutes)

## Step 1: Install Anthropic SDK

Run in PowerShell (or conda terminal):
```powershell
pip install anthropic
```

Expected output: `Successfully installed anthropic-X.X.X`

## Step 2: Set Your API Key

### Option A: Using .env File (Recommended)
```powershell
# Copy the example file
cp .env.example .env

# Edit .env with your actual key
notepad .env
```

Then in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-YOUR_ACTUAL_KEY_HERE
```

The agent will automatically load it when it runs.

### Option B: PowerShell Environment Variable (Temporary)
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-YOUR_ACTUAL_KEY_HERE"
```

This applies only to your current PowerShell session.

### Option C: PowerShell Permanent (Restarts Required)
```powershell
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-YOUR_ACTUAL_KEY_HERE", "User")
```

Then restart PowerShell for it to take effect.

**Where to get your key**: https://console.anthropic.com/account/keys

## Step 3: Establish Baseline

Before starting autoresearch, run one quick baseline evaluation to confirm the evaluator works:

```powershell
cd C:\Code\F1_StrategySimulator
conda activate f1StrategySim

python scripts/evaluate_candidate.py `
  --skip-training `
  --eval-runs 5 `
  --eval-seeds "101" `
  --out metrics/baseline.json

# Then check the result:
cat metrics/baseline.json | jq '.objective_score'
```

You should see a number like `0.529847` (your baseline win rate).

**Note**: If this fails with "ModuleNotFoundError", make sure the f1StrategySim conda environment is activated.

## Step 4: Start the AI Loop

### Method A: Using PowerShell Launcher (Recommended)
```powershell
.\scripts\start_agent.ps1 -MaxIterations 30
```

The script will:
- ✓ Verify your API key is set
- ✓ Check Python environment
- ✓ Verify anthropic package is installed
- ✓ Ask for confirmation
- ✓ Run the agent loop

### Method B: Direct Python (Advanced)
```powershell
python scripts/autoresearch_agent.py `
  --branch autoresearch/test `
  --max-iterations 30 `
  --eval-runs 50
```

## What Happens Next

The agent will enter a loop (~5 min per iteration):

```
[1] ===== ITERATION 1 / 30 =====
[1] File: src/agents/DQN.py
[1] Reasoning: Increased learning rate from 1e-4 to 2e-4
[1] Applying change...
[1] Committed: a1b2c3d
[1] Running quick eval (50 runs)...
[1] Score: 0.551234
[1] ✓ KEEP (improvement: +0.021387)
[1] Best so far: 0.551234

[2] ===== ITERATION 2 / 30 =====
...
```

**Each iteration**:
1. Claude suggests ONE code change
2. Script applies it to disk
3. Git commits it
4. Evaluator runs (50 runs × 3 seeds ≈ 5 minutes)
5. If score ≥ (baseline - 2%), **keep** it
6. If score < (baseline - 2%), **discard** it (reset git)
7. Results recorded in `results.tsv`

## Monitoring During Loop

In a separate terminal, watch progress:
```powershell
# Every 10 seconds, show last 5 results
while ($true) { Clear-Host; Get-Content results.tsv | Select-Object -Last 5; Start-Sleep -Seconds 10 }
```

## Stop & Resume

- **Stop**: Press `Ctrl+C`
- **Resume**: Run same command again; it will continue from where it left off:
  ```powershell
  python scripts/autoresearch_agent.py --branch autoresearch/test --max-iterations 100
  ```

## After Loop Completes

1. **Check results**:
   ```powershell
   cat results.tsv
   ```

2. **Find best iteration**:
   ```powershell
   (Get-Content results.tsv | Select-Object -Skip 1) -split "`n" | Sort-Object { [float]($_-split "`t")[1] } -Descending | Select-Object -First 5
   ```

3. **Inspect winning code**:
   ```powershell
   git log --oneline -10
   git show <commit>
   ```

4. **Run full validation** (500 runs) on best commit:
   ```powershell
   git checkout <best_commit>
   python scripts/evaluate_candidate.py `
     --skip-training `
     --eval-runs 500 `
     --eval-seeds "101,202,303,404,505" `
     --out metrics/final_eval.json
   ```

## Troubleshooting

### `ERROR: ANTHROPIC_API_KEY environment variable not set`
Run:
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key"
```

### `ModuleNotFoundError: anthropic`
Run:
```powershell
pip install anthropic
```

### `ERROR: Could not parse Claude response as JSON`
This is usually a transient API issue. The agent logs it and retries on next iteration. If persistent:
- Check internet connection
- Verify API key is valid at https://console.anthropic.com/account/keys
- Check account has enough credits

### Evaluator keeps crashing
Run manual test:
```powershell
python scripts/evaluate_candidate.py --skip-training --eval-runs 1 --eval-seeds "101" --out test.json
```

If that crashes, the issue is in DQN.py or simulator.py (not agent code). Check the error and fix syntax/logic.

## Next Steps

See [AUTORESEARCH_README.md](AUTORESEARCH_README.md) for:
- Detailed parameter tuning ideas (learning rate, gamma, epsilon, reward shaping)
- Two-phase evaluation strategy (quick + full)
- Git workflow for branching/merging
- Advanced agent customization

---

**Ready?** Start with Step 1 (install anthropic) and you'll be running in 5 minutes.
