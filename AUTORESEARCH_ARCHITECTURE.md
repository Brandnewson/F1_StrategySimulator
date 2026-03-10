# Autoresearch Architecture

## Overview

The autoresearch system automatically improves your F1 Strategy Simulator's DQN agent by:

1. **Claude AI** suggests hyperparameter tweaks (every 5 min)
2. **Agent script** applies changes, runs evaluations, makes keep/discard decisions
3. **Git** tracks all changes for reproducibility and rollback
4. **Results TSV** logs wins/losses for analysis

## Components

```
┌─────────────────────────────────────────────────────────────┐
│                  Claude Opus (via API)                      │
│  Suggests code changes based on program.md + current code   │
└────────────────────┬────────────────────────────────────────┘
                     │ JSON: {file, old_text, new_text, reasoning}
                     │
     ┌───────────────▼───────────────┐
     │ scripts/autoresearch_agent.py  │  Main agent loop
     │ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~  │  - Reads Claude suggestions
     │ - Apply changes to files       │  - Applies code edits
     │ - Git commit                   │  - Runs evaluator
     │ - Run evaluator               │  - Decides keep/discard
     │ - Parse metrics               │  - Logs results
     │ - Update results.tsv          │
     └───────────────┬───────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ scripts/evaluate_candidate.py   │  Fitness evaluator
     │ ~~~~~~~~~~~~~~~~~~~~~~~~~      │  - Runs 50 train+eval sim
     │ - Run train phase (1 seed)     │  - Computes win_rate_vs_baseline
     │ - Run eval phase (3 seeds)     │  - Writes metrics JSON
     │ - Compute win_rate CI95        │  - Returns objective_score
     └───────────────┬────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │  src/agents/DQN.py              │  Neural network agent
     │  src/simulator.py               │  (gets modified each iter)
     │  config.json                    │
     └────────────────────────────────┘
                     │
             ┌───────▼────────┐
             │  Git History   │
             │  + results.tsv │
             └────────────────┘
```

## Files & Folders

| File | Purpose |
|------|---------|
| `scripts/autoresearch_agent.py` | **Main loop** - Claude API client, file ops, git control, evaluator runner |
| `scripts/start_agent.ps1` | **Launcher** - Validates env, checks dependencies, runs agent with confirmation |
| `scripts/AUTORESEARCH_README.md` | **Detailed guide** - Params, examples, troubleshooting, next steps |
| `AUTORESEARCH_QUICKSTART.md` | **Quick start** - 4-step setup (install, API key, baseline, run) |
| `requirements-autoresearch.txt` | **Dependencies** - Lists `anthropic` package version |
| `tools/autoresearch/program.md` | **AI Context** - Tells Claude what to optimize, how to modify code, decision rules |
| `results.tsv` | **Log** - Tab-separated results: commit, score, phase, status, description |
| `.gitignore` | **VCS** - Excludes metrics/, logs/, results.tsv from commits |
| `scripts/evaluate_candidate.py` | **Pre-existing** - Runs training + evals, returns objective_score JSON |

## Data Flow: One Iteration

```
┌─ Iteration Start ─────────────────────────────────────┐
│                                                       │
│  1. agent reads DQN.py, simulator.py, config.json   │
│                                                       │
│  2. agent calls Claude API with program.md context   │
│     ├→ "Here's the code, improve win_rate"           │
│     └→ Claude: "Change learning_rate from 1e-4 to 2e-4"
│                                                       │
│  3. agent applies change to disk                      │
│     ├→ DQN.py: self.learning_rate = 2e-4             │
│     └→ Write file                                     │
│                                                       │
│  4. agent git commits ("Attempt 5: ...")             │
│                                                       │
│  5. agent runs evaluator subprocess                   │
│     ├→ evaluate_candidate.py runs 50 train runs      │
│     └→ Outputs metrics/candidate_eval.json           │
│                                                       │
│  6. agent parses objective_score from JSON           │
│     └→ score = 0.551234 (win rate)                   │
│                                                       │
│  7. agent decides: keep if score > baseline - 0.02   │
│     ├→ If KEEP: commit stays, best = 0.551234       │
│     └→ If DISCARD: git reset --hard HEAD~1           │
│                                                       │
│  8. agent updates results.tsv                         │
│     └→ commit | score  | phase | status | desc       │
│        a1b2c3d | 0.551234 | quick | keep | ...      │
│                                                       │
└─ Repeat for next iteration (or exit if max reached) ─┘
```

## Decision Logic

### Keep Decision (Phase: Quick)
```python
improvement = score - baseline
if improvement > -0.02:  # Allow 2% regression
    KEEP()
else:
    DISCARD(git reset --hard)
```

**Why allow regression?** 
- Training stochasticity: small swings (±2%) are noise
- Exploration: some "worse" ideas enable future breakthroughs
- Prevents premature convergence

### Future: Full Phase (Not Yet Implemented)
```
Quick Phase: 50 runs × 3 seeds → 5 min
├─ If score qualifies (e.g., > baseline - 0.05)
└─ Promote to FULL phase:
      500 runs × 5 seeds → 50 min
      ├─ If full score improves, KEEP with confidence
      └─ Else DISCARD despite quick phase success
```

## How Claude Optimizes

Claude reads `tools/autoresearch/program.md`, which tells it:

**What to optimize**:
- Primary: `win_rate_vs_baseline` (higher = better)
- Secondary: race quality, tactical metrics

**Where to edit**:
- `src/agents/DQN.py` → hyperparams (learning_rate, gamma, epsilon_decay, hidden_size, target_update_freq)
- `src/simulator.py` → reward shaping (overtake_bonus, position_gain, dnf_penalty)
- `config.json` → simulation params (num_opponents, track, race_length)

**How to decide**:
- Quick phase: ~5 min per candidate
- Keep if: score > best_kept - 0.02
- Discard if: score < best_kept - 0.02

**Claude's typical loop**:
1. Iteration 1: "Try higher learning_rate"
2. Iteration 2: "Try lower epsilon_decay"
3. Iteration 3: "Try bigger hidden_size"
4. Iteration 4: "Revert hidden_size, adjust reward shaping"
5. Etc.

Claude doesn't see evaluator results directly; it infers from the agent's **keep/discard decisions** in results.tsv.

## Git Workflow

Each iteration creates a commit:
```
main
├─ autoresearch/mar10
│  ├─ a1b2c3d: Attempt 1 (KEEP)
│  ├─ e5f6g7h: Attempt 2 (DISCARD) - reset removes this
│  ├─ i9j0k1l: Attempt 3 (KEEP)
│  └─ m3n4o5p: Attempt 4 (DISCARD) - reset removes this
```

### Resuming a Run

If you stop the agent mid-loop:
```bash
# Check where you left off
cat results.tsv | tail -5

# Add more iterations (starts from HEAD)
python scripts/autoresearch_agent.py \
  --branch autoresearch/mar10 \
  --max-iterations 50  # Will run 50 MORE, totaling ~100
```

### Merging Back to Main

Once satisfied with results:
```bash
git checkout main
git merge autoresearch/mar10
git log --oneline -n 20
```

## Customization

### Change Improvement Threshold

Default allows 2% regression. To be stricter:
```powershell
python scripts/autoresearch_agent.py `
  --improvement-threshold 0.01  # Only 1% regression
```

### Longer Evals Per Iteration

Default: 50 runs. For more confidence:
```powershell
python scripts/autoresearch_agent.py `
  --eval-runs 100  # ~10 min per iteration
```

### More Seeds for Stability

Default: 3 seeds (101, 202, 303). More variance estimation:
```powershell
python scripts/autoresearch_agent.py `
  --eval-seeds "101,202,303,404,505,606"
```

## Cost Estimation

| Component | Cost |
|-----------|------|
| **Claude API** | ~$0.03 per iteration (via Claude Opus) |
| **Anthropic Quota** | ~$0.03 × 50 iterations = $1.50 for test run |
| **Full experiment** | ~$0.03 × 500 = $15 (assuming 500 iterations) |

*Use case: $100-500 Anthropic credit will run hundreds of iterations.*

## Limitations & Future Work

**Current**:
- ✓ Quick phase only (5 min per eval)
- ✓ Single hyperparameter change per iteration
- ✓ Binary keep/discard logic
- ✓ Git-based rollback

**TODO** (not yet implemented):
- ⏳ Full phase (500-run validation for top candidates)
- ⏳ Multi-change suggestions from Claude
- ⏳ Confidence interval thresholds
- ⏳ Automatic hyperparameter bounds learning
- ⏳ Ray Tune integration for parallel trials

## Support

**Stuck?**
1. Read [AUTORESEARCH_QUICKSTART.md](AUTORESEARCH_QUICKSTART.md) (4-step setup)
2. Check [scripts/AUTORESEARCH_README.md](scripts/AUTORESEARCH_README.md) (examples, troubleshooting)
3. Review `tools/autoresearch/program.md` (Claude context, decision rules)
4. Inspect `results.tsv` (log of all iterations)

**Debug a failing iteration**:
```bash
git log -1 --stat          # What changed?
git diff HEAD~1            # Show the change
python scripts/evaluate_candidate.py --skip-training --eval-runs 1 --eval-seeds "101"
```

---

**See also**: [program.md](tools/autoresearch/program.md) — the AI instructions that Claude reads every iteration.
