<#
.SYNOPSIS
Run autoresearch agent to autonomously improve F1 Strategy Simulator.

.PARAMETER MaxIterations
Maximum number of experiments to try (default: 50).

.PARAMETER TimePerRun
Time budget per training run in minutes (default: ~5 min equivalent via run count).

.PARAMETER OutputDir
Directory to store experiment logs (default: logs/autoresearch/).
#>

param(
    [int]$MaxIterations = 50,
    [int]$TimePerRun = 5,
    [string]$OutputDir = "logs/autoresearch"
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$AUTORESEARCH_DIR = Join-Path $ROOT "tools" "autoresearch"
$EVALUATOR_SCRIPT = Join-Path $ROOT "scripts" "evaluate_candidate.py"

# Ensure autoresearch submodule is initialized
Write-Host "[run_autoresearch] Initializing autoresearch submodule..."
Push-Location $ROOT
git submodule update --init --recursive tools/autoresearch
Pop-Location

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "[run_autoresearch] Starting autoresearch with:"
Write-Host "  Max iterations: $MaxIterations"
Write-Host "  Evaluator: $EVALUATOR_SCRIPT"
Write-Host "  Output dir: $OutputDir"
Write-Host ""

# Change to autoresearch directory and run
Push-Location $AUTORESEARCH_DIR

# Run the autoresearch CLI
# (You will invoke this via the Claude/LLM prompt; this is just the setup)
Write-Host "[run_autoresearch] Ready. Point your AI agent at program.md in $(Resolve-Path .)"
Write-Host "[run_autoresearch] Evaluator command template:"
Write-Host "  python $EVALUATOR_SCRIPT --eval-runs 50 --eval-seeds 101,202,303 --out metrics/candidate.json"

Pop-Location