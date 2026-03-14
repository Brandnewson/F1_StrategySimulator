#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Quick launcher for F1 Strategy Simulator autoresearch agent.

.DESCRIPTION
    Validates setup, sets environment variables, and starts the agent loop.
    Requires: ANTHROPIC_API_KEY environment variable set.

.PARAMETER MaxIterations
    Number of research iterations to run. Default: 50.

.PARAMETER Branch
    Git branch name. Default: autoresearch/MMDD (today's date).

.PARAMETER EvalRuns
    Number of eval runs per iteration. Default: 50.

.EXAMPLE
    .\scripts\start_agent.ps1 -MaxIterations 100
    .\scripts\start_agent.ps1 -Branch autoresearch/aggressive -EvalRuns 50
#>

param(
    [int]$MaxIterations = 50,
    [string]$Branch = "",
    [int]$EvalRuns = 20,
    [int]$TrainRuns = 150,
    [int]$TrainSeed = 1337,
    [int]$SmokeLaps = 1,
    [int]$EvalLaps = 10
)

# Defaults
if (-not $Branch) {
    $Branch = "autoresearch/$(Get-Date -Format 'MMdd')"
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  F1 Strategy Simulator - Autoresearch Agent Launcher" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# Check API key
if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "[ERROR] ANTHROPIC_API_KEY not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "Set it with one of these commands:"
    Write-Host '  [Temporary] $env:ANTHROPIC_API_KEY = "sk-ant-..."'
    Write-Host '  [Permanent] [Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")'
    Write-Host ""
    exit 1
}

Write-Host "[OK] ANTHROPIC_API_KEY found (${env:ANTHROPIC_API_KEY.Substring(0, 10)}...)" -ForegroundColor Green

# Check Python
$pythonPath = "C:/Users/brans/miniconda3/envs/f1StrategySim/python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "[ERROR] Python not found at $pythonPath" -ForegroundColor Red
    Write-Host "Make sure conda environment is created: conda create -n f1StrategySim python=3.11"
    exit 1
}

Write-Host "[OK] Python found: $pythonPath" -ForegroundColor Green

# Check agent script
$agentScript = "scripts/autoresearch_agent.py"
if (-not (Test-Path $agentScript)) {
    Write-Host "[ERROR] Agent script not found: $agentScript" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Agent script found: $agentScript" -ForegroundColor Green

# Check evaluator
$evaluator = "scripts/evaluate_candidate.py"
if (-not (Test-Path $evaluator)) {
    Write-Host "[ERROR] Evaluator not found: $evaluator" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Evaluator found: $evaluator" -ForegroundColor Green

# Check anthropic package
Write-Host "Checking anthropic package..." -ForegroundColor Yellow
$pkgCheck = & $pythonPath -c "import anthropic; print('OK')" 2>&1
if ($pkgCheck -ne "OK") {
    Write-Host "[ERROR] anthropic package not installed" -ForegroundColor Red
    Write-Host "Install with: pip install anthropic"
    exit 1
}

Write-Host "[OK] anthropic package installed" -ForegroundColor Green
Write-Host ""

# Show settings
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Branch:          $Branch"
Write-Host "  Max Iterations:  $MaxIterations"
Write-Host "  Eval Runs:       $EvalRuns per iteration (~5 min each)"
Write-Host "  Train Runs:      $TrainRuns per iteration"
Write-Host "  Train Seed:      $TrainSeed"
Write-Host "  Smoke Laps:      $SmokeLaps"
Write-Host "  Eval Laps:       $EvalLaps"
Write-Host ""

# Ask for confirmation
$confirm = Read-Host "Ready to start? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Starting autoresearch agent..." -ForegroundColor Green
Write-Host "(Press Ctrl+C to stop. You can resume later with the same branch.)" -ForegroundColor Gray
Write-Host ""

# Run agent
& $pythonPath $agentScript "--branch" "$Branch" "--max-iterations" "$MaxIterations" "--eval-runs" "$EvalRuns" "--train-runs" "$TrainRuns" "--train-seed" "$TrainSeed" "--smoke-laps" "$SmokeLaps" "--eval-laps" "$EvalLaps"

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Agent loop completed successfully" -ForegroundColor Green
    Write-Host "Review results with: cat results.tsv" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "[ERROR] Agent exited with code $LASTEXITCODE" -ForegroundColor Red
}
