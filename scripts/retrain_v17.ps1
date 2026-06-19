# V17 架构重训一键脚本
# 用法: powershell -ExecutionPolicy Bypass -File scripts/retrain_v17.ps1 [-InstallDeps]

param(
    [switch]$InstallDeps,
    [switch]$SkipShap,
    [switch]$SkipDirection,
    [switch]$SkipLocation,
    [switch]$SkipBacktest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($InstallDeps) {
    pip install -r requirements.txt
}

Write-Host "=== V17 Phase 0: SHAP + Labels ===" -ForegroundColor Cyan
if (-not $SkipShap) {
    py -3 scripts/shap_feature_audit.py --npz data/clean/training_horizon_v16.npz
}
py -3 scripts/prepare_direction_regression_labels.py
py -3 scripts/prepare_location_binary_labels.py
py -3 scripts/validate_v17_labels.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== V17 Phase 1: DirectionScorer ===" -ForegroundColor Cyan
if (-not $SkipDirection) {
    py -3 scripts/train_direction_scorer.py
    py -3 scripts/accept_direction_scorer.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "=== V17 Phase 2: LocationGate ===" -ForegroundColor Cyan
if (-not $SkipLocation) {
    py -3 scripts/train_location_gate.py
    py -3 scripts/accept_location_gate.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "=== V17 Phase 3: Full-chain backtest ===" -ForegroundColor Cyan
if (-not $SkipBacktest) {
    py -3 scripts/backtest_v17_full_chain.py --with-cost
}

Write-Host "=== V17 Phase 4: RL (use train_rl_agent.py --v16 after models pass) ===" -ForegroundColor Yellow
Write-Host "  py -3 scripts/train_rl_agent.py --v16 --symbol XAUUSD" -ForegroundColor Gray
Write-Host "V17 retrain pipeline finished." -ForegroundColor Green
