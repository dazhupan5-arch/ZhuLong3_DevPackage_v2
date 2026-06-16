# 烛龙 v11 三分类 XGBoost 流水线
param([string]$Symbol = 'XAUUSD', [switch]$Quick, [switch]$SkipLabels)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$reportDir = "data/training/reports/v11/$Symbol"
New-Item -ItemType Directory -Force -Path $reportDir, "models/$Symbol/v11" | Out-Null

if (-not $SkipLabels) {
    Write-Host "== v11 step 1: triple labels ==" -ForegroundColor Cyan
    py -3 scripts/generate_triple_labels.py `
        --input "data/training/lgb/$Symbol/${Symbol}_M5.csv" `
        --output "data/training/${Symbol}_labeled_triple.csv" `
        --horizon 12 --gain 0.0020
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "== v11 step 2: resample ==" -ForegroundColor Cyan
py -3 scripts/resample_triple.py --input "data/training/${Symbol}_labeled_triple.csv"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== v11 step 3: train ==" -ForegroundColor Cyan
$trainArgs = @('-3', 'scripts/train_triple_xgb.py', '--symbol', $Symbol)
if ($Quick) { $trainArgs += '--quick' }
py @trainArgs
$exit = $LASTEXITCODE

Write-Host "== v11 step 4: tune + backtest ==" -ForegroundColor Cyan
py -3 scripts/tune_threshold_triple.py --symbol $Symbol
py -3 scripts/backtest_triple.py --symbol $Symbol --split test1
py -3 scripts/write_acceptance_v11.py --symbol $Symbol

exit $exit
