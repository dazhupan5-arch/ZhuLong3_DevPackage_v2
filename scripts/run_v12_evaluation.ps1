# v12 双向优化评估
param(
    [string]$Symbol = 'XAUUSD',
    [switch]$Retrain
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path "data/training/reports/v12/$Symbol" | Out-Null

py -3 scripts/eval_v12.py --symbol $Symbol --split test1 --model-version v11
$exit = $LASTEXITCODE

if ($Retrain -or $exit -ne 0) {
    Write-Host "=== v12 fallback 5.1: short boost retrain ==="
    py -3 scripts/train_v12_xgb.py --symbol $Symbol
    py -3 scripts/eval_v12.py --symbol $Symbol --split test1 --model-version v12
    $exit = $LASTEXITCODE
}

py -3 scripts/write_acceptance_v12.py --symbol $Symbol
exit $exit
