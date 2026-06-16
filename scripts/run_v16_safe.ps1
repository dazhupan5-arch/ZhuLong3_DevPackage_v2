# V16 安全管线：单线程结构特征，避免 CPU 打满
# 用法: .\scripts\run_v16_safe.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Step 1: prepare (50k bars, jobs=1) ===" -ForegroundColor Cyan
py -3 -u scripts/prepare_horizon_v16_data.py --symbol XAUUSD --quick 50000 --jobs 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Step 2: train horizon ===" -ForegroundColor Cyan
py -3 -u scripts/train_horizon_v16.py --epochs 60
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Step 3: backtest ===" -ForegroundColor Cyan
py -3 -u scripts/backtest_v16.py --start 2025-01-01 --end 2025-12-31
exit $LASTEXITCODE
