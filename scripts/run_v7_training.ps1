# 烛龙 v7 LSTM 端到端训练
param(
    [string]$Symbol = 'XAUUSD',
    [switch]$Quick,
    [string]$LstmUnits = '64,32'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$logDir = "data/training/reports/lstm/$Symbol"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "== v7 step 0: install tensorflow ==" -ForegroundColor Cyan
py -3 -m pip install "tensorflow>=2.15" --quiet

Write-Host "== v7 step 1: prepare LSTM data ==" -ForegroundColor Cyan
py -3 scripts/prepare_lstm_data.py `
    --symbol $Symbol `
    --input "data/training/lgb/$Symbol/${Symbol}_M5.csv" `
    --labels "data/training/${Symbol}_labeled_profit_24.csv" `
    --output-dir "data/training/lstm/$Symbol"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== v7 step 2: train LSTM ==" -ForegroundColor Cyan
$trainArgs = @(
    '-3', 'scripts/train_lstm.py',
    '--symbol', $Symbol,
    '--data-dir', "data/training/lstm/$Symbol",
    '--output', "models/$Symbol/lstm/lstm_model.keras",
    '--lstm-units', $LstmUnits
)
if ($Quick) { $trainArgs += '--quick' }

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& py @trainArgs 2>&1 | Tee-Object -FilePath "$logDir/full_train_v7.log"
$trainExit = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($trainExit -ne 0) { exit $trainExit }

Write-Host "== v7 step 3: tune threshold ==" -ForegroundColor Cyan
py -3 scripts/tune_threshold_lstm.py --symbol $Symbol

Write-Host "== v7 step 4: backtest ==" -ForegroundColor Cyan
$cfg = Get-Content "models/$Symbol/lstm/config_v7.json" | ConvertFrom-Json
py -3 scripts/backtest_lstm.py --symbol $Symbol --split test --threshold $cfg.threshold

Write-Host "== v7 step 5: acceptance report ==" -ForegroundColor Cyan
py -3 scripts/write_acceptance_v7.py --symbol $Symbol

exit $trainExit
