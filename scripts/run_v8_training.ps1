# 烛龙 v8 多尺度分解 + 双模型集成
param(
    [string]$Symbol = 'XAUUSD',
    [switch]$Quick,
    [switch]$SkipDecompose
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$logDir = "data/training/reports/v8/$Symbol"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "== v8 step 0: deps ==" -ForegroundColor Cyan
py -3 -m pip install vmdpy EMD-signal shap yfinance --quiet

Write-Host "== v8 step 1: prepare data ==" -ForegroundColor Cyan
$prepArgs = @('-3', 'scripts/prepare_v8_data.py', '--symbol', $Symbol)
if ($SkipDecompose) { $prepArgs += '--skip-decompose' }
py @prepArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== v8 step 2: train ==" -ForegroundColor Cyan
$trainArgs = @('-3', 'scripts/train_v8.py', '--symbol', $Symbol)
if ($Quick) { $trainArgs += '--quick' }
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& py @trainArgs 2>&1 | Tee-Object -FilePath "$logDir/full_train_v8.log"
$trainExit = $LASTEXITCODE
$ErrorActionPreference = $prevEap

Write-Host "== v8 step 3: SHAP ==" -ForegroundColor Cyan
py -3 scripts/shap_v8.py --symbol $Symbol

Write-Host "== v8 step 4: report ==" -ForegroundColor Cyan
py -3 scripts/write_acceptance_v8.py --symbol $Symbol

exit $trainExit
