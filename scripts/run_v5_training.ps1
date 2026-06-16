# 烛龙 v5 二分类训练流水线
param(
    [string]$Symbol = 'XAUUSD',
    [switch]$Quick,
    [double]$Gain = 0.0025,
    [int]$Horizon = 12
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$logDir = Join-Path $root "data/training/reports/lgb/$Symbol"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "full_train_v5.log"

Write-Host "== v5 step 1: binary labels ==" -ForegroundColor Cyan
py -3 scripts/generate_binary_labels.py `
    --input "data/training/${Symbol}_labeled_v4_2.csv" `
    --output "data/training/${Symbol}_labeled_v5.csv"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== v5 step 2: resample train 1:5 ==" -ForegroundColor Cyan
py -3 scripts/resample_binary.py `
    --input "data/training/${Symbol}_labeled_v5.csv" `
    --output "data/train_balanced.csv" `
    --pos-neg-ratio 5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== v5 step 3: train binary LGB ==" -ForegroundColor Cyan
$pyArgs = @(
    '-3', 'scripts/train_binary_lgb.py',
    '--symbol', $Symbol,
    '--labels', "data/training/${Symbol}_labeled_v5.csv",
    '--horizon', $Horizon,
    '--gain', $Gain
)
if ($Quick) { $pyArgs += '--quick' }

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& py @pyArgs 2>&1 | Tee-Object -FilePath $logFile
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $prevEap
exit $exitCode
