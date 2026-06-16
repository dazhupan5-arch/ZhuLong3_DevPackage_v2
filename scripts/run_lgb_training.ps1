# 烛龙 v4.2 LightGBM 多分类训练流水线（60min 预测，4h 持仓）
param(
    [string]$Symbol = 'XAUUSD',
    [string]$InputCsv = 'C:\Users\xiaomi\Desktop\XAUUSD5.csv',
    [switch]$Quick,
    [switch]$FullGrid,
    [string]$AcceptanceStage = 'v42',
    [int]$Horizon = 12,
    [double]$Gain = 0.0025,
    [double]$TargetPrecision = 0.50,
    [double]$MaxSignalsPerDay = 8.0
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

py -3 -m pip install lightgbm pyarrow -q

$logDir = Join-Path $root "data/training/reports/lgb/$Symbol"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "full_train_v4_2.log"

$pyArgs = @(
    '-3', 'scripts/train_lightgbm.py',
    '--symbol', $Symbol,
    '--input', $InputCsv,
    '--horizon', $Horizon,
    '--gain', $Gain,
    '--acceptance-stage', $AcceptanceStage,
    '--target-precision', $TargetPrecision,
    '--max-signals-per-day', $MaxSignalsPerDay,
    '--labels-file', (Join-Path $root "data/training/${Symbol}_labeled_v4_2.csv"),
    '--skip-import'
)
if ($Quick) { $pyArgs += '--quick' }
if ($FullGrid) { $pyArgs += '--full-grid' }

Write-Host "== LightGBM v4.2 $Symbol h=$Horizon gain=$Gain -> $logFile ==" -ForegroundColor Cyan
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& py @pyArgs 2>&1 | Tee-Object -FilePath $logFile
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $prevEap
exit $exitCode
