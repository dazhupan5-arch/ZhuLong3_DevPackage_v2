# 烛龙 v5.1 二分类训练流水线（gain=0.20%，60min 冷却）
param(
    [string]$Symbol = 'XAUUSD',
    [switch]$Quick,
    [double]$Gain = 0.0020,
    [int]$Horizon = 12,
    [int]$CooldownBars = 12
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$logDir = Join-Path $root "data/training/reports/lgb/$Symbol"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "full_train_v5_1.log"
$m5 = "data/training/lgb/$Symbol/${Symbol}_M5.csv"

Write-Host "== v5.1 step 1: labels gain=$Gain ==" -ForegroundColor Cyan
py -3 -m zhulong.training.lgb.labels --input $m5 --output "data/training/${Symbol}_labeled_v5_1.csv" --horizon $Horizon --gain $Gain
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== v5.1 step 2: binary labels + resample ==" -ForegroundColor Cyan
py -3 scripts/generate_binary_labels.py --input "data/training/${Symbol}_labeled_v5_1.csv" --output "data/training/${Symbol}_labeled_v5_1_binary.csv"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -3 scripts/resample_binary.py --input "data/training/${Symbol}_labeled_v5_1_binary.csv" --output "data/train_balanced_v5_1.csv" --pos-neg-ratio 5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== v5.1 step 3: train ==" -ForegroundColor Cyan
$pyArgs = @(
    '-3', 'scripts/train_binary_lgb.py',
    '--labels', "data/training/${Symbol}_labeled_v5_1_binary.csv",
    '--output', "models/$Symbol/lgb/lgb_binary_v5_1.txt",
    '--horizon', $Horizon,
    '--gain', $Gain,
    '--acceptance-stage', 'v51',
    '--cooldown-bars', $CooldownBars
)
if ($Quick) { $pyArgs += '--quick' }

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& py @pyArgs 2>&1 | Tee-Object -FilePath $logFile
$trainExit = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($trainExit -ne 0 -and $trainExit -ne 1) { exit $trainExit }

Write-Host "== v5.1 step 4: tune threshold ==" -ForegroundColor Cyan
py -3 scripts/tune_threshold_binary.py `
    --model "models/$Symbol/lgb/lgb_binary_v5_1.txt" `
    --meta "models/$Symbol/lgb/lgb_binary_v5_1_meta.pkl" `
    --horizon $Horizon --gain $Gain `
    --target-precision 0.45 --max-signals-per-day 8

Write-Host "== v5.1 step 5: backtest with cooldown ==" -ForegroundColor Cyan
$thr = (Get-Content "models/$Symbol/lgb/config_v5_1.json" | ConvertFrom-Json).threshold
py -3 scripts/backtest_binary.py --split test1 --threshold $thr --cooldown-bars $CooldownBars --gain $Gain

exit $trainExit
