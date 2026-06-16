# 烛龙 v6 盈亏标签训练流水线
param(
    [string]$Symbol = 'XAUUSD',
    [switch]$Quick,
    [int]$MaxHoldBars = 12,
    [int]$CooldownBars = 12,
    [double]$SlMult = 1.2,
    [double]$TpMult = 2.0,
    [string]$LabelOut = 'data/training/XAUUSD_labeled_profit.csv'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$logDir = Join-Path $root "data/training/reports/lgb/$Symbol"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "full_train_v6.log"
$m5 = "data/training/lgb/$Symbol/${Symbol}_M5.csv"

Write-Host "== v6 step 1: profit labels ==" -ForegroundColor Cyan
py -3 scripts/generate_profit_labels.py `
    --input $m5 `
    --output $LabelOut `
    --sl-mult $SlMult `
    --tp-mult $TpMult `
    --max-hold-bars $MaxHoldBars
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== v6 step 2: train ==" -ForegroundColor Cyan
$pyArgs = @(
    '-3', 'scripts/train_binary_lgb.py',
    '--labels', $LabelOut,
    '--output', "models/$Symbol/lgb/lgb_profit.txt",
    '--acceptance-stage', 'v6',
    '--max-hold-bars', $MaxHoldBars,
    '--cooldown-bars', $CooldownBars,
    '--no-downsample'
)
if ($Quick) { $pyArgs += '--quick' }

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& py @pyArgs 2>&1 | Tee-Object -FilePath $logFile
$trainExit = $LASTEXITCODE
$ErrorActionPreference = $prevEap

Write-Host "== v6 step 3: tune threshold ==" -ForegroundColor Cyan
py -3 scripts/tune_threshold_binary.py `
    --model "models/$Symbol/lgb/lgb_profit.txt" `
    --meta "models/$Symbol/lgb/lgb_profit_meta.pkl" `
    --profit-labels `
    --target-precision 0.55 --max-signals-per-day 8

Write-Host "== v6 step 4: backtest ==" -ForegroundColor Cyan
$cfg = Get-Content "models/$Symbol/lgb/config_v6.json" | ConvertFrom-Json
py -3 scripts/backtest_binary.py --split test1 --threshold $cfg.threshold `
    --max-hold-bars $MaxHoldBars --cooldown-bars $CooldownBars --profit-labels

exit $trainExit
