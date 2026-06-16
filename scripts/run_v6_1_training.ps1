# 烛龙 v6.1 盈亏标签（24 根 M5 / 2 小时持仓）
param(
    [string]$Symbol = 'XAUUSD',
    [switch]$Quick,
    [int]$MaxHoldBars = 24,
    [int]$CooldownBars = 12
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$logDir = Join-Path $root "data/training/reports/lgb/$Symbol"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "full_train_v61.log"
$m5 = "data/training/lgb/$Symbol/${Symbol}_M5.csv"
$labels = "data/training/${Symbol}_labeled_profit_24.csv"

Write-Host "== v6.1 step 1: profit labels (hold=$MaxHoldBars) ==" -ForegroundColor Cyan
py -3 scripts/generate_profit_labels.py `
    --input $m5 `
    --output $labels `
    --sl-mult 1.2 `
    --tp-mult 2.0 `
    --max-hold-bars $MaxHoldBars
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== v6.1 step 2: train ==" -ForegroundColor Cyan
$pyArgs = @(
    '-3', 'scripts/train_binary_lgb.py',
    '--labels', $labels,
    '--output', "models/$Symbol/lgb/lgb_profit_24.txt",
    '--acceptance-stage', 'v61',
    '--max-hold-bars', $MaxHoldBars,
    '--cooldown-bars', $CooldownBars,
    '--target-precision', '0.50',
    '--target-recall', '0.15',
    '--no-downsample'
)
if ($Quick) { $pyArgs += '--quick' }

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& py @pyArgs 2>&1 | Tee-Object -FilePath $logFile
$trainExit = $LASTEXITCODE
$ErrorActionPreference = $prevEap

Write-Host "== v6.1 step 3: tune threshold ==" -ForegroundColor Cyan
py -3 scripts/tune_threshold_binary.py `
    --labels-file $labels `
    --model "models/$Symbol/lgb/lgb_profit_24.txt" `
    --meta "models/$Symbol/lgb/lgb_profit_24_meta.pkl" `
    --profit-labels `
    --acceptance-stage v61 `
    --target-precision 0.50 --target-recall 0.15 --max-signals-per-day 8

Write-Host "== v6.1 step 4: backtest ==" -ForegroundColor Cyan
$cfg = Get-Content "models/$Symbol/lgb/config_v61.json" | ConvertFrom-Json
py -3 scripts/backtest_binary.py --split test1 --threshold $cfg.threshold `
    --model "models/$Symbol/lgb/lgb_profit_24.txt" `
    --meta "models/$Symbol/lgb/lgb_profit_24_meta.pkl" `
    --max-hold-bars $MaxHoldBars --cooldown-bars $CooldownBars --profit-labels

exit $trainExit
