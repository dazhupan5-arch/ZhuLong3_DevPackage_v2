# 烛龙 USOIL v1 三分类 XGBoost 训练流水线
param([string]$Symbol = 'USOIL', [switch]$Quick, [switch]$SkipLabels, [switch]$SkipFeatures)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$reportDir = "data/training/reports/oil_v1/$Symbol"
New-Item -ItemType Directory -Force -Path $reportDir, "models/$Symbol/v1", "data/training/oil_v1/$Symbol" | Out-Null

if (-not $SkipFeatures) {
    Write-Host "== oil v1 step 1: features + VMD ==" -ForegroundColor Cyan
    $featArgs = @('-3', 'scripts/prepare_oil_v1_data.py', '--symbol', $Symbol)
    if ($Quick) { $featArgs += '--skip-decompose' }
    py @featArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipLabels) {
    Write-Host "== oil v1 step 2: triple labels ==" -ForegroundColor Cyan
    py -3 scripts/generate_oil_labels.py `
        --input "data/training/lgb/$Symbol/${Symbol}_M5.csv" `
        --output "data/training/${Symbol}_labeled_triple.csv" `
        --horizon 18 --gain-fixed 0.003
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "== oil v1 step 3: resample ==" -ForegroundColor Cyan
py -3 scripts/resample_triple.py `
    --input "data/training/${Symbol}_labeled_triple.csv" `
    --output "data/train_balanced_triple_oil.csv" `
    --features "data/training/oil_v1/$Symbol/features.parquet" `
    --feature-cols "data/training/oil_v1/$Symbol/feature_columns.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== oil v1 step 4: train ==" -ForegroundColor Cyan
$trainArgs = @('-3', 'scripts/train_oil_v1.py', '--symbol', $Symbol)
if ($Quick) { $trainArgs += '--quick' }
py @trainArgs
$exit = $LASTEXITCODE

Write-Host "== oil v1 step 5: acceptance report ==" -ForegroundColor Cyan
py -3 scripts/write_acceptance_oil_v1.py --symbol $Symbol

exit $exit
