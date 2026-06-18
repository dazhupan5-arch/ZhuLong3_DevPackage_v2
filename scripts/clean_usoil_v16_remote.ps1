<#
.SYNOPSIS
  USOIL V16 训练机数据清洗 + NPZ 准备（git pull 后第一步）

.DESCRIPTION
  从仓库内原始 M5 CSV 清洗并生成 Horizon 训练 NPZ（含 P4 location 标签）。
  训练机 workflow:
    git pull && git lfs pull
    powershell -File scripts/clean_usoil_v16_remote.ps1
    powershell -File scripts/train_usoil_v16_gpu_remote.ps1 -SkipPrepare

  原始 CSV 候选（按顺序）:
    data/training/lgb/USOIL/USOIL_M5.csv
    data/training/USOIL_M5.csv
    data/USOIL_M5.csv

.PARAMETER Jobs
  结构特征并行度（Windows 建议 1）

.PARAMETER SkipStruct
  仅清洗 CSV，不跑结构特征（已有 NPZ 时）

.PARAMETER FullRebuild
  强制重算结构特征（忽略 checkpoint）

.PARAMETER Quick
  仅末 N 根冒烟测试

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/clean_usoil_v16_remote.ps1
  powershell -File scripts/clean_usoil_v16_remote.ps1 -Jobs 4
#>
[CmdletBinding()]
param(
    [int]$Jobs = 1,
    [switch]$SkipStruct,
    [switch]$FullRebuild,
    [int]$Quick = 0
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Write-Step([string]$Msg) {
    Write-Host "`n=== $Msg ===" -ForegroundColor Cyan
}

$rawCandidates = @(
    "data/training/lgb/USOIL/USOIL_M5.csv",
    "data/training/USOIL_M5.csv",
    "data/USOIL_M5.csv"
)
$rawCsv = $null
foreach ($rel in $rawCandidates) {
    $p = Join-Path $Root $rel
    if (Test-Path $p) {
        $rawCsv = $p
        break
    }
}
if (-not $rawCsv) {
    Write-Error @"
未找到 USOIL 原始 M5 CSV。请 git lfs pull 或放置于:
  $($rawCandidates -join "`n  ")
"@
}

$rawRel = [System.IO.Path]::GetRelativePath($Root, $rawCsv) -replace '\\', '/'
Write-Step "USOIL V16 训练机数据准备"
Write-Host "原始 CSV: $rawRel" -ForegroundColor Gray
Write-Host "horizon=18 (1.5h), gain=0.3%, struct=30d" -ForegroundColor Gray

Write-Step "1/4 清洗 M5 → data/clean/USOIL_M5_clean.csv"
$cleanArgs = @(
    "--symbol", "USOIL",
    "--csv", $rawRel,
    "--skip-horizon",
    "--skip-kn2"
)
if ($FullRebuild) { $cleanArgs += "--full-rebuild" }
py -3 scripts/clean_training_data_v16.py @cleanArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($SkipStruct) {
    Write-Host "SkipStruct：已写入 clean CSV，跳过 NPZ" -ForegroundColor Yellow
    exit 0
}

Write-Step "2/4 结构特征 + Horizon NPZ (CPU，可数小时，支持断点续跑)"
$prepArgs = @(
    "--symbol", "USOIL",
    "--horizon", "18",
    "--gain", "0.003",
    "--csv", "data/clean/USOIL_M5_clean.csv",
    "--out", "data/clean/training_horizon_v16_usoil.npz",
    "--jobs", "$Jobs"
)
if ($Quick -gt 0) { $prepArgs += @("--quick", "$Quick") }
if ($FullRebuild) { $prepArgs += "--rebuild" }
py -3 scripts/prepare_horizon_v16_data.py @prepArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "3/4 enrich OHLCV (RL 回测用)"
py -3 scripts/enrich_horizon_v16_npz.py --symbol USOIL --npz data/clean/training_horizon_v16_usoil.npz
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "4/4 位置感知方向标签 P4"
py -3 scripts/prepare_horizon_v16_location_labels.py `
    --npz data/clean/training_horizon_v16_usoil.npz `
    --out data/clean/training_horizon_v16_usoil_location.npz
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "数据就绪"
foreach ($rel in @(
    "data/clean/USOIL_M5_clean.csv",
    "data/clean/training_horizon_v16_usoil.npz",
    "data/clean/training_horizon_v16_usoil_location.npz"
)) {
    $p = Join-Path $Root $rel
    if (Test-Path $p) {
        $mb = [math]::Round((Get-Item $p).Length / 1MB, 1)
        Write-Host ("  OK  {0} ({1} MB)" -f $rel, $mb) -ForegroundColor Green
    }
}

Write-Host @"

下一步 GPU 训练:
  powershell -ExecutionPolicy Bypass -File scripts/train_usoil_v16_gpu_remote.ps1 -SkipPrepare

"@ -ForegroundColor Cyan
