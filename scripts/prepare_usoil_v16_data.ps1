<#
.SYNOPSIS
  USOIL 原油 V16 训练数据准备（开发机，CPU 密集）

.DESCRIPTION
  1. 清洗 M5 CSV → data/clean/USOIL_M5_clean.csv
  2. 结构特征 30 维 + 方向标签 (horizon=18, gain=0.3%)
  3. 位置感知标签 P4
  4. KN2 65 维特征（需 Horizon ONNX；若无则跳过 KN2，GPU 机 Horizon 训完后再跑 prepare_kn2）

  完成后 git add + git lfs push，供训练机 GPU 训练。

.PARAMETER Jobs
  结构特征并行度（Windows 建议 1；Linux 可 4+）

.PARAMETER Quick
  仅末 50000 根冒烟

.PARAMETER SkipStruct
  已有 struct cache 时跳过全量重算

.EXAMPLE
  powershell -File scripts/prepare_usoil_v16_data.ps1
  powershell -File scripts/prepare_usoil_v16_data.ps1 -Quick 50000
#>
[CmdletBinding()]
param(
    [int]$Jobs = 1,
    [int]$Quick = 0,
    [switch]$SkipKn2,
    [switch]$FullRebuild
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Write-Step([string]$Msg) {
    Write-Host "`n=== $Msg ===" -ForegroundColor Cyan
}

Write-Step "USOIL V16 数据准备"
Write-Host "horizon=18 bars (1.5h), gain=0.3%, struct=30d" -ForegroundColor Gray

# Step 1: 清洗 CSV（不重建 horizon 若尚无 struct）
Write-Step "1/5 清洗 M5 CSV"
$cleanArgs = @("--symbol", "USOIL", "--skip-horizon", "--skip-kn2")
if ($FullRebuild) { $cleanArgs += "--full-rebuild" }
py -3 scripts/clean_training_data_v16.py @cleanArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 2: 结构特征 + horizon NPZ
Write-Step "2/5 结构特征 + Horizon NPZ (CPU, 可能数小时)"
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

# Step 3: enrich OHLCV for RL
Write-Step "3/5 enrich NPZ (RL OHLCV)"
py -3 scripts/enrich_horizon_v16_npz.py --symbol USOIL --npz data/clean/training_horizon_v16_usoil.npz
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 4: 位置感知标签
Write-Step "4/5 位置感知方向标签 P4"
py -3 scripts/prepare_horizon_v16_location_labels.py `
    --npz data/clean/training_horizon_v16_usoil.npz `
    --out data/clean/training_horizon_v16_usoil_location.npz
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipKn2) {
    Write-Step "5/5 KN2 65 维特征"
    Write-Host "若尚无 horizon ONNX，请先 GPU 训练 Horizon 再运行:" -ForegroundColor Yellow
    Write-Host "  py -3 scripts/prepare_kn2_v16_data.py --npz data/clean/training_horizon_v16_usoil.npz --out data/clean/kn2_training_v16_usoil.npz --horizon-onnx models/USOIL/v16/horizon_v16.onnx" -ForegroundColor Yellow
    py -3 scripts/prepare_kn2_v16_data.py `
        --npz data/clean/training_horizon_v16_usoil.npz `
        --out data/clean/kn2_training_v16_usoil.npz `
        --horizon-onnx models/USOIL/v16/horizon_v16.onnx `
        --horizon-scaler models/USOIL/v16/horizon_v16_scaler.pkl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "KN2 特征跳过（Horizon 模型未就绪属正常）" -ForegroundColor Yellow
    } else {
        py -3 scripts/prepare_kn2_v16_location_labels.py `
            --npz data/clean/kn2_training_v16_usoil.npz `
            --out data/clean/kn2_training_v16_usoil_location.npz
    }
}

Write-Step "完成"
Write-Host @"

下一步:
  git add data/clean/*usoil* data/clean/USOIL_M5_clean.csv
  git commit -m "Add USOIL V16 training data"
  git lfs push origin main
  git push origin main

训练机:
  git pull && git lfs pull
  powershell -File scripts/train_usoil_v16_gpu_remote.ps1

"@ -ForegroundColor Cyan
