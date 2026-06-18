<#
.SYNOPSIS
  USOIL 原油 V16 跨机 GPU 整套训练（Horizon → ONNX → KN2 → PPO）

.DESCRIPTION
  训练机 workflow:
    git pull
    git lfs pull
    powershell -ExecutionPolicy Bypass -File scripts/train_usoil_v16_gpu_remote.ps1 -InstallDeps

  默认会先跑 scripts/clean_usoil_v16_remote.ps1（清洗 + 结构特征 + location NPZ）。
  若 NPZ 已就绪可加 -SkipPrepare 跳过。

  开发机也可单独跑:
    powershell -File scripts/clean_usoil_v16_remote.ps1
    powershell -File scripts/prepare_usoil_v16_data.ps1

.PARAMETER InstallDeps
  首次 GPU 机：安装 requirements + CUDA torch

.PARAMETER SkipHorizon
  跳过 Horizon 训练（已有 horizon_v16.pth 时）

.PARAMETER SkipKn2
  跳过 KN2 训练

.PARAMETER SkipPrepare
  跳过数据清洗/NPZ 准备（已有 location NPZ 时）

.PARAMETER PrepareOnly
  仅清洗 + 准备 NPZ，不训练

.PARAMETER PrepareJobs
  结构特征并行度（传给 clean_usoil_v16_remote.ps1）

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/train_usoil_v16_gpu_remote.ps1 -InstallDeps
#>
[CmdletBinding()]
param(
    [switch]$InstallDeps,
    [switch]$SkipPrepare,
    [switch]$PrepareOnly,
    [int]$PrepareJobs = 1,
    [switch]$SkipHorizon,
    [switch]$SkipKn2,
    [switch]$SkipRl,
    [switch]$SkipDataCheck,
    [int]$HorizonEpochs = 100,
    [int]$Kn2Epochs = 120,
    [int]$Kn2BatchSize = 48
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Write-Step([string]$Msg) {
    Write-Host "`n=== $Msg ===" -ForegroundColor Cyan
}

$Symbol = "USOIL"
$V16Dir = Join-Path $Root "models\USOIL\v16"
$HorizonNpz = Join-Path $Root "data\clean\training_horizon_v16_usoil_location.npz"
$Kn2Npz = Join-Path $Root "data\clean\kn2_training_v16_usoil_location.npz"

Write-Step "环境 ($Symbol V16)"
py -3 --version
if ($InstallDeps) {
    py -3 -m pip install -U pip
    py -3 -m pip install -r requirements.txt
    py -3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
}

py -3 -c @"
import torch
print('torch', torch.__version__)
ok = torch.cuda.is_available()
print('cuda_available', ok)
if ok:
    print('gpu', torch.cuda.get_device_name(0))
else:
    raise SystemExit('CUDA 不可用：请 -InstallDeps 或安装 cu124 PyTorch')
"@

if (-not $SkipPrepare) {
    if (-not (Test-Path $HorizonNpz)) {
        Write-Step "数据准备 (clean + struct + location NPZ)"
        & (Join-Path $Root "scripts\clean_usoil_v16_remote.ps1") -Jobs $PrepareJobs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Write-Host "已有 $HorizonNpz，跳过数据准备（需重跑请加 -FullRebuild 到 clean_usoil_v16_remote.ps1 或删 NPZ）" -ForegroundColor Gray
    }
}

if ($PrepareOnly) {
    Write-Host "PrepareOnly：数据准备完成，未启动 GPU 训练。" -ForegroundColor Green
    exit 0
}

if (-not $SkipDataCheck) {
    Write-Step "数据校验 (git lfs)"
    if (-not (Test-Path $HorizonNpz)) {
        Write-Host @"

缺少: $HorizonNpz

训练机先清洗:
  powershell -File scripts/clean_usoil_v16_remote.ps1

或一键（含清洗）:
  powershell -File scripts/train_usoil_v16_gpu_remote.ps1 -InstallDeps

"@ -ForegroundColor Yellow
        exit 1
    }
    $mb = [math]::Round((Get-Item $HorizonNpz).Length / 1MB, 1)
    Write-Host "OK $HorizonNpz (${mb} MB)" -ForegroundColor Green
    if (-not (Test-Path $Kn2Npz)) {
        Write-Host "KN2 NPZ 尚未就绪 — Horizon 训完 ONNX 后自动生成" -ForegroundColor Yellow
    } else {
        $mb2 = [math]::Round((Get-Item $Kn2Npz).Length / 1MB, 1)
        Write-Host "OK $Kn2Npz (${mb2} MB)" -ForegroundColor Green
    }
}

New-Item -ItemType Directory -Force -Path $V16Dir | Out-Null
$logDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not $SkipHorizon) {
    Write-Step "Step 1/4 Horizon V16 ($Symbol, hidden=64, horizon=18, gain=0.3%)"
    $log = Join-Path $logDir "usoil_horizon_v16_$ts.log"
    py -3 -u scripts/train_horizon_v16.py `
        --symbol USOIL `
        --npz data/clean/training_horizon_v16_usoil_location.npz `
        --label-mode location `
        --temporal-val `
        --train-end 2024-12-31 `
        --epochs $HorizonEpochs `
        --patience 18 `
        --class-weights 2.6,1.0,2.6 `
        --smote-ratio 0.55 `
        --lr 0.00035 `
        --hidden-dim 64 `
        --log-suffix usoil_p4 `
        2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Step "Step 2/4 导出 Horizon ONNX"
    py -3 scripts/convert_knowledge_net_to_onnx.py `
        --model models/USOIL/v16/horizon_v16.pth `
        --out models/USOIL/v16/horizon_v16.onnx `
        --no-benchmark
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipKn2) {
    if (-not (Test-Path $Kn2Npz)) {
        Write-Step "Step 2b/4 生成 KN2 65 维特征 (Horizon ONNX 推理)"
        py -3 scripts/prepare_kn2_v16_data.py `
            --npz data/clean/training_horizon_v16_usoil.npz `
            --out data/clean/kn2_training_v16_usoil.npz `
            --horizon-onnx models/USOIL/v16/horizon_v16.onnx `
            --horizon-scaler models/USOIL/v16/horizon_v16_scaler.pkl
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        py -3 scripts/prepare_kn2_v16_location_labels.py `
            --npz data/clean/kn2_training_v16_usoil.npz `
            --out data/clean/kn2_training_v16_usoil_location.npz
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    Write-Step "Step 3/4 KN2 V16 GRU ($Symbol)"
    $log = Join-Path $logDir "usoil_kn2_v16_$ts.log"
    py -3 -u scripts/train_kn2_v16.py `
        --npz data/clean/kn2_training_v16_usoil_location.npz `
        --label-mode location `
        --mode fast `
        --device auto `
        --batch-size $Kn2BatchSize `
        --epochs $Kn2Epochs `
        --patience 25 `
        --hidden-dim 192 `
        --output models/USOIL/v16/kn2_trader_v16.pth `
        2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipRl) {
    Write-Step "Step 4/4 PPO RL ($Symbol V16 state=86d)"
    $log = Join-Path $logDir "usoil_rl_v16_$ts.log"
    py -3 -u scripts/train_rl_v16_oil.py 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Step "完成 — 拷回开发机"
$artifacts = @(
    "models\USOIL\v16\horizon_v16.pth",
    "models\USOIL\v16\horizon_v16.onnx",
    "models\USOIL\v16\horizon_v16_scaler.pkl",
    "models\USOIL\v16\horizon_v16.meta.json",
    "models\USOIL\v16\kn2_trader_v16.pth",
    "models\USOIL\v16\kn2_trader_v16.meta.json",
    "models\rl_agent_oil.zip",
    "models\USOIL\v16\rl_meta.json"
)
foreach ($rel in $artifacts) {
    $full = Join-Path $Root $rel
    if (Test-Path $full) {
        Write-Host ("  OK  {0}" -f $rel) -ForegroundColor Green
    } else {
        Write-Host ("  SKIP  {0}" -f $rel) -ForegroundColor Yellow
    }
}

Write-Host @"

开发机：合并模型后更新 config_agent.json 中 USOIL 的 V16 路径并验收。

"@ -ForegroundColor Cyan
