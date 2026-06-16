<#
.SYNOPSIS
  Horizon V16 P4 跨机 GPU 训练（位置感知方向标签）

.DESCRIPTION
  1. 检查 Python / CUDA / PyTorch
  2. 可选 -InstallDeps 安装 CUDA torch + requirements
  3. 校验 data/clean/training_horizon_v16_location.npz（git pull + git lfs pull）
  4. GPU 训练 horizon_v16.pth（location 标签）
  5. 校准 + 导出 ONNX + accept_horizon_v16.py

.PARAMETER InstallDeps
  首次 GPU 机运行加此开关

.EXAMPLE
  git pull; git lfs pull
  powershell -ExecutionPolicy Bypass -File scripts/train_horizon_v16_remote.ps1 -InstallDeps
  powershell -ExecutionPolicy Bypass -File scripts/train_horizon_v16_remote.ps1
#>
[CmdletBinding()]
param(
    [switch]$InstallDeps,
    [int]$Epochs = 80,
    [int]$Patience = 15,
    [switch]$SkipDataCheck,
    [switch]$SkipAccept,
    [switch]$SkipCalibrate,
    [string]$LabelMode = "location"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Write-Step([string]$Msg) {
    Write-Host "`n=== $Msg ===" -ForegroundColor Cyan
}

Write-Step "环境"
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

$npz = Join-Path $Root "data\clean\training_horizon_v16_location.npz"
if (-not $SkipDataCheck) {
    Write-Step "P4 数据 (git lfs)"
    if (-not (Test-Path $npz)) {
        Write-Host @"

缺少位置标签 NPZ: $npz

GPU 机器：
  git pull
  git lfs pull

开发机生成：
  py -3 scripts/prepare_horizon_v16_location_labels.py

"@ -ForegroundColor Yellow
        exit 1
    }
    $sizeMb = [math]::Round((Get-Item $npz).Length / 1MB, 1)
    py -3 -c "import numpy as np; d=np.load(r'$npz', allow_pickle=True); print('rows', len(d['struct']), 'loc_ver', 'loc_horizon_version' in d.files, 'legacy' in d.files)"
    Write-Host "npz size: ${sizeMb} MB" -ForegroundColor Green
}

Write-Step "Horizon V16 训练 (label-mode=$LabelMode, temporal-val)"
$logDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("horizon_v16_location_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

py -3 -u scripts/train_horizon_v16.py `
    --npz data/clean/training_horizon_v16_location.npz `
    --label-mode $LabelMode `
    --temporal-val `
    --train-end 2024-12-31 `
    --epochs $Epochs `
    --patience $Patience `
    --class-weights 2.4,1.1,2.4 `
    --smote-ratio 0.52 `
    --lr 0.00025 `
    --log-suffix location_p4 `
    2>&1 | Tee-Object -FilePath $log

if ($LASTEXITCODE -ne 0) {
    Write-Host "训练失败，见 $log" -ForegroundColor Red
    exit $LASTEXITCODE
}

if (-not $SkipCalibrate) {
    Write-Step "验证集 F1 校准"
    py -3 scripts/calibrate_horizon_v16.py --temporal-val --apply --npz data/clean/training_horizon_v16_location.npz
}

Write-Step "导出 ONNX"
py -3 scripts/convert_knowledge_net_to_onnx.py --model models/horizon_v16.pth --out models/horizon_v16.onnx --no-benchmark
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipAccept) {
    Write-Step "验收 accept_horizon_v16.py"
    py -3 -u scripts/accept_horizon_v16.py --apply
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
        Write-Host "验收 exit=$LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Step "完成 — 拷回开发机"
$artifacts = @(
    "models\horizon_v16.pth",
    "models\horizon_v16.onnx",
    "models\horizon_v16_scaler.pkl",
    "models\horizon_v16.meta.json",
    "data\training\reports\horizon_v16\location_label_report.json",
    "data\training\reports\v16\acceptance_report.json"
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

开发机部署：
  powershell -File scripts/deploy_horizon_v16_production.ps1
  powershell -File scripts/deploy_v16_execution_gates.ps1

"@ -ForegroundColor Cyan
