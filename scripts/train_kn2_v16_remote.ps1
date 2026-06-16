<#
.SYNOPSIS
  KN2 V16 跨机 GPU 训练（结构位置标签版）

.DESCRIPTION
  1. 检查 Python / CUDA / PyTorch
  2. 可选安装 CUDA 版 PyTorch 与项目依赖
  3. 校验 data/clean/kn2_training_v16_location.npz（git pull + git lfs pull）
  4. 以 fast 模式 + location 标签训练 kn2_trader_v16.pth
  5. 自动跑 accept_kn2_v16.py

.PARAMETER InstallDeps
  首次运行时加此开关：pip install -r requirements.txt + CUDA torch

.PARAMETER BatchSize
  RTX 3050 8GB 建议 32–48，OOM 改 24

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/train_kn2_v16_remote.ps1 -InstallDeps
  powershell -ExecutionPolicy Bypass -File scripts/train_kn2_v16_remote.ps1
#>
[CmdletBinding()]
param(
    [switch]$InstallDeps,
    [int]$BatchSize = 48,
    [int]$Epochs = 120,
    [int]$Patience = 25,
    [switch]$SkipDataCheck,
    [switch]$SkipAccept
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
    Write-Host "Installing requirements.txt ..."
    py -3 -m pip install -U pip
    py -3 -m pip install -r requirements.txt
    Write-Host "Installing PyTorch CUDA 12.4 wheel ..."
    py -3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
}

py -3 -c @"
import torch
print('torch', torch.__version__)
ok = torch.cuda.is_available()
print('cuda_available', ok)
if ok:
    print('gpu', torch.cuda.get_device_name(0))
    print('vram_gb', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
else:
    raise SystemExit('CUDA 不可用：请安装 NVIDIA 驱动 + cu124 版 PyTorch（-InstallDeps）')
"@

$npz = Join-Path $Root "data\clean\kn2_training_v16_location.npz"
if (-not $SkipDataCheck) {
    Write-Step "数据 (git lfs)"
    if (-not (Test-Path $npz)) {
        Write-Host @"

缺少位置标签 NPZ: $npz

在 GPU 机器上：
  git pull
  git lfs pull

本地生成（开发机）：
  py -3 scripts/prepare_kn2_v16_location_labels.py

"@ -ForegroundColor Yellow
        exit 1
    }
    $sizeMb = [math]::Round((Get-Item $npz).Length / 1MB, 1)
    py -3 -c "import numpy as np; d=np.load(r'$npz', allow_pickle=True); print('rows', d['market_feat'].shape[0], 'dim', d['market_feat'].shape[1], 'loc_action' in d.files)"
    Write-Host "npz size: ${sizeMb} MB" -ForegroundColor Green
}

Write-Step "KN2 V16 训练 (location labels, mode=fast, batch=$BatchSize)"
$logDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("kn2_v16_location_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

py -3 -u scripts/train_kn2_v16.py `
    --npz data/clean/kn2_training_v16_location.npz `
    --label-mode location `
    --mode fast `
    --device auto `
    --batch-size $BatchSize `
    --epochs $Epochs `
    --patience $Patience `
    2>&1 | Tee-Object -FilePath $log

if ($LASTEXITCODE -ne 0) {
    Write-Host "训练失败，见日志: $log" -ForegroundColor Red
    exit $LASTEXITCODE
}

if (-not $SkipAccept) {
    Write-Step "验收 accept_kn2_v16.py"
    py -3 scripts/accept_kn2_v16.py `
        --model models/kn2_trader_v16.pth `
        --npz data/clean/kn2_training_v16_location.npz `
        --label-mode location
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
        Write-Host "验收脚本异常 exit=$LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Step "完成 — 拷回开发机以下文件"
$artifacts = @(
    "models\kn2_trader_v16.pth",
    "models\kn2_trader_v16.meta.json",
    "data\training\reports\kn2_v16\train_report.json",
    "data\training\reports\kn2_v16\acceptance_report.json"
)
foreach ($rel in $artifacts) {
    $full = Join-Path $Root $rel
    if (Test-Path $full) {
        $fi = Get-Item $full
        Write-Host ("  OK  {0}  ({1:N0} bytes)" -f $rel, $fi.Length) -ForegroundColor Green
    } else {
        Write-Host ("  MISSING  {0}" -f $rel) -ForegroundColor Yellow
    }
}

Write-Host @"

开发机部署（模型拷回后）:
  py -3 scripts/deploy_kn2_v16_when_ready.ps1

"@ -ForegroundColor Cyan
