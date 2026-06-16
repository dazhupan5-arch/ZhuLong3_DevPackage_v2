# KN2 V16 GPU 机器训练（RTX 3050 等）
# 用法：在 GPU 机器 clone 代码后，把 data/kn2_training_v16.npz 拷到同路径，再运行本脚本。
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$npz = Join-Path (Get-Location) "data\kn2_training_v16.npz"
if (-not (Test-Path $npz)) {
    Write-Error "缺少 $npz — 从开发机拷贝，或先运行 prepare_kn2_v16_data.py"
}

Write-Host "=== PyTorch / CUDA 检查 ===" -ForegroundColor Cyan
py -3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU — install cu124 torch')"

$logDir = Join-Path (Get-Location) "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("kn2_v16_gpu_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

Write-Host "=== KN2 V16 GPU TRAIN (fast mode) ===" -ForegroundColor Cyan
Write-Host "Log: $log"

py -3 -u scripts/train_kn2_v16.py `
    --mode fast `
    --device auto `
    --batch-size 48 `
    --epochs 120 `
    --patience 25 `
    --class-weights "0.85,2.5,2.5,1.0,1.0,1.0" `
    2>&1 | Tee-Object -FilePath $log

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== 训练完成，请拷回开发机 ===" -ForegroundColor Green
Write-Host "  models/kn2_trader_v16.pth"
Write-Host "  models/kn2_trader_v16.meta.json"
Write-Host "  data/training/reports/kn2_v16/train_report.json"
