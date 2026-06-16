# 管理员部署：V16 模型 + Python 补丁 → Program Files（实机子进程只读安装目录）
#Requires -RunAsAdministrator
param(
    [string]$InstallDir = "C:\Program Files\ZhuLong",
    [string]$DevRoot = ""
)

$ErrorActionPreference = "Stop"
if (-not $DevRoot) {
    $DevRoot = Split-Path $PSScriptRoot -Parent
}
$appData = Join-Path $env:APPDATA "ZhuLong"

Write-Host "=== V16 Admin Deploy ===" -ForegroundColor Cyan
Write-Host "Install: $InstallDir"
Write-Host "AppData: $appData"
Write-Host "Dev:     $DevRoot"

$modelFiles = @(
    "models\horizon_v16.onnx",
    "models\horizon_v16_scaler.pkl",
    "models\horizon_v16.meta.json",
    "models\horizon_v16.pth",
    "models\rl_agent_xau.zip",
    "models\XAUUSD\v16\rl_meta.json",
    "data\agent_state_scaler_xauusd.json"
)

foreach ($rel in $modelFiles) {
    $src = Join-Path $appData $rel
    if (-not (Test-Path $src)) { $src = Join-Path $DevRoot $rel }
    if (-not (Test-Path $src)) { Write-Warning "skip missing $rel"; continue }
    $dst = Join-Path $InstallDir $rel
    $dir = Split-Path $dst -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Copy-Item -Force $src $dst
    Write-Host "  OK $rel"
}

$pyPatches = @(
    @{ Src = "zhulong\agent\rl_agent.py"; Dst = "zhulong\agent\rl_agent.py" },
    @{ Src = "zhulong\agent\horizon_predictor.py"; Dst = "zhulong\agent\horizon_predictor.py" },
    @{ Src = "zhulong\agent\trading_agent.py"; Dst = "zhulong\agent\trading_agent.py" },
    @{ Src = "zhulong\agent\tick_brief.py"; Dst = "zhulong\agent\tick_brief.py" },
    @{ Src = "zhulong\agent\structure_service.py"; Dst = "zhulong\agent\structure_service.py" },
    @{ Src = "zhulong\agent\trader_mind.py"; Dst = "zhulong\agent\trader_mind.py" },
    @{ Src = "zhulong\agent\cognition.py"; Dst = "zhulong\agent\cognition.py" },
    @{ Src = "zhulong\utils\paths.py"; Dst = "zhulong\utils\paths.py" },
    @{ Src = "ZhuLong.PythonEngine\inference_cli.py"; Dst = "ZhuLong.PythonEngine\inference_cli.py" }
)

foreach ($p in $pyPatches) {
    $src = Join-Path $DevRoot $p.Src
    if (-not (Test-Path $src)) { Write-Warning "skip patch $($p.Src)"; continue }
    $dst = Join-Path $InstallDir $p.Dst
    $dir = Split-Path $dst -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Copy-Item -Force $src $dst
    Write-Host "  PATCH $($p.Dst)"
}

Write-Host "`nDone. Restart ZhuLong.exe" -ForegroundColor Green
