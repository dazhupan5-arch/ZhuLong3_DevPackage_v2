#Requires -RunAsAdministrator
# 管理员：部署 V16 闭合修复（C# + Python 引擎 + 模型）到 Program Files
param(
    [string]$InstallDir = "C:\Program Files\ZhuLong",
    [string]$DevRoot = "D:\trae_projects\ZhuLong3_DevPackage_v2"
)

$ErrorActionPreference = "Stop"
$appData = Join-Path $env:APPDATA "ZhuLong"
$binDir = Join-Path $DevRoot "src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64"

Write-Host "=== V16 Live Closure (Admin) ===" -ForegroundColor Cyan

if (-not (Test-Path $binDir)) {
    Write-Error "Build missing: dotnet build src\ZhuLong.App\ZhuLong.App.csproj -c Release -p:Platform=x64"
}

$exeFiles = @("ZhuLong.exe", "ZhuLong.dll", "ZhuLong.Core.dll", "ZhuLong.deps.json", "ZhuLong.runtimeconfig.json")
foreach ($f in $exeFiles) {
    $src = Join-Path $binDir $f
    if (Test-Path $src) {
        Copy-Item -Force $src (Join-Path $InstallDir $f)
        Write-Host "OK $f" -ForegroundColor Green
    }
}

$pyFiles = @(
    "ZhuLong.PythonEngine\inference_cli.py",
    "ZhuLong.PythonEngine\mt5_ops.py",
    "zhulong\agent\trading_agent.py",
    "zhulong\agent\horizon_predictor.py",
    "zhulong\agent\knowledge_net_kn2.py",
    "zhulong\agent\kn2_location_labels.py",
    "zhulong\agent\knowledge_net.py",
    "zhulong\agent\tick_brief.py",
    "zhulong\agent\cognition.py",
    "zhulong\agent\trader_mind.py",
    "zhulong\agent\structure_service.py",
    "zhulong\utils\paths.py"
)

foreach ($rel in $pyFiles) {
    $src = Join-Path $DevRoot $rel
    if (-not (Test-Path $src)) { Write-Warning "skip $rel"; continue }
    foreach ($base in @($InstallDir, $appData)) {
        $dst = Join-Path $base $rel
        $dir = Split-Path $dst -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        Copy-Item -Force $src $dst
        Write-Host "OK $rel -> $base" -ForegroundColor Green
    }
}

$installCfg = Join-Path $DevRoot "config\config_agent.json"
if (Test-Path $installCfg) {
    $cfgDst = Join-Path $InstallDir "config\config_agent.json"
    Copy-Item -Force $installCfg $cfgDst
    Write-Host "OK install config_agent.json (KN2 LIVE)" -ForegroundColor Green
}

& (Join-Path $DevRoot "scripts\deploy_kn2_v16_when_ready.ps1") -EnableLive -SkipAcceptCheck -InstallDir $InstallDir

Get-Process ZhuLong -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -eq "" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-Process (Join-Path $InstallDir "ZhuLong.exe") -WorkingDirectory $InstallDir
Write-Host "Restarted ZhuLong.exe" -ForegroundColor Green
