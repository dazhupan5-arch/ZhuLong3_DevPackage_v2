# 部署 M1 时区修复（需管理员：覆盖 C:\Program Files\ZhuLong）
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path $PSScriptRoot -Parent
$Src = Join-Path $RepoRoot '_publish_m1fix'
$Dst = 'C:\Program Files\ZhuLong'

if (-not (Test-Path $Src)) {
    Write-Host "请先运行: dotnet publish src/ZhuLong.App/ZhuLong.App.csproj -c Release -r win-x64 -o _publish_m1fix" -ForegroundColor Yellow
    exit 1
}

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host '需要管理员权限，正在提权...' -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit 0
}

Get-Process ZhuLong -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

Copy-Item (Join-Path $RepoRoot 'ZhuLong.PythonEngine\mt5_ops.py') (Join-Path $Dst 'ZhuLong.PythonEngine\mt5_ops.py') -Force
Copy-Item (Join-Path $Src 'ZhuLong.Core.dll') (Join-Path $Dst 'ZhuLong.Core.dll') -Force
Copy-Item (Join-Path $Src 'ZhuLong.dll') (Join-Path $Dst 'ZhuLong.dll') -Force
Copy-Item (Join-Path $RepoRoot 'indicators\ZhuLongIndicator.mq5') (Join-Path $Dst 'indicators\ZhuLongIndicator.mq5') -Force
Copy-Item (Join-Path $RepoRoot 'mql5\ZhuLongIndicator.mq5') (Join-Path $Dst 'mql5\ZhuLongIndicator.mq5') -Force

Write-Host 'M1 修复已部署到' $Dst -ForegroundColor Green
Write-Host '请在 MT5 MetaEditor 中重新编译 ZhuLongIndicator.mq5 并重新挂载到 XAUUSD M1（及 USOIL M1 如需双品种实时）。' -ForegroundColor Cyan
Write-Host '然后启动 ZhuLong.exe → 连接 MT5 → 开始运行。' -ForegroundColor Cyan
