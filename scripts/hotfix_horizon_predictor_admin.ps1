# 一键热修复：Horizon V16 推理补丁 → Program Files（需管理员）
#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"
$DevRoot = Split-Path $PSScriptRoot -Parent
$InstallDir = "C:\Program Files\ZhuLong"
$patch = Join-Path $DevRoot "zhulong\agent\horizon_predictor.py"
$dst = Join-Path $InstallDir "zhulong\agent\horizon_predictor.py"
Copy-Item -Force $patch $dst
Write-Host "Patched $dst" -ForegroundColor Green
Get-Process ZhuLong -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Start-Process (Join-Path $InstallDir "ZhuLong.exe")
Write-Host "ZhuLong restarted." -ForegroundColor Green
