# 注册 Windows 计划任务：每日 02:00 备份 trading.db
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$backup = Join-Path $root "scripts\backup_trading_db.ps1"
$taskName = "ZhuLong_TradingDbBackup"

if (-not (Test-Path $backup)) {
    Write-Error "找不到 $backup"
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$backup`""

$trigger = New-ScheduledTaskTrigger -Daily -At "02:00"

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Description "烛龙 trading.db 每日备份" -Force | Out-Null

Write-Host "已注册计划任务: $taskName (每日 02:00)" -ForegroundColor Green
Write-Host "手动运行: Start-ScheduledTask -TaskName $taskName"
