# 在新窗口启动训练守护进程（与 Cursor 后台 shell 脱离，避免中途被 kill）
$root = Split-Path $PSScriptRoot -Parent
$daemon = Join-Path $root 'scripts\run_train_daemon.ps1'
Start-Process powershell -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-NoExit',
    '-File', $daemon
) -WorkingDirectory $root
Write-Host 'Train daemon started in new PowerShell window.'
Write-Host 'Log: data\training\train_daemon.log'
Write-Host 'PID file: data\training\train.pid'
