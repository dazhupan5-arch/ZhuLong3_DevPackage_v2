# 烛龙验收训练守护进程：独立窗口运行，崩溃自动重启，不占用 Cursor 后台 shell
param(
    [string]$Symbol = 'XAUUSD',
    [string]$Csv = 'data/training/XAUUSD_M5.csv'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$log = Join-Path $root 'data\training\train_daemon.log'
$pidFile = Join-Path $root 'data\training\train.pid'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

# 仅停止本脚本上次启动的训练（读 pid 文件），不误杀其他 python
if (Test-Path $pidFile) {
    $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($oldPid -match '^\d+$') {
        $old = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue
        if ($null -ne $old -and $old.ProcessName -match 'python') {
            Write-Host "Stopping previous train pid=$oldPid"
            Stop-Process -Id ([int]$oldPid) -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }
    }
}

$pyArgs = @(
    '-3', 'scripts/train_until_accepted.py',
    '--symbol', $Symbol,
    '--m5-csv', $Csv,
    '--log-file', 'data/training/train_daemon.log'
)

Write-Host "Log: $log"
Write-Host "Daemon: auto-restart on crash; exit 0 = acceptance passed"

while ($true) {
    $started = Get-Date
    Write-Host "=== train start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    $p = Start-Process -FilePath 'py' -ArgumentList $pyArgs `
        -WorkingDirectory $root -PassThru -NoNewWindow -Wait

    $code = $p.ExitCode
    Write-Host "=== train exit code=$code elapsed=$(((Get-Date) - $started).TotalMinutes.ToString('0.0')) min ==="

    if ($code -eq 0) {
        Write-Host 'ACCEPTANCE PASSED - daemon exit'
        exit 0
    }

    Write-Host 'Training stopped unexpectedly; restart in 15s... (Ctrl+C to cancel daemon)'
    Start-Sleep -Seconds 15
}
