#Requires -Version 5.1
<#
.SYNOPSIS
  以独立进程启动 USOIL 智能体训练（不依赖 Cursor 终端）。
  启动前会结束其它 train_usoil / prepare_training_data(USOIL) 残留进程。
#>
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$logDir = Join-Path $root 'logs\training'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'usoil_until_pass.log'
$runLog = Join-Path $logDir ('usoil_run_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$pidFile = Join-Path $logDir 'usoil_train.pid'

function Get-TrainingProcessIds {
    $ids = @()
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | ForEach-Object {
        $cl = $_.CommandLine
        if ($null -eq $cl) { return }
        if ($cl -match 'train_usoil_agent_until_pass\.py' -or
            ($cl -match 'prepare_training_data\.py' -and $cl -match 'USOIL')) {
            $ids += [int]$_.ProcessId
        }
    }
    return $ids | Sort-Object -Unique
}

function Stop-TrainingProcesses([int[]] $ExceptIds) {
    $except = @{}
    foreach ($id in $ExceptIds) { $except[$id] = $true }
    foreach ($id in (Get-TrainingProcessIds)) {
        if ($except.ContainsKey($id)) { continue }
        Write-Host "结束残留训练进程 PID=$id"
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
}

$keepPid = 0
if (Test-Path $pidFile) {
    $keepPid = [int](Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$aliveKeep = $false
if ($keepPid -gt 0) {
    $procs = Get-TrainingProcessIds
    $hasWorker = $procs | Where-Object { $_ -ne $keepPid }
    $launcher = Get-Process -Id $keepPid -ErrorAction SilentlyContinue
    if ($launcher -and $hasWorker) {
        $aliveKeep = $true
        Write-Host "训练已在运行 launcher PID=$keepPid（worker 数=$($hasWorker.Count)），清理其它残留..."
        Stop-TrainingProcesses -ExceptIds @($keepPid) + [int[]]$hasWorker
        Write-Host "日志: $log"
        exit 0
    }
    if ($launcher -and -not $hasWorker) {
        Write-Host "launcher PID=$keepPid 无 worker，将重启训练"
        Stop-Process -Id $keepPid -Force -ErrorAction SilentlyContinue
    }
}

Stop-TrainingProcesses -ExceptIds @()
Start-Sleep -Seconds 2

$py = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }

$cmd = "chcp 65001>nul&& set PYTHONUNBUFFERED=1&& `"$py`" -3 -u scripts/train_usoil_agent_until_pass.py > `"$runLog`" 2>&1"
$p = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/c', $cmd `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -PassThru

$p.Id | Set-Content -Path $pidFile -Encoding ASCII
$runLog | Set-Content -Path (Join-Path $logDir 'usoil_current_run.log') -Encoding UTF8
Write-Host "USOIL 训练已启动 PID=$($p.Id)"
Write-Host "日志: $runLog"
Write-Host "查看: Get-Content '$runLog' -Wait -Tail 30"
