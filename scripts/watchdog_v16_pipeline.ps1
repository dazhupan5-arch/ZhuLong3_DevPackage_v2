# V16 管线 watchdog：每 2 分钟检查进度 + 进程，断了自动续跑并写状态
# 用法: .\scripts\watchdog_v16_pipeline.ps1
# 建议单独开一个 PowerShell 窗口长期运行，或后台启动一次即可
param(
    [int]$IntervalSec = 120,
    [int]$StaleMinutes = 25
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..

$logDir = Join-Path (Get-Location) "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$watchLog = Join-Path $logDir "v16_watchdog.log"
$lockFile = Join-Path $logDir "v16_pipeline.lock"
$progressFile = Join-Path (Get-Location) "data\training\v16\XAUUSD\struct_progress.json"

function Write-Watch([string]$Msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $line
    try {
        Add-Content -Path $watchLog -Value $line -Encoding utf8
    } catch { }
}

function Test-PipelineRunning {
    $needles = @("prepare_horizon_v16_data", "run_v16_pipeline", "train_horizon_v16", "train_rl_v16", "train_rl_agent")
    foreach ($p in (Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" -ErrorAction SilentlyContinue)) {
        $cmd = $p.CommandLine
        if (-not $cmd) { continue }
        foreach ($n in $needles) {
            if ($cmd -like "*$n*") { return $true }
        }
    }
    return $false
}

function Get-ProgressDone {
    if (-not (Test-Path $progressFile)) { return 0, 0 }
    try {
        $j = Get-Content $progressFile -Raw -Encoding UTF8 | ConvertFrom-Json
        return [int]$j.done, [int]$j.total
    } catch {
        return 0, 0
    }
}

function Test-PrepComplete {
    $npz = Join-Path (Get-Location) "data\training_horizon_v16.npz"
    if (-not (Test-Path $npz)) { return $false }
    try {
        $rows = & py -3 -c "import numpy as np; print(len(np.load(r'$npz')['struct']))" 2>$null
        return [int]$rows -ge 700000
    } catch {
        return $false
    }
}

function Start-Pipeline {
    if (Test-Path $lockFile) {
        $lockAge = (Get-Date) - (Get-Item $lockFile).LastWriteTime
        if ($lockAge.TotalMinutes -lt 3) {
            Write-Watch "skip start: lock fresh ($([int]$lockAge.TotalSeconds)s)"
            return
        }
    }
    Set-Content -Path $lockFile -Value (Get-Date -Format o) -Encoding utf8
    Write-Watch "START run_v16_full_auto.ps1 (resume from checkpoint)"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "powershell.exe"
    $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\run_v16_full_auto.ps1`""
    $psi.WorkingDirectory = (Get-Location).Path
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $false
    [void][System.Diagnostics.Process]::Start($psi)
}

Write-Watch "watchdog started interval=${IntervalSec}s stale=${StaleMinutes}m"

$lastDone = -1
$lastProgressAt = Get-Date

while ($true) {
    $running = Test-PipelineRunning
    $done, $total = Get-ProgressDone
    $pct = if ($total -gt 0) { [math]::Round(100.0 * $done / $total, 1) } else { 0 }

    if ($done -ne $lastDone) {
        $lastDone = $done
        $lastProgressAt = Get-Date
        Write-Watch "progress $done / $total ($pct%) running=$running"
    }

    & py -3 scripts/v16_write_status.py $(if ($running) { "--running" }) --note "watchdog_ok" | Out-Null

    $prepDone = Test-PrepComplete
    $acceptance = Join-Path (Get-Location) "data\training\reports\v16\acceptance_report.json"
    if ((Test-Path $acceptance) -and $prepDone) {
        try {
            $rep = Get-Content $acceptance -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($rep.passed -eq $true) {
                Write-Watch "acceptance PASSED — watchdog exit"
                break
            }
            if (-not $running) {
                Write-Watch "acceptance FAILED — watchdog exit (manual fix needed)"
                break
            }
        } catch { }
    }

    $staleMin = ((Get-Date) - $lastProgressAt).TotalMinutes
    if ($running -and $staleMin -ge $StaleMinutes -and -not $prepDone) {
        Write-Watch "WARN: stale ${staleMin}m at $done/$total — process may be hung"
    }

    if (-not $running) {
        if (-not $prepDone -and ($total -eq 0 -or $done -lt $total)) {
            Write-Watch "ALERT: pipeline not running, prep incomplete ($done/$total) — restarting"
            Start-Pipeline
        } elseif ($prepDone -and -not (Test-Path $acceptance)) {
            Write-Watch "ALERT: prep done but no acceptance — restarting chain from train"
            Start-Pipeline
        } else {
            Write-Watch "idle ok running=$running prepDone=$prepDone pct=$pct"
        }
    }

    Start-Sleep -Seconds $IntervalSec
}
