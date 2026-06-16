# 监控 RL 智能体训练进度，写入 logs/training/watch_status.txt
param(
    [int]$IntervalSec = 300,
    [int]$TargetSteps = 2000000
)
$root = Join-Path $PSScriptRoot '..'
Set-Location $root
$out = Join-Path $root 'logs/training/watch_status.txt'
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null

function Get-Step([string]$sym) {
    $p = Join-Path $root "logs/rl/$sym/evaluations.npz"
    if (-not (Test-Path $p)) { return $null }
    $code = @"
import numpy as np
d=np.load(r'$p')
print(int(d['timesteps'][-1]), float(d['results'][-1].mean()), len(d['timesteps']))
"@
    $r = py -3 -c $code 2>$null
    if (-not $r) { return $null }
    $parts = $r.Trim() -split '\s+'
    return @{ step = [int]$parts[0]; reward = [double]$parts[1]; evals = [int]$parts[2] }
}

function Get-Procs {
    Get-Process python* -ErrorAction SilentlyContinue | ForEach-Object {
        $c = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
        if ($c -match 'train_rl_agent|prepare_training_data|backtest_rl') {
            [PSCustomObject]@{
                Id = $_.Id
                Hours = [math]::Round(((Get-Date) - $_.StartTime).TotalHours, 2)
                CPU = [math]::Round($_.CPU, 0)
                Cmd = if ($c -match 'train_rl_agent\.py --symbol (\w+)') { "RL $($Matches[1])" }
                      elseif ($c -match 'prepare_training_data\.py --symbol (\w+)') { "PREPARE $($Matches[1])" }
                      elseif ($c -match 'backtest_rl') { 'BACKTEST' }
                      else { 'OTHER' }
            }
        }
    }
}

while ($true) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $lines = @("[$ts] === Agent Training Watch ===")
    foreach ($sym in @('xauusd', 'usoil')) {
        $s = Get-Step $sym
        if ($s) {
            $pct = [math]::Round(100 * $s.step / $TargetSteps, 2)
            $lines += "RL $($sym.ToUpper()) : $($s.step)/$TargetSteps ($pct%) reward=$([math]::Round($s.reward,2)) evals=$($s.evals)"
        } else {
            $lines += "RL $($sym.ToUpper()) : waiting"
        }
    }
    $oilZip = Test-Path (Join-Path $root 'models/rl_agent_oil.zip')
    $xauZip = Test-Path (Join-Path $root 'models/rl_agent_xau.zip')
    $lines += "models: rl_agent_xau=$xauZip rl_agent_oil=$oilZip"
    $procs = @(Get-Procs)
    if ($procs.Count -eq 0) {
        $lines += 'processes: NONE (training may have finished or stopped)'
    } else {
        foreach ($p in $procs) { $lines += "proc PID=$($p.Id) $($p.Cmd) $($p.Hours)h CPU=$($p.CPU)" }
    }
    $text = $lines -join "`n"
    Set-Content -Path $out -Value $text -Encoding utf8
    Add-Content -Path (Join-Path $root 'logs/training/watch_history.log') -Value $text -Encoding utf8
    Add-Content -Path (Join-Path $root 'logs/training/watch_history.log') -Value '---' -Encoding utf8

    $rlRunning = ($procs | Where-Object { $_.Cmd -like 'RL *' }).Count -gt 0
    $prepRunning = ($procs | Where-Object { $_.Cmd -like 'PREPARE *' }).Count -gt 0
    if (-not $rlRunning -and -not $prepRunning) {
        Add-Content -Path (Join-Path $root 'logs/training/watch_history.log') -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL TRAINING PROCESSES STOPPED" -Encoding utf8
        break
    }
    Start-Sleep -Seconds $IntervalSec
}
