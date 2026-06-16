# V16 结构特征：断点续跑 + 失败自动重试 + 完成后自动 Horizon→PPO→验收
param(
    [switch]$PrepOnly
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..

$logDir = Join-Path (Get-Location) "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("prepare_v16_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
$attempt = 0

function Write-LogLine([string]$Line) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $msg = "[$ts] $Line"
    Write-Host $msg
    try {
        Add-Content -Path $log -Value $msg -Encoding utf8 -ErrorAction Stop
    } catch {
        Write-Host "WARN: log write failed: $_"
    }
}

while ($true) {
    $attempt++
    Write-LogLine "=== prepare attempt $attempt ==="

    & py -3 -u scripts/prepare_horizon_v16_data.py --symbol XAUUSD --jobs 1 --checkpoint-every 10000
    $rc = $LASTEXITCODE

    if ($rc -eq 0) {
        Write-LogLine "PREPARE DONE"
        break
    }

    Write-LogLine "prepare exited $rc — retry in 15s (checkpoint preserved)"
    Start-Sleep -Seconds 15
}

if ($PrepOnly) { exit 0 }

Write-LogLine "=== AUTO CHAIN: Horizon -> ONNX -> PPO -> acceptance ==="
$chainLog = Join-Path $logDir ("v16_pipeline_chain_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
& py -3 -u scripts/run_v16_pipeline.py --skip-prepare --skip-enrich --jobs 1 --epochs 80 *>&1 |
    ForEach-Object {
        Write-Host $_
        try { Add-Content -Path $chainLog -Value $_ -Encoding utf8 } catch { }
    }

exit $LASTEXITCODE
