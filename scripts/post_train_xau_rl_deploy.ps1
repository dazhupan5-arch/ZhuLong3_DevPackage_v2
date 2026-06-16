# Wait for XAU PPO training -> backtest -> deploy if PASS
$ErrorActionPreference = "Stop"
$root = "D:\ZhuLong3_Migration_20260609.zip"
$deploy = "d:\Program Files\ZhuLong"
$logDir = Join-Path $root "logs\training"

Write-Host "=== Waiting for XAU RL training to finish ==="
while ($true) {
    $running = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*train_rl_agent.py*XAUUSD*' }
    if (-not $running) { break }
    Start-Sleep -Seconds 30
}
Write-Host "Training process ended"

$waitLog = Get-ChildItem $logDir -Filter "xau_rl_retrain_*.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($waitLog) {
    Write-Host "Training log tail:"
    Get-Content $waitLog.FullName -Tail 5 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
}

Set-Location $root
Write-Host "=== Backtest validation 2025 OOS ==="
py -3 scripts/backtest_rl.py --symbol XAUUSD
$btCode = $LASTEXITCODE
if ($btCode -ne 0) {
    Write-Host "Backtest FAILED exit=$btCode - skip PPO deploy" -ForegroundColor Yellow
    exit $btCode
}

Write-Host "=== Backtest PASS - deploying RL model ==="
$models = Join-Path $deploy "models"
$dataDst = Join-Path $deploy "data"
New-Item -ItemType Directory -Force -Path $models, $dataDst | Out-Null

Copy-Item -Force "$root\models\rl_agent_xau.zip" "$models\rl_agent_xau.zip"
if (Test-Path "$root\data\agent_state_scaler_xauusd.json") {
    Copy-Item -Force "$root\data\agent_state_scaler_xauusd.json" "$dataDst\agent_state_scaler_xauusd.json"
}
Copy-Item -Recurse -Force "$root\zhulong\agent" "$deploy\zhulong\"

$p = Get-Process -Name ZhuLong -ErrorAction SilentlyContinue
if ($p) { $p | Stop-Process -Force; Start-Sleep 2 }

Write-Host "=== Restarting ZhuLong ==="
Start-Process -FilePath "$deploy\ZhuLong.exe" -WorkingDirectory $deploy
Write-Host "=== XAU PPO deploy complete ==="
