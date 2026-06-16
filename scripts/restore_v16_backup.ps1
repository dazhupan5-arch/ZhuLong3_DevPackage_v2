param(
    [Parameter(Mandatory = $true)]
    [string]$Tag
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Src = Join-Path $Root "models\backups\$Tag"

if (-not (Test-Path $Src)) {
    Write-Error "Backup not found: $Src"
}

$manifest = Join-Path $Src "BACKUP_MANIFEST.json"
if (Test-Path $manifest) {
    Write-Host "Restoring backup: $Tag"
    Get-Content $manifest -Raw | ConvertFrom-Json | Select-Object backup_id, purpose, acceptance | Format-List
}

$map = @(
    @{ from = "models\horizon_v16.onnx"; to = "models\horizon_v16.onnx" },
    @{ from = "models\horizon_v16.pth"; to = "models\horizon_v16.pth" },
    @{ from = "models\horizon_v16_scaler.pkl"; to = "models\horizon_v16_scaler.pkl" },
    @{ from = "models\horizon_v16.meta.json"; to = "models\horizon_v16.meta.json" },
    @{ from = "models\rl_agent_xau.zip"; to = "models\rl_agent_xau.zip" },
    @{ from = "models\XAUUSD\v16\rl_meta.json"; to = "models\XAUUSD\v16\rl_meta.json" },
    @{ from = "data\training_horizon_v16.npz"; to = "data\training_horizon_v16.npz" },
    @{ from = "data\agent_state_scaler_xauusd.json"; to = "data\agent_state_scaler_xauusd.json" },
    @{ from = "reports\acceptance_report.json"; to = "data\training\reports\v16\acceptance_report.json" },
    @{ from = "reports\backtest_summary.json"; to = "data\training\reports\v16\backtest_summary.json" }
)

foreach ($item in $map) {
    $from = Join-Path $Src $item.from
    $to = Join-Path $Root $item.to
    if (-not (Test-Path $from)) {
        Write-Warning "Skip missing: $($item.from)"
        continue
    }
    $dir = Split-Path $to -Parent
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    Copy-Item -Force $from $to
    Write-Host "Restored $($item.to)"
}

Write-Host "Done. Re-run: py -3 scripts/accept_horizon_v16.py"
