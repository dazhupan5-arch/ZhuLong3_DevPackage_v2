# 监控 USOIL struct 进度，完成后 enrich + location + git push
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$progress = Join-Path $Root "data\training\v16\USOIL\struct_progress.json"
$npz = Join-Path $Root "data\clean\training_horizon_v16_usoil.npz"
$log = Join-Path $Root "logs\usoil_v16_npz_watch.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Log([string]$Msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Msg"
    Add-Content -Path $log -Value $line
    Write-Host $line
}

Log "watch started"

while ($true) {
    if (Test-Path $npz) {
        $age = (Get-Date) - (Get-Item $npz).LastWriteTime
        if ($age.TotalSeconds -lt 120) {
            Log "NPZ ready: $npz"
            break
        }
    }
    if (Test-Path $progress) {
        $p = Get-Content $progress -Raw | ConvertFrom-Json
        $pct = [math]::Round(100.0 * $p.done / $p.total, 1)
        Log "struct progress $($p.done)/$($p.total) ($pct%)"
    }
    Start-Sleep -Seconds 300
}

Log "running finish_usoil_v16_npz_push.ps1"
& (Join-Path $Root "scripts\finish_usoil_v16_npz_push.ps1") 2>&1 | ForEach-Object { Log $_ }
Log "watch complete"
