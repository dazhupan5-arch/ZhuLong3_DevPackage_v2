# KN2 V16 验收通过后部署（GPU 机训练完成 + accept_kn2_v16.py PASS 后运行）
param(
    [switch]$EnableLive,
    [switch]$ForceShadow,
    [switch]$SkipAcceptCheck,
    [string]$InstallDir = "C:\Program Files\ZhuLong"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$report = Join-Path $Root "data\training\reports\kn2_v16\acceptance_report.json"
if (-not (Test-Path $report)) {
    if ($SkipAcceptCheck -or $EnableLive) {
        Write-Warning "missing acceptance report - continue deploy (SkipAcceptCheck / GPU passed)"
    } else {
        Write-Error "missing acceptance report: $report - run accept_kn2_v16.py on GPU first"
    }
} else {
    $acc = Get-Content $report -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $acc.passed) {
        if ($SkipAcceptCheck -and $EnableLive) {
            Write-Warning "local acceptance passed=false - GPU passed, continue LIVE (SkipAcceptCheck)"
        } elseif ($ForceShadow -and -not $EnableLive) {
            Write-Warning "KN2 acceptance failed — deploying SHADOW only (-ForceShadow)"
        } else {
            Write-Error "KN2 验收未通过: $($acc.failures -join ', '). Use -ForceShadow or -SkipAcceptCheck -EnableLive."
        }
    } else {
        Write-Host "KN2 acceptance PASSED" -ForegroundColor Green
    }
}

function Resolve-Kn2Src([string]$Rel) {
    $candidates = @(
        (Join-Path $Root $Rel),
        (Join-Path $Root ($Rel -replace "^models\\", "data\"))
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$appData = Join-Path $env:APPDATA "ZhuLong"
$kn2Files = @("models\kn2_trader_v16.pth", "models\kn2_trader_v16.meta.json")
foreach ($rel in $kn2Files) {
    $src = Resolve-Kn2Src $rel
    if (-not $src) { Write-Error "missing $rel (models/ or data/)" }
    foreach ($base in @($appData, $InstallDir)) {
        if (-not (Test-Path $base)) { continue }
        $dst = Join-Path $base $rel
        $dir = Split-Path $dst -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        try {
            Copy-Item -Force $src $dst -ErrorAction Stop
            Write-Host "OK $rel -> $base" -ForegroundColor Green
        } catch {
            Write-Warning "skip $base (need admin?): $rel"
        }
    }
}

$pyPatches = @(
    "zhulong\agent\knowledge_net_kn2.py",
    "zhulong\agent\trading_agent.py",
    "zhulong\agent\kn2_location_labels.py"
)
foreach ($rel in $pyPatches) {
    $src = Join-Path $Root $rel
    if (-not (Test-Path $src)) { continue }
    foreach ($base in @($appData, $InstallDir)) {
        if (-not (Test-Path $base)) { continue }
        $dst = Join-Path $base $rel
        $dir = Split-Path $dst -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        try {
            Copy-Item -Force $src $dst -ErrorAction Stop
            Write-Host "OK patch $rel -> $base" -ForegroundColor Yellow
        } catch {
            Write-Warning "skip patch $rel -> $base"
        }
    }
}

$cfgPath = Join-Path $appData "config_agent.json"
$cfg = Get-Content $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $cfg.kn2) { $cfg | Add-Member -NotePropertyName kn2 -NotePropertyValue @{} }
$cfg.kn2.enabled = [bool]$EnableLive
$cfg.kn2.shadow_mode = -not [bool]$EnableLive
$cfg.kn2.model_path = "models/kn2_trader_v16.pth"
$cfg.kn2.min_confidence = 0.48
[System.IO.File]::WriteAllText($cfgPath, ($cfg | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))

function Set-InstallKn2Live([string]$InstallConfigPath) {
    if (-not (Test-Path $InstallConfigPath)) { return }
    try {
        $ic = Get-Content $InstallConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $ic.kn2) { $ic | Add-Member -NotePropertyName kn2 -NotePropertyValue @{} }
        $ic.kn2.enabled = [bool]$EnableLive
        $ic.kn2.shadow_mode = -not [bool]$EnableLive
        $ic.kn2.model_path = "models/kn2_trader_v16.pth"
        $ic.kn2.min_confidence = 0.48
        [System.IO.File]::WriteAllText($InstallConfigPath, ($ic | ConvertTo-Json -Depth 12), [System.Text.UTF8Encoding]::new($false))
        Write-Host "OK install config kn2.enabled=$($ic.kn2.enabled) -> $InstallConfigPath" -ForegroundColor Green
    } catch {
        Write-Warning "cannot write install config (need admin): $InstallConfigPath"
    }
}

Set-InstallKn2Live (Join-Path $InstallDir "config\config_agent.json")
Set-InstallKn2Live (Join-Path $Root "config\config_agent.json")

$mode = if ($EnableLive) { "LIVE" } else { "shadow" }
Write-Host "KN2 deployed ($mode). kn2.enabled=$($cfg.kn2.enabled) shadow=$($cfg.kn2.shadow_mode)" -ForegroundColor Green
Write-Host "Restart ZhuLong.exe"

if (Test-Path $InstallDir) {
    Write-Host "Tip: run deploy_v16_models_admin.ps1 as admin to sync Program Files"
}
