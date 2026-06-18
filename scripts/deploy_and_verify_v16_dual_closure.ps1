# V16 full deploy + dual-closure verify (required before pack-installer)
param(
    [string]$InstallDir = "C:\Program Files\ZhuLong",
    [switch]$SkipAdmin,
    [switch]$AllowPackOnPass
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$appData = Join-Path $env:APPDATA "ZhuLong"
$binDir = Join-Path $Root "src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64"

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Write-Step "Pre-check: deploy gate + acceptance"
py -3 scripts/pre_deploy_v16_gate.py --require-kn2-live
if ($LASTEXITCODE -ne 0) { throw "pre_deploy_v16_gate FAILED" }
$metaPath = Join-Path $Root "models\horizon_v16.meta.json"
if (-not (Test-Path $metaPath)) { throw "Missing models/horizon_v16.meta.json" }
$meta = Get-Content $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $meta.passed) { throw "Horizon meta passed=false - abort deploy" }
$kn2Report = Join-Path $Root "data\training\reports\kn2_v16\acceptance_report.json"
if (-not (Test-Path $kn2Report)) { throw "Missing KN2 acceptance report" }
$kn2Acc = Get-Content $kn2Report -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $kn2Acc.passed) { throw "KN2 acceptance failed - abort deploy" }
foreach ($rel in @(
        "models\horizon_v16.onnx", "models\horizon_v16_scaler.pkl", "models\horizon_v16.pth",
        "models\kn2_trader_v16.pth", "models\rl_agent_xau.zip"
    )) {
    if (-not (Test-Path (Join-Path $Root $rel))) { throw "Missing artifact: $rel" }
}
Write-Host "Horizon trial=$($meta.trial) macro_f1=$($meta.macro_f1) passed=$($meta.passed)" -ForegroundColor Green

Write-Step "Build Release x64"
dotnet build src/ZhuLong.App/ZhuLong.App.csproj -c Release -p:Platform=x64 -p:GenerateAppIcons=false
if (-not $?) { exit 1 }

Write-Step "Sync models + config to AppData"
New-Item -ItemType Directory -Force -Path (Join-Path $appData "models\XAUUSD\v16") | Out-Null
$modelFiles = @(
    "models\horizon_v16.onnx", "models\horizon_v16_scaler.pkl", "models\horizon_v16.meta.json", "models\horizon_v16.pth",
    "models\kn2_trader_v16.pth", "models\kn2_trader_v16.meta.json", "models\rl_agent_xau.zip",
    "models\XAUUSD\v16\rl_meta.json", "data\agent_state_scaler_xauusd.json"
)
foreach ($rel in $modelFiles) {
    $src = Join-Path $Root $rel
    if (-not (Test-Path $src)) { $src = Join-Path $appData $rel }
    if (-not (Test-Path $src)) { Write-Warning "skip $rel"; continue }
    $dst = Join-Path $appData $rel
    $dir = Split-Path $dst -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Copy-Item -Force $src $dst
    Write-Host "AppData OK $rel"
}

$cfgSrc = Join-Path $Root "config\config_agent.json"
$cfgDst = Join-Path $appData "config_agent.json"
Copy-Item -Force $cfgSrc $cfgDst
Write-Host "AppData config_agent.json (V16 + KN2 LIVE)"

$macroCsvSrc = Join-Path $Root "data\macro_events.csv"
$macroCsvDst = Join-Path $appData "data\macro_events.csv"
if (Test-Path $macroCsvSrc) {
    New-Item -ItemType Directory -Force -Path (Split-Path $macroCsvDst -Parent) | Out-Null
    Copy-Item -Force $macroCsvSrc $macroCsvDst
    Write-Host "AppData OK data\macro_events.csv"
}

$cfgAppSrc = Join-Path $Root "config.json"
$cfgAppDst = Join-Path $appData "config.json"
if (Test-Path $cfgAppSrc) {
    Copy-Item -Force $cfgAppSrc $cfgAppDst
    Write-Host "AppData OK config.json (macro force_silence + V16 flags)"
}

$pyHot = @(
    "ZhuLong.PythonEngine\inference_cli.py", "ZhuLong.PythonEngine\inference_worker.py", "ZhuLong.PythonEngine\mt5_ops.py",
    "zhulong\utils\json_safe.py", "zhulong\engine\agent_engine.py",
    "zhulong\agent\horizon_predictor.py", "zhulong\agent\trading_agent.py",
    "zhulong\agent\execution_composer.py",
    "zhulong\agent\knowledge_net_kn2.py", "zhulong\agent\kn2_location_labels.py",
    "zhulong\agent\knowledge_net.py", "zhulong\agent\tick_brief.py",
    "zhulong\agent\cognition.py", "zhulong\agent\trader_mind.py",
    "zhulong\agent\structure_service.py", "zhulong\utils\paths.py"
)
foreach ($rel in $pyHot) {
    $src = Join-Path $Root $rel
    if (-not (Test-Path $src)) { continue }
    $dst = Join-Path $appData $rel
    $dir = Split-Path $dst -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Copy-Item -Force $src $dst
}

Write-Step "Overlay build dir"
foreach ($item in @("config", "models", "data", "zhulong", "ZhuLong.PythonEngine", "config.json")) {
    $src = Join-Path $Root $item
    $dst = Join-Path $binDir $item
    if (-not (Test-Path $src)) { continue }
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Copy-Item -Recurse -Force $src $dst
}
Copy-Item -Force (Join-Path $Root "config\config_agent.json") (Join-Path $binDir "config\config_agent.json")

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $SkipAdmin -and $isAdmin -and (Test-Path $InstallDir)) {
    Write-Step "Admin sync to Program Files"
    & (Join-Path $Root "scripts\fix_v16_live_closure_admin.ps1") -InstallDir $InstallDir -DevRoot $Root
} elseif (-not $SkipAdmin -and (Test-Path $InstallDir)) {
    Write-Warning "Not admin - skip Program Files; run fix_v16_live_closure_admin.ps1 as Administrator"
    try {
        foreach ($f in @("ZhuLong.exe", "ZhuLong.dll", "ZhuLong.Core.dll")) {
            Copy-Item -Force (Join-Path $binDir $f) (Join-Path $InstallDir $f) -ErrorAction Stop
        }
    } catch {
        Write-Warning "Could not update Program Files exe - live test uses build dir"
    }
}

Write-Step "Engineering closure (ZhuLong stopped)"
Get-Process ZhuLong -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
$engOk = $true
py -3 scripts\kn2_v16_live_boot.py
if ($LASTEXITCODE -ne 0) { $engOk = $false }
py -3 scripts\verify_v16_full_stack.py
if ($LASTEXITCODE -ne 0) { $engOk = $false }

Write-Step "Restart ZhuLong for live test"
$startExe = Join-Path $binDir "ZhuLong.exe"
Start-Process -FilePath $startExe -WorkingDirectory (Split-Path $startExe -Parent)
Write-Host "Started $startExe"
Start-Sleep -Seconds 30

py -3 scripts\horizon_v16_post_restart_check.py
if ($LASTEXITCODE -ne 0) { $engOk = $false }

Write-Step "Live scenario closure (log audit, poll up to 5 min)"
$liveOk = $false
for ($i = 0; $i -lt 20; $i++) {
    py -3 scripts\audit_v16_live_log.py
    if ($LASTEXITCODE -eq 0) {
        $liveOk = $true
        break
    }
    Write-Host "  waiting for agent tick in log ($($i + 1)/20)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 15
}
if ($liveOk) { Write-Host "[PASS] Live log closure" -ForegroundColor Green }
else { Write-Host "[FAIL] Live log closure" -ForegroundColor Red }

Write-Step "Summary"
Write-Host "Engineering: $(if ($engOk) { 'PASS' } else { 'FAIL' })"
Write-Host "Live:        $(if ($liveOk) { 'PASS' } else { 'FAIL' })"

if (-not ($engOk -and $liveOk)) {
    Write-Host "`nDUAL CLOSURE: FAIL - pack-installer BLOCKED" -ForegroundColor Red
    exit 2
}

Write-Host "`nDUAL CLOSURE: PASS - ready to pack" -ForegroundColor Green
if ($AllowPackOnPass) {
    & (Join-Path $Root "scripts\pack-installer.ps1") -ForcePack
    exit $LASTEXITCODE
}
exit 0
