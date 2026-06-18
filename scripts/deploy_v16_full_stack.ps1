# V16 全栈部署：Horizon + 执行门控 + KN2 shadow + AppData + 可选 Admin 同步 + 构建 UI
param(
    [switch]$SkipBuild,
    [switch]$SkipRestart,
    [switch]$ForceKn2Shadow,
    [string]$InstallDir = "C:\Program Files\ZhuLong"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$appData = Join-Path $env:APPDATA "ZhuLong"

Write-Host "`n========== V16 Full Stack Deploy ==========" -ForegroundColor Cyan

# P0-P3: Horizon + execution gates
& (Join-Path $Root "scripts\deploy_horizon_v16_production.ps1")
if (-not $?) { exit 1 }

& (Join-Path $Root "scripts\deploy_v16_execution_gates.ps1")
if (-not $?) { exit 1 }

# P2/P3: KN2 — 验收通过则 LIVE，否则 shadow
$kn2Deploy = Join-Path $Root "scripts\deploy_kn2_v16_when_ready.ps1"
if (Test-Path $kn2Deploy) {
    $kn2Report = Join-Path $Root "data\training\reports\kn2_v16\acceptance_report.json"
    $kn2Passed = $false
    if (Test-Path $kn2Report) {
        $kn2Acc = Get-Content $kn2Report -Raw -Encoding UTF8 | ConvertFrom-Json
        $kn2Passed = [bool]$kn2Acc.passed
    }
    if ($kn2Passed -and -not $ForceKn2Shadow) {
        & $kn2Deploy -EnableLive
    } else {
        if (-not $kn2Passed) {
            Write-Warning "KN2 acceptance not passed — deploying SHADOW only"
        }
        & $kn2Deploy -ForceShadow
    }
    if (-not $?) {
        Write-Warning "KN2 deploy script failed — check models and acceptance report"
    }
}

# Sync Python patches → Program Files (best effort)
$pyList = @(
    "zhulong\agent\trading_agent.py",
    "zhulong\agent\trader_mind.py",
    "zhulong\agent\execution_composer.py",
    "zhulong\agent\kn2_location_labels.py",
    "zhulong\agent\horizon_predictor.py",
    "zhulong\agent\knowledge_net.py",
    "zhulong\agent\knowledge_net_kn2.py",
    "zhulong\agent\cognition.py",
    "zhulong\agent\structure_service.py",
    "zhulong\agent\tick_brief.py",
    "ZhuLong.PythonEngine\inference_cli.py"
)
if (Test-Path $InstallDir) {
    foreach ($rel in $pyList) {
        $src = Join-Path $Root $rel
        if (-not (Test-Path $src)) { continue }
        $dst = Join-Path $InstallDir $rel
        $dir = Split-Path $dst -Parent
        try {
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
            Copy-Item -Force $src $dst -ErrorAction Stop
            Write-Host "InstallDir OK $rel" -ForegroundColor Green
        } catch {
            Write-Warning "InstallDir skip (need admin): $rel"
        }
    }
}

if (-not $SkipBuild) {
    Write-Host "`n=== Build ZhuLong.App Release x64 ===" -ForegroundColor Cyan
    dotnet build src/ZhuLong.App/ZhuLong.App.csproj -c Release -p:Platform=x64 `
        -p:PublishTrimmed=false -p:PublishReadyToRun=false
    if (-not $?) { exit 1 }
    $bin = Join-Path $Root "src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64"
    foreach ($item in @("config.json", "models", "data", "zhulong", "ZhuLong.PythonEngine")) {
        $src = Join-Path $Root $item
        $dst = Join-Path $bin $item
        if ((Test-Path $src) -and -not (Test-Path $dst)) {
            Copy-Item -Recurse -Force $src $dst
        }
    }
    # overlay latest python
    foreach ($rel in $pyList) {
        $src = Join-Path $Root $rel
        if (-not (Test-Path $src)) { continue }
        $dst = Join-Path $bin $rel
        $dir = Split-Path $dst -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        Copy-Item -Force $src $dst
    }
    $devExe = Join-Path $bin "ZhuLong.exe"
    if (Test-Path $devExe) {
        Write-Host "Built: $devExe" -ForegroundColor Green
        try {
            $instExe = Join-Path $InstallDir "ZhuLong.exe"
            if (Test-Path $InstallDir) {
                Copy-Item -Force $devExe $instExe -ErrorAction Stop
                Write-Host "Updated Program Files ZhuLong.exe" -ForegroundColor Green
            }
        } catch {
            Write-Warning "Could not copy exe to Program Files — run dev build from: $devExe"
        }
    }
}

if (-not $SkipRestart) {
    Write-Host "`n=== Restart ZhuLong ===" -ForegroundColor Cyan
    try {
        Get-Process ZhuLong -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction Stop
        Start-Sleep -Seconds 2
    } catch {
        Write-Warning "Could not stop ZhuLong (close manually or run as admin): $_"
    }
    $startExe = Join-Path $InstallDir "ZhuLong.exe"
    if (-not (Test-Path $startExe)) {
        $startExe = Join-Path $Root "src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64\ZhuLong.exe"
    }
    if (Test-Path $startExe) {
        Start-Process -FilePath $startExe -WorkingDirectory (Split-Path $startExe -Parent)
        Write-Host "Started $startExe" -ForegroundColor Green
    } else {
        Write-Warning "ZhuLong.exe not found — start manually"
    }
}

Write-Host "`n=== Post-deploy verify ===" -ForegroundColor Cyan
py -3 scripts/verify_v16_full_stack.py
py -3 scripts/horizon_v16_post_restart_check.py
exit $LASTEXITCODE
