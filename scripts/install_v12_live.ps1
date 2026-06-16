# ZhuLong v12 live experiment: deploy model, MT5 indicator, start app
#Requires -Version 5.1
param(
    [switch]$SkipBuild,
    [switch]$SkipStart
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

Write-Host '== [1/7] deploy v12 production model ==' -ForegroundColor Cyan
py -3 scripts/deploy_v12_production.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== [2/7] Python deps (optional) ==' -ForegroundColor Cyan
try {
    & (Join-Path $root 'scripts\install_python_deps.ps1')
} catch {
    Write-Host 'WARN: skip python deps script' -ForegroundColor Yellow
}

Write-Host '== [3/7] MT5 pipe + indicator ==' -ForegroundColor Cyan
$pipeDll = Join-Path $root 'mql5\Libraries\ZhuLongMt5Pipe.dll'
if (-not (Test-Path $pipeDll)) {
    try { & (Join-Path $root 'scripts\build-zhulong-mt5-pipe.ps1') } catch { Write-Host 'WARN: pipe build skipped' -ForegroundColor Yellow }
} else {
    Write-Host '  using existing ZhuLongMt5Pipe.dll' -ForegroundColor Gray
}
try { & (Join-Path $root 'scripts\deploy-mt5-indicator.ps1') } catch { Write-Host 'WARN: MT5 deploy skipped' -ForegroundColor Yellow }

Write-Host '== [4/7] validate v12 inference ==' -ForegroundColor Cyan
py -3 scripts/probe_v12_live.py
if ($LASTEXITCODE -ne 0) { Write-Host 'WARN: probe failed' -ForegroundColor Yellow }

if (-not $SkipBuild) {
    Write-Host '== [5/7] build ZhuLong Release ==' -ForegroundColor Cyan
    dotnet build src/ZhuLong.App/ZhuLong.App.csproj -c Release -p:Platform=x64 `
        -p:PublishTrimmed=false -p:PublishReadyToRun=false
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$bin = Join-Path $root 'src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64'
Write-Host '== [6/7] sync runtime assets ==' -ForegroundColor Cyan
foreach ($item in @('config.json', 'models', 'data', 'zhulong', 'ZhuLong.PythonEngine', 'mql5')) {
    $src = Join-Path $root $item
    $dst = Join-Path $bin $item
    if (-not (Test-Path $src)) { continue }
    if ($item -eq 'models') {
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    }
    Copy-Item -Recurse -Force $src $dst
}

Write-Host '== [7/7] production model gate ==' -ForegroundColor Cyan
$gateJson = powershell -NoProfile -File (Join-Path $root 'scripts\check_production_models.ps1') 2>&1 | Select-Object -Last 1
Write-Host $gateJson

if (-not $SkipStart) {
    Get-Process ZhuLong -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    $exe = Join-Path $bin 'ZhuLong.exe'
    if (-not (Test-Path $exe)) { throw 'ZhuLong.exe not found' }
    Write-Host "Starting ZhuLong: $exe" -ForegroundColor Green
    Start-Process $exe -WorkingDirectory $bin
    Write-Host 'Next: MT5 demo, ZhuLongIndicator on XAUUSD M1, click Start in ZhuLong' -ForegroundColor Cyan
}

Write-Host 'v12 live install done' -ForegroundColor Green
