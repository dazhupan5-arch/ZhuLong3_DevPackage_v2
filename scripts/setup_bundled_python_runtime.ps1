# Build bundled python_runtime (embeddable Python + pip deps + Horizon probe)
#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot,
    [string]$RepoRoot = '',
    [switch]$Force,
    [string]$PythonVersion = '3.11.9'
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
$InstallRoot = (Get-Item -LiteralPath $InstallRoot).FullName
$RepoRoot = (Get-Item -LiteralPath $RepoRoot).FullName

$PyDir = Join-Path $InstallRoot 'python_runtime'
$ReqFile = Join-Path $InstallRoot 'requirements_runtime.txt'
if (-not (Test-Path -LiteralPath $ReqFile)) {
    $ReqFile = Join-Path $RepoRoot 'requirements_runtime.txt'
}
if (-not (Test-Path -LiteralPath $ReqFile)) {
    throw 'requirements_runtime.txt not found'
}

$reqHash = (Get-FileHash -LiteralPath $ReqFile -Algorithm SHA256).Hash
$markerPath = Join-Path $PyDir 'BUNDLED.json'
$pyExe = Join-Path $PyDir 'python.exe'

if (-not $Force -and (Test-Path -LiteralPath $markerPath) -and (Test-Path -LiteralPath $pyExe)) {
    try {
        $marker = Get-Content -LiteralPath $markerPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($marker.python_version -eq $PythonVersion -and $marker.requirements_sha256 -eq $reqHash) {
            Write-Host '  bundled python_runtime up-to-date (skip rebuild)' -ForegroundColor Green
            & (Join-Path $RepoRoot 'scripts\probe_bundled_horizon.ps1') -InstallDir $InstallRoot -Quiet
            if ($LASTEXITCODE -eq 0) { exit 0 }
            Write-Host '  marker OK but Horizon probe failed - rebuilding' -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host '  invalid BUNDLED.json - rebuilding' -ForegroundColor Yellow
    }
}

$majorMinor = ($PythonVersion -split '\.')[0..1] -join '.'
$pthTag = ($PythonVersion -split '\.')[0] + ($PythonVersion -split '\.')[1]
$pthFile = Join-Path $PyDir "python$pthTag._pth"

Write-Host "== Building bundled python_runtime $PythonVersion ==" -ForegroundColor Cyan
Write-Host "  target: $PyDir"

$cacheDir = Join-Path $RepoRoot 'build_cache'
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
$zipName = "python-$PythonVersion-embed-amd64.zip"
$zipPath = Join-Path $cacheDir $zipName
$zipUrl = "https://www.python.org/ftp/python/$PythonVersion/$zipName"

if (-not (Test-Path -LiteralPath $zipPath)) {
    Write-Host "  downloading $zipUrl ..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
}

if (Test-Path -LiteralPath $PyDir) {
    Remove-Item -Recurse -Force -LiteralPath $PyDir
}
New-Item -ItemType Directory -Force -Path $PyDir | Out-Null
Expand-Archive -LiteralPath $zipPath -DestinationPath $PyDir -Force

$sitePackages = Join-Path $PyDir 'Lib\site-packages'
New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null

$pthLines = @(
    "python$pthTag.zip"
    '.'
    'Lib\site-packages'
    'import site'
)
Set-Content -LiteralPath $pthFile -Value ($pthLines -join [Environment]::NewLine) -Encoding ASCII

$getPip = Join-Path $cacheDir 'get-pip.py'
if (-not (Test-Path -LiteralPath $getPip)) {
    Write-Host '  downloading get-pip.py ...' -ForegroundColor Cyan
    Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getPip -UseBasicParsing
}

$env:PYTHONHOME = $PyDir
$prevPath = $env:PATH
$env:PATH = "$PyDir;$env:PATH"

Write-Host '  installing pip ...' -ForegroundColor Cyan
& $pyExe $getPip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw 'get-pip failed' }

Write-Host '  pip install requirements_runtime.txt (may take several minutes) ...' -ForegroundColor Cyan
& $pyExe -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $pyExe -m pip install --prefer-binary -r $ReqFile
if ($LASTEXITCODE -ne 0) { throw 'pip install requirements_runtime failed' }

Write-Host '  import smoke ...' -ForegroundColor Cyan
$smokeFile = Join-Path $env:TEMP 'zhulong_bundled_smoke.py'
# Forward slashes avoid PowerShell/Python raw-string backslash doubling bugs.
$rootPy = ($InstallRoot -replace '\\', '/')
@(
    'import sys, os',
    'from pathlib import Path',
    "install = Path('$rootPy')",
    'sys.path.insert(0, str(install))',
    'from zhulong.utils.win_dll import configure_native_dll_paths',
    'configure_native_dll_paths()',
    '# onnxruntime must load before numpy/pandas on Windows (DLL search order)',
    'import onnxruntime as ort',
    'import importlib',
    'mods = ["numpy","pandas","sklearn","joblib","MetaTrader5","torch","stable_baselines3"]',
    'for m in mods:',
    '    importlib.import_module(m)',
    '    print("OK", m)',
    'onnx = install / "models" / "horizon_v16.onnx"',
    'if onnx.is_file():',
    '    ort.InferenceSession(str(onnx), ort.SessionOptions(), providers=["CPUExecutionProvider"])',
    '    print("ONNX_SESSION_OK")',
    'print("SMOKE_OK")'
) | Set-Content -LiteralPath $smokeFile -Encoding UTF8
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
& $pyExe $smokeFile
if ($LASTEXITCODE -ne 0) { throw 'bundled import smoke failed' }

$env:PATH = $prevPath
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue

& (Join-Path $RepoRoot 'scripts\trim_python_runtime_for_release.ps1') -StagingRoot $InstallRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '  post-trim import check ...' -ForegroundColor Cyan
$postTrim = Join-Path $env:TEMP 'zhulong_bundled_post_trim.py'
@(
    'import sys',
    'from pathlib import Path',
    "install = Path('$rootPy')",
    'sys.path.insert(0, str(install))',
    'from zhulong.utils.win_dll import configure_native_dll_paths',
    'configure_native_dll_paths()',
    'import onnxruntime',
    'import narwhals',
    'import sklearn',
    'print("POST_TRIM_OK")'
) | Set-Content -LiteralPath $postTrim -Encoding UTF8
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
& $pyExe $postTrim
if ($LASTEXITCODE -ne 0) { throw 'post-trim import check failed' }

$horizonOnnx = Join-Path $InstallRoot 'models\horizon_v16.onnx'
$agentCfg = Join-Path $InstallRoot 'config\config_agent.json'
if ((Test-Path -LiteralPath $horizonOnnx) -and (Test-Path -LiteralPath $agentCfg)) {
    & (Join-Path $RepoRoot 'scripts\probe_bundled_horizon.ps1') -InstallDir $InstallRoot
    if ($LASTEXITCODE -ne 0) { throw 'Horizon probe failed on bundled python - pack blocked' }
}
else {
    Write-Host '  Horizon probe skipped (model/config not staged yet; pack-installer runs probe after models copy)' -ForegroundColor Yellow
}

$marker = [ordered]@{
    python_version      = $PythonVersion
    python_major_minor  = $majorMinor
    requirements_sha256 = $reqHash
    built_utc           = (Get-Date).ToUniversalTime().ToString('o')
    bundled             = $true
}
$marker | ConvertTo-Json | Set-Content -LiteralPath $markerPath -Encoding UTF8

$total = (Get-ChildItem $PyDir -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("  bundled python_runtime OK: {0:N1} MB" -f ($total / 1MB)) -ForegroundColor Green
exit 0
