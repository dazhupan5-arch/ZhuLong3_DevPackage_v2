# 安装后：修复 runtimeconfig + 若本机有 Python 则安装 pip 依赖
param(
    [string]$InstallDir = ''
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$InstallDir = (Get-Item -LiteralPath $InstallDir).FullName

Write-Host "ZhuLong post-install: $InstallDir"

# 覆盖安装时清理旧版自包含 host 残留，避免与 framework-dependent runtimeconfig 混用导致 .NET 弹窗
$staleHostFiles = @(
    'coreclr.dll', 'hostfxr.dll', 'hostpolicy.dll', 'clrjit.dll',
    'mscordaccore.dll', 'mscordbi.dll', 'createdump.exe'
)
$rcPath = Join-Path $InstallDir 'ZhuLong.runtimeconfig.json'
$isFrameworkDependent = $false
if (Test-Path -LiteralPath $rcPath) {
    try {
        $rc = Get-Content -LiteralPath $rcPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $isFrameworkDependent = $null -ne $rc.runtimeOptions.frameworks -and -not $rc.runtimeOptions.includedFrameworks
    } catch { }
}
if ($isFrameworkDependent) {
    foreach ($f in $staleHostFiles) {
        $p = Join-Path $InstallDir $f
        if (Test-Path -LiteralPath $p) {
            Remove-Item -LiteralPath $p -Force
            Write-Host "  removed stale host file: $f" -ForegroundColor Yellow
        }
    }
}

$fixRc = Join-Path $InstallDir 'scripts\fix_runtimeconfig.ps1'
if (Test-Path -LiteralPath $fixRc) {
    & $fixRc -StageDir $InstallDir
}

$resolvePy = Join-Path $InstallDir 'resolve_system_python.ps1'
if (-not (Test-Path -LiteralPath $resolvePy)) {
    $resolvePy = Join-Path $InstallDir 'scripts\resolve_system_python.ps1'
}

$python = $null
if (Test-Path -LiteralPath $resolvePy) {
    try {
        & $resolvePy -Quiet
        $cache = Join-Path $env:APPDATA 'ZhuLong\python_exe.txt'
        if (Test-Path -LiteralPath $cache) {
            $python = (Get-Content -LiteralPath $cache -Raw).Trim()
        }
    } catch {
        Write-Warning "Python not found on this machine: $_"
    }
}

if (-not $python -or -not (Test-Path -LiteralPath $python)) {
    Write-Warning 'Skip pip install: Python 3.10+ not found. Install Python then run install_python_deps.ps1'
    exit 0
}

$deps = Join-Path $InstallDir 'install_python_deps.ps1'
if (-not (Test-Path -LiteralPath $deps)) {
    Write-Warning "Missing install_python_deps.ps1"
    exit 0
}

Write-Host "Running install_python_deps.ps1 with $python"
& powershell -NoProfile -ExecutionPolicy Bypass -File $deps -PythonExe $python -Root $InstallDir
exit $LASTEXITCODE
