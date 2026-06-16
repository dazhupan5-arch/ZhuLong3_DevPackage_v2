# 编译 ZhuLongMt5Pipe.dll（x64）→ mql5\Libraries\
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$vcx = Join-Path $root 'native\ZhuLongMt5Pipe\ZhuLongMt5Pipe.vcxproj'
if (-not (Test-Path $vcx)) { throw "Missing $vcx" }

$msbuild = $null
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $install = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath 2>$null
    if ($install) {
        $cand = Join-Path $install 'MSBuild\Current\Bin\MSBuild.exe'
        if (Test-Path $cand) { $msbuild = $cand }
    }
}
if (-not $msbuild) {
    $msbuild = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $msbuild) { throw 'MSBuild not found. Install VS 2022 with C++ workload.' }

$ok = $false
foreach ($ts in @('v143', 'v142')) {
    Write-Host "MSBuild PlatformToolset=$ts ..."
    & $msbuild $vcx /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=$ts /v:m
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
}
if (-not $ok) { throw 'MSBuild failed' }

$out = Join-Path $root 'mql5\Libraries\ZhuLongMt5Pipe.dll'
if (-not (Test-Path $out)) { throw "Expected output missing: $out" }
Write-Host "OK: $out" -ForegroundColor Green
