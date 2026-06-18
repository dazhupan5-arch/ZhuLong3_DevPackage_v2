# Register bundled python_runtime from install dir (self-contained, no system Python scan)
#Requires -Version 5.1
param(
    [string]$InstallDir = '',
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ((Split-Path -Leaf $InstallDir) -eq 'scripts') {
        $InstallDir = Split-Path -Parent $InstallDir
    }
}
$InstallDir = (Get-Item -LiteralPath $InstallDir).FullName

$rtDir = Join-Path $InstallDir 'python_runtime'
$exe = Join-Path $rtDir 'python.exe'
$marker = Join-Path $rtDir 'BUNDLED.json'

if (-not (Test-Path -LiteralPath $exe)) {
    if (-not $Quiet) {
        Write-Host "Bundled python.exe not found under $InstallDir" -ForegroundColor Red
    }
    exit 1
}

$dll = Get-ChildItem -LiteralPath $rtDir -Filter 'python3*.dll' -File -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $dll) {
    if (-not $Quiet) { Write-Host 'Bundled python3*.dll missing' -ForegroundColor Red }
    exit 1
}

$env:PYTHONNET_PYDLL = $dll.FullName
$env:ZHULONG_PYTHON = $exe
$env:ZHULONG_BUNDLED_PYTHON = '1'

$cacheDir = Join-Path $env:APPDATA 'ZhuLong'
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
Set-Content -LiteralPath (Join-Path $cacheDir 'python_dll.txt') -Value $dll.FullName -Encoding utf8 -NoNewline
Set-Content -LiteralPath (Join-Path $cacheDir 'python_exe.txt') -Value $exe -Encoding utf8 -NoNewline
Set-Content -LiteralPath (Join-Path $cacheDir 'python_bundled.txt') -Value '1' -Encoding utf8 -NoNewline

if (-not $Quiet) {
    Write-Host "Bundled Python: $exe" -ForegroundColor Green
    Write-Host "PYTHONNET_PYDLL: $($dll.FullName)" -ForegroundColor Green
    if (-not (Test-Path -LiteralPath $marker)) {
        Write-Host 'Note: BUNDLED.json not yet written (pack-time build OK)' -ForegroundColor Yellow
    }
}

exit 0
