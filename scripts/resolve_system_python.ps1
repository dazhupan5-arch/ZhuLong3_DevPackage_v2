# Resolve Python: system py/python first, optional bundled python_runtime fallback
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

function Get-SystemPythonInfo {
    $starters = @(
        @{ Exe = 'py'; Args = @('-3') },
        @{ Exe = 'python'; Args = @() },
        @{ Exe = 'python3'; Args = @() }
    )
    foreach ($s in $starters) {
        if (-not (Get-Command $s.Exe -ErrorAction SilentlyContinue)) { continue }
        $code = "import sys,os; b=getattr(sys,'base_prefix',sys.prefix); d=os.path.join(b,'python%d%d.dll'%(sys.version_info.major,sys.version_info.minor)); print(sys.executable); print(d)"
        $argList = @() + $s.Args + @('-c', $code)
        try {
            $out = & $s.Exe @argList 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $out) { continue }
            $lines = @($out | Where-Object { $_ -match '\S' })
            if ($lines.Count -lt 2) { continue }
            $exe = $lines[0].Trim()
            $dll = $lines[1].Trim()
            if ((Test-Path $exe) -and (Test-Path $dll)) {
                return [pscustomobject]@{ Executable = $exe; Dll = $dll; Launcher = $s.Exe }
            }
        }
        catch { continue }
    }

    $candidates = @()
    $localPy = Join-Path $env:LOCALAPPDATA 'Programs\Python'
    if (Test-Path $localPy) {
        $candidates += Get-ChildItem -Path $localPy -Recurse -Filter 'python.exe' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    }
    foreach ($pf in @(${env:ProgramFiles}, ${env:ProgramFiles(x86)})) {
        if (-not $pf) { continue }
        foreach ($leaf in @('Python313', 'Python312', 'Python311', 'Python310')) {
            $p = Join-Path (Join-Path $pf $leaf) 'python.exe'
            if (Test-Path $p) { $candidates += $p }
        }
    }
    foreach ($exe in ($candidates | Select-Object -Unique)) {
        if ($exe -match '\\WindowsApps\\') { continue }
        $dir = Split-Path $exe -Parent
        $dll = Get-ChildItem -Path $dir -Filter 'python3*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($dll) {
            return [pscustomobject]@{ Executable = $exe; Dll = $dll.FullName; Launcher = 'scan' }
        }
    }
    return $null
}

$info = Get-SystemPythonInfo
if (-not $info) {
    $bundledScript = Join-Path $InstallDir 'scripts\resolve_bundled_python.ps1'
    if (-not (Test-Path -LiteralPath $bundledScript)) {
        $bundledScript = Join-Path $InstallDir 'resolve_bundled_python.ps1'
    }
    if (Test-Path -LiteralPath $bundledScript) {
        & $bundledScript -InstallDir $InstallDir -Quiet:$Quiet
        if ($LASTEXITCODE -eq 0) { exit 0 }
    }
    Write-Host 'Python not found. Install Python 3.10+ and run install_python_deps.ps1.' -ForegroundColor Red
    exit 1
}

$env:PYTHONNET_PYDLL = $info.Dll
$env:ZHULONG_PYTHON = $info.Executable

$cacheDir = Join-Path $env:APPDATA 'ZhuLong'
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
Set-Content -Path (Join-Path $cacheDir 'python_dll.txt') -Value $info.Dll -Encoding utf8 -NoNewline
Set-Content -Path (Join-Path $cacheDir 'python_exe.txt') -Value $info.Executable -Encoding utf8 -NoNewline
Set-Content -Path (Join-Path $cacheDir 'python_bundled.txt') -Value '0' -Encoding utf8 -NoNewline

if (-not $Quiet) {
    Write-Host "System Python: $($info.Executable)" -ForegroundColor Yellow
    Write-Host "PYTHONNET_PYDLL: $($info.Dll)" -ForegroundColor Yellow
}

exit 0
