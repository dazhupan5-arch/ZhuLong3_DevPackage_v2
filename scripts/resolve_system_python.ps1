# Resolve system Python (py -3 / python), set PYTHONNET_PYDLL, cache to AppData
param([switch]$Quiet)

$ErrorActionPreference = 'Stop'

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
        } catch { continue }
    }
    return $null
}

$info = Get-SystemPythonInfo
if (-not $info) {
    Write-Host 'Python 3 not found. Install Python 3.10+ (Add to PATH) or ensure py -3 works.' -ForegroundColor Red
    Write-Host 'Then run install_python_deps.ps1 in the ZhuLong install folder.' -ForegroundColor Yellow
    exit 1
}

$env:PYTHONNET_PYDLL = $info.Dll
$env:ZHULONG_PYTHON = $info.Executable

$cacheDir = Join-Path $env:APPDATA 'ZhuLong'
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
Set-Content -Path (Join-Path $cacheDir 'python_dll.txt') -Value $info.Dll -Encoding utf8 -NoNewline
Set-Content -Path (Join-Path $cacheDir 'python_exe.txt') -Value $info.Executable -Encoding utf8 -NoNewline

if (-not $Quiet) {
    Write-Host "Python: $($info.Executable)" -ForegroundColor Green
    Write-Host "PYTHONNET_PYDLL: $($info.Dll)" -ForegroundColor Green
}

exit 0
