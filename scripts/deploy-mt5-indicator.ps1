# Deploy ZhuLongMt5Pipe.dll + ZhuLongIndicator to all local MT5 data folders.
# Dev:  .\scripts\deploy-mt5-indicator.ps1 [-StopMt5First] [-Compile]
# Installed: .\scripts\deploy-mt5-indicator.ps1 -InstallDir "C:\Program Files\ZhuLong" -StopMt5First
param(
    [string] $InstallDir = '',
    [switch] $StopMt5First,
    [switch] $Compile,
    [switch] $PatchDefaultChart,
    [switch] $Quiet
)

$ErrorActionPreference = 'Stop'

function Write-DeployLog {
    param([string]$Message, [string]$Level = 'INFO')
    $line = "[{0}] [{1}] {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    if (-not $Quiet) {
        $color = switch ($Level) { 'WARN' { 'Yellow' } 'ERR' { 'Red' } default { 'Gray' } }
        Write-Host $line -ForegroundColor $color
    }
    try {
        $logDir = Join-Path $env:LOCALAPPDATA 'ZhuLong'
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        Add-Content -Path (Join-Path $logDir 'mt5_deploy.log') -Value $line -Encoding UTF8
    }
    catch { }
}

function Resolve-ProductRoot {
    param([string]$Dir)
    if ($Dir -and (Test-Path -LiteralPath $Dir)) {
        $Dir = (Resolve-Path -LiteralPath $Dir).Path
        if (Test-Path (Join-Path $Dir 'mql5\Libraries\ZhuLongMt5Pipe.dll')) { return $Dir }
        if (Test-Path (Join-Path $Dir 'mql5\ZhuLongIndicator.mq5')) { return $Dir }
    }
    $viaScript = Split-Path $PSScriptRoot -Parent
    if (Test-Path (Join-Path $viaScript 'mql5\Libraries\ZhuLongMt5Pipe.dll')) { return $viaScript }
    if (Test-Path (Join-Path $viaScript 'mql5\ZhuLongIndicator.mq5')) { return $viaScript }
    throw 'Product root not found (need mql5\Libraries\ZhuLongMt5Pipe.dll). Use -InstallDir.'
}

function Get-Mt5DataRoots {
    $roots = [System.Collections.Generic.List[string]]::new()

    $appData = Join-Path $env:APPDATA 'MetaQuotes\Terminal'
    if (Test-Path $appData) {
        Get-ChildItem $appData -Directory -EA SilentlyContinue |
            Where-Object { $_.Name -match '^[0-9A-F]{32}$' -and (Test-Path (Join-Path $_.FullName 'MQL5')) } |
            ForEach-Object { [void]$roots.Add($_.FullName) }
    }

    $portableCandidates = @(
        'C:\Program Files\WCG Group MT5 Terminal',
        'C:\Program Files\MetaTrader 5',
        'D:\Program Files\MetaTrader 5',
        'C:\Program Files (x86)\MetaTrader 5'
    )
    foreach ($p in $portableCandidates) {
        if ((Test-Path $p) -and (Test-Path (Join-Path $p 'MQL5'))) {
            [void]$roots.Add((Resolve-Path $p).Path)
        }
    }

    foreach ($pf in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, 'D:\Program Files', 'E:\Program Files')) {
        if (-not $pf -or -not (Test-Path $pf)) { continue }
        Get-ChildItem $pf -Directory -EA SilentlyContinue | Where-Object {
            (Test-Path (Join-Path $_.FullName 'terminal64.exe')) -and (Test-Path (Join-Path $_.FullName 'MQL5'))
        } | ForEach-Object { [void]$roots.Add($_.FullName) }
    }

    $proc = Get-Process terminal64 -EA SilentlyContinue | Select-Object -First 1
    if ($proc -and $proc.Path) {
        $fromProc = Split-Path $proc.Path -Parent
        if (Test-Path (Join-Path $fromProc 'MQL5')) { [void]$roots.Add($fromProc) }
    }

    return $roots | Select-Object -Unique
}

function Find-MetaEditor {
    $candidates = @()
    foreach ($root in (Get-Mt5DataRoots)) {
        $candidates += Join-Path $root 'metaeditor64.exe'
    }
    $candidates += @(
        'C:\Program Files\WCG Group MT5 Terminal\metaeditor64.exe',
        'C:\Program Files\MetaTrader 5\metaeditor64.exe',
        'D:\Program Files\MetaTrader 5\metaeditor64.exe'
    )
    return $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

function Copy-FileWithRetry {
    param([string]$Src, [string]$Dst, [int]$MaxTry = 3)
    for ($i = 0; $i -lt $MaxTry; $i++) {
        try {
            Copy-Item -Force -LiteralPath $Src -Destination $Dst
            return
        }
        catch {
            if ($i -lt ($MaxTry - 1)) {
                Write-DeployLog "File locked, retry without closing MT5: $Dst" 'WARN'
                Start-Sleep -Seconds 2
            }
            else { throw }
        }
    }
}

$root = Resolve-ProductRoot -Dir $InstallDir
$dll = Join-Path $root 'mql5\Libraries\ZhuLongMt5Pipe.dll'
$mq5 = Join-Path $root 'mql5\ZhuLongIndicator.mq5'
$ex5 = Join-Path $root 'mql5\ZhuLongIndicator.ex5'
$readme = Join-Path $root 'mql5\Libraries\ZhuLong_部署说明.txt'

if (-not (Test-Path $mq5)) { $mq5 = Join-Path $root 'indicators\ZhuLongIndicator.mq5' }
if (-not (Test-Path $ex5)) { $ex5 = Join-Path $root 'indicators\ZhuLongIndicator.ex5' }

if (-not (Test-Path $dll)) {
    $buildScript = Join-Path (Split-Path $PSScriptRoot -Parent) 'scripts\build-zhulong-mt5-pipe.ps1'
    if (Test-Path $buildScript) {
        Write-DeployLog 'ZhuLongMt5Pipe.dll missing, building...' 'WARN'
        & $buildScript
    }
}
if (-not (Test-Path $dll)) {
    Write-DeployLog "Missing ZhuLongMt5Pipe.dll: $dll" 'ERR'
    exit 1
}
if (-not (Test-Path $mq5)) {
    Write-DeployLog "Missing ZhuLongIndicator.mq5: $mq5" 'ERR'
    exit 1
}

Write-DeployLog "Product root: $root"
Write-DeployLog "Indicator source: $mq5"

if ($StopMt5First) {
    Get-Process terminal64 -EA SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-DeployLog 'Stopped terminal64'
}

$targets = @(Get-Mt5DataRoots)
if (-not $targets -or $targets.Count -eq 0) {
    Write-DeployLog 'No MT5 terminal folder found. Install MT5 first, then re-run this script.' 'WARN'
    Write-DeployLog 'Manual: copy mql5\Libraries\ZhuLongMt5Pipe.dll -> <MT5>\MQL5\Libraries\' 'WARN'
    Write-DeployLog 'Manual: copy ZhuLongIndicator.ex5 + .mq5 -> <MT5>\MQL5\Indicators\' 'WARN'
    exit 0
}

$metaEditor = Find-MetaEditor
$deployed = 0

foreach ($base in $targets) {
    $lib = Join-Path $base 'MQL5\Libraries'
    $ind = Join-Path $base 'MQL5\Indicators'
    New-Item -ItemType Directory -Force -Path $lib, $ind | Out-Null

    Copy-FileWithRetry -Src $dll -Dst (Join-Path $lib 'ZhuLongMt5Pipe.dll')
    $copied = Get-Item (Join-Path $lib 'ZhuLongMt5Pipe.dll') -ErrorAction Stop
    if ($copied.Length -lt 10000) { throw "DLL deploy verify failed: $($copied.FullName) size=$($copied.Length)" }
    Copy-FileWithRetry -Src $mq5 -Dst (Join-Path $ind 'ZhuLongIndicator.mq5')

    $destEx5 = Join-Path $ind 'ZhuLongIndicator.ex5'
    if (Test-Path $ex5) {
        Copy-FileWithRetry -Src $ex5 -Dst $destEx5
        Write-DeployLog "Deployed prebuilt ex5 -> $destEx5"
    }
    elseif ($Compile -and $metaEditor) {
        $destMq5 = Join-Path $ind 'ZhuLongIndicator.mq5'
        $p = Start-Process -FilePath $metaEditor -ArgumentList @("/compile:$destMq5", '/log') -PassThru -Wait
        if (Test-Path $destEx5) {
            Write-DeployLog "MetaEditor compile OK -> $destEx5"
        }
        else {
            Write-DeployLog "MetaEditor did not produce ex5 (exit=$($p.ExitCode)); F7 compile $destMq5 in MT5" 'WARN'
        }
    }
    else {
        Write-DeployLog 'No ex5 bundled and -Compile not set; F7 compile in MetaEditor if needed' 'WARN'
    }

    if (Test-Path $readme) {
        Copy-Item -Force $readme (Join-Path $lib 'ZhuLong_部署说明.txt')
    }

    if ($PatchDefaultChart) {
        $chart = Join-Path $base 'MQL5\Profiles\Charts\Default\chart01.chr'
        if (Test-Path $chart) {
            $text = Get-Content $chart -Raw -Encoding Default
            if ($text -notmatch 'ZhuLongIndicator') {
                $block = @"

<indicator>
name=ZhuLongIndicator
path=Indicators\ZhuLongIndicator.ex5
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1
</indicator>
"@
                $text = $text -replace '(?s)(<indicator>\r?\nname=Main.*?</indicator>)', ('${1}' + $block)
                [IO.File]::WriteAllText($chart, $text, [Text.Encoding]::Default)
                Write-DeployLog "Patched default chart -> $chart"
            }
        }
    }

    Write-DeployLog "OK -> $base"
    $deployed++
}

Write-DeployLog ("Deploy complete: {0} MT5 root(s). Restart MT5 and attach ZhuLongIndicator on XAUUSD M1." -f $deployed)
Write-DeployLog 'MT5: Tools -> Options -> Expert Advisors -> Allow DLL imports'
if (-not $Quiet) {
    Write-Host ''
    Write-Host 'Deploy complete. Next steps:' -ForegroundColor Green
    Write-Host '  1. Restart MT5' -ForegroundColor Cyan
    Write-Host '  2. Tools -> Options -> Expert Advisors -> Allow DLL imports' -ForegroundColor Cyan
    Write-Host '  3. XAUUSD M1 chart -> Navigator -> Indicators -> ZhuLongIndicator' -ForegroundColor Cyan
    Write-Host '  4. Start ZhuLong -> Connect MT5 -> Start trading' -ForegroundColor Cyan
    Write-Host ("  Log: {0}" -f (Join-Path $env:LOCALAPPDATA 'ZhuLong\mt5_deploy.log')) -ForegroundColor Gray
}
