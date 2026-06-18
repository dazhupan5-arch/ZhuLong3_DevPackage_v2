# 烛龙 ZhuLong_3 安装包（对齐 V14 pack-installer 流程）
#Requires -Version 5.1
param(
    [ValidateSet('Setup', 'Folder')]
    [string] $Output = 'Setup',
    [switch] $SkipPublish,
    [switch] $KeepPublish,
    [switch] $SkipLogo,
    [switch] $SkipModels,
    [switch] $ForcePack
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

Write-Host '== Pre-pack Python syntax gate (source) ==' -ForegroundColor Cyan
& (Join-Path $RepoRoot 'scripts\validate_python_agent_bundle.ps1') -Root $RepoRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $ForcePack) {
    Write-Host 'Running V16 dual-closure gate before pack...' -ForegroundColor Cyan
    & (Join-Path $RepoRoot 'scripts\deploy_and_verify_v16_dual_closure.ps1') -SkipAdmin
    if ($LASTEXITCODE -ne 0) {
        Write-Error 'Dual closure FAILED — pack blocked. Fix issues or use -ForcePack (not recommended).'
    }
}
$StageDir = Join-Path $RepoRoot 'publish\win-x64'
$BinDir = Join-Path $RepoRoot 'src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64'
$outDir = Join-Path $RepoRoot 'output'
$iss = Join-Path $RepoRoot 'installer\build_installer.iss'

function Find-Iscc {
    foreach ($p in @(
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
        )) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Stage-InstallerModels {
    param([string]$DestRoot)
    $modelsDst = Join-Path $DestRoot 'models'
    if (Test-Path $modelsDst) { Remove-Item -Recurse -Force $modelsDst }
    New-Item -ItemType Directory -Force -Path $modelsDst | Out-Null

    $xauSrc = Join-Path $RepoRoot 'models\XAUUSD'
    $xauDst = Join-Path $modelsDst 'XAUUSD'
    New-Item -ItemType Directory -Force -Path $xauDst | Out-Null
    # 运行时 load_v14_bundle 读取 models/XAUUSD/v14/
    $xauV14Files = @('xgb_v14.json', 'v14_meta.pkl', 'feature_columns.json', 'config_v14.json')
    $xauV14Src = Join-Path $xauSrc 'v14'
    $xauV14Dst = Join-Path $xauDst 'v14'
    New-Item -ItemType Directory -Force -Path $xauV14Dst | Out-Null
    foreach ($f in $xauV14Files) {
        $src = Join-Path $xauV14Src $f
        if (-not (Test-Path $src)) { $src = Join-Path $xauSrc $f }
        if (-not (Test-Path $src)) {
            if ($f -eq 'config_v14.json') { continue }
            throw "Missing XAUUSD v14 artifact: $f"
        }
        Copy-Item -Force $src (Join-Path $xauV14Dst $f)
    }
    $xauRootFiles = @('manifest.json', 'acceptance_summary.json', 'imf_vmd.parquet', 'imf_vmd.csv')
    foreach ($f in $xauRootFiles) {
        $src = Join-Path $xauSrc $f
        if (-not (Test-Path $src)) {
            if ($f -eq 'imf_vmd.csv') { continue }
            throw "Missing XAUUSD artifact: $src"
        }
        Copy-Item -Force $src (Join-Path $xauDst $f)
    }

    Write-Host '  models: USOIL v14 production' -ForegroundColor Cyan
    $usoV14Src = Join-Path (Join-Path $RepoRoot 'models') 'USOIL\v14'
    $usoV14Dst = Join-Path (Join-Path $modelsDst 'USOIL') 'v14'
    if (Test-Path $usoV14Src) {
        New-Item -ItemType Directory -Force -Path $usoV14Dst | Out-Null
        Copy-Item -Recurse -Force "$usoV14Src\*" $usoV14Dst
    } else {
        Write-Warning "USOIL v14 model not found, skipping"
    }

    $oilSummary = Join-Path $RepoRoot 'models\USOIL\acceptance_summary.json'
    if (Test-Path $oilSummary) {
        Copy-Item -Force $oilSummary (Join-Path (Join-Path $modelsDst 'USOIL') 'acceptance_summary.json')
    }
    $oilManifest = Join-Path $RepoRoot 'models\USOIL\manifest.json'
    if (Test-Path $oilManifest) {
        Copy-Item -Force $oilManifest (Join-Path (Join-Path $modelsDst 'USOIL') 'manifest.json')
    }
}

function Stage-ZhulongRuntime {
    param([string]$DestRoot)
    $zhSrc = Join-Path $RepoRoot 'zhulong'
    $zhDst = Join-Path $DestRoot 'zhulong'
    if (Test-Path $zhDst) { Remove-Item -Recurse -Force $zhDst }
    Copy-TreeClean -Src $zhSrc -Dst $zhDst -ExcludeDirs @('__pycache__', '.pytest_cache', 'tests')

    $trainDst = Join-Path $zhDst 'training'
    if (Test-Path $trainDst) {
        Get-ChildItem $trainDst -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne 'lgb' } |
            ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
        $lgbDst = Join-Path $trainDst 'lgb'
        if (Test-Path $lgbDst) {
            foreach ($f in @(
                    'train.py', 'train_binary.py', 'labels.py', 'labels_profit.py',
                    'triple_barrier.py', 'data_io.py', 'splits.py', 'acceptance.py'
                )) {
                $p = Join-Path $lgbDst $f
                if (Test-Path $p) { Remove-Item -Force $p }
            }
        }
    }
    foreach ($legacy in @(
            'v12_live.py', 'live_v8_features.py', 'live_oil_features.py',
            'training\v12', 'strategies\v12_structure_filter.py',
            'inference\v12.py', 'inference\oil_v1.py'
        )) {
        $p = Join-Path $zhDst $legacy
        if (Test-Path -LiteralPath $p) { Remove-Item -Recurse -Force -LiteralPath $p }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $zhDst 'v14_live.py'))) {
        throw "Staging missing zhulong/v14_live.py"
    }
    Write-Host '  zhulong: runtime slice (lgb/features+backtest only)' -ForegroundColor Cyan
}

function Copy-TreeClean {
    param([string]$Src, [string]$Dst, [string[]]$ExcludeDirs = @('__pycache__', '.pytest_cache', 'tests', 'training'))
    if (-not (Test-Path $Src)) { return }
    if (Test-Path $Dst) { Remove-Item -Recurse -Force $Dst }
    New-Item -ItemType Directory -Force -Path $Dst | Out-Null
    $robocopyArgs = @($Src, $Dst, '/E', '/NFL', '/NDL', '/NJH', '/NJS')
    if ($ExcludeDirs.Count -gt 0) {
        $robocopyArgs += '/XD'
        $robocopyArgs += $ExcludeDirs
    }
    & robocopy @robocopyArgs | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed: $Src -> $Dst (exit $LASTEXITCODE)" }
    $global:LASTEXITCODE = 0
}

Write-Host '== [0/6] Installer redist (WinUI runtime + VC++) ==' -ForegroundColor Cyan
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot 'scripts\fetch_installer_redist.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== [1/6] Logo ==' -ForegroundColor Cyan
if ($SkipLogo) {
    Write-Host '  skipped (-SkipLogo)' -ForegroundColor Yellow
} else {
    py -3 scripts/generate_app_icons.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host '== [2/6] production models (XAUUSD v14 + USOIL v14) ==' -ForegroundColor Cyan
if ($SkipModels) {
    Write-Host '  skipped (-SkipModels)' -ForegroundColor Yellow
} else {
    py -3 scripts/deploy_v14_production.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
# USOIL v14 already in models/USOIL/v14, skip v1 deploy

$mt5Dll = Join-Path $RepoRoot 'mql5\Libraries\ZhuLongMt5Pipe.dll'
if (Test-Path $mt5Dll) {
    Write-Host "== [3/6] ZhuLongMt5Pipe.dll (skip rebuild, use existing) ==" -ForegroundColor Cyan
} else {
    Write-Host '== [3/6] ZhuLongMt5Pipe.dll ==' -ForegroundColor Cyan
    & (Join-Path $RepoRoot 'scripts\build-zhulong-mt5-pipe.ps1')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host '== [4/6] Clean publish staging ==' -ForegroundColor Cyan
if ($KeepPublish) {
    if (-not (Test-Path (Join-Path $StageDir 'ZhuLong.exe'))) {
        throw "KeepPublish: missing $StageDir\ZhuLong.exe — run dotnet publish first"
    }
    Write-Host '  kept existing publish staging (-KeepPublish)' -ForegroundColor Yellow
} else {
    if (Test-Path $StageDir) { Remove-Item -Recurse -Force $StageDir }
    New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
}

if ($KeepPublish) {
    Write-Host '== [5/6] skipped publish (-KeepPublish) ==' -ForegroundColor Yellow
} elseif ($SkipPublish) {
    Write-Host '== [5/6] reuse bin build (-SkipPublish) ==' -ForegroundColor Cyan
    if (-not (Test-Path (Join-Path $BinDir 'ZhuLong.exe'))) {
        throw "Missing bin build: $BinDir\ZhuLong.exe - run dotnet publish first"
    }
    $publishExclude = @('data','models','assets','zhulong','mql5','ZhuLong.PythonEngine','python_runtime','config','indicators','scripts','redist')
    $robocopyArgs = @($BinDir, $StageDir, '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/XD') + $publishExclude
    & robocopy @robocopyArgs | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed: $BinDir -> $StageDir (exit $LASTEXITCODE)" }
    $global:LASTEXITCODE = 0
    Write-Host "  publish slice only (excluded dev data/models): $BinDir" -ForegroundColor Green
} else {
    Write-Host '== [5/6] dotnet publish (self-contained, 与 build_release 一致) ==' -ForegroundColor Cyan
    dotnet publish (Join-Path $RepoRoot 'src\ZhuLong.App\ZhuLong.App.csproj') `
        -c Release -r win-x64 "-p:Platform=x64" `
        --self-contained `
        -p:WindowsAppSDKSelfContained=false `
        -p:PublishTrimmed=false -p:PublishReadyToRun=false `
        -o $StageDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path (Join-Path $StageDir 'coreclr.dll'))) {
    Write-Host '  fix runtimeconfig (WindowsDesktop.App, framework-dependent only)' -ForegroundColor Cyan
    & (Join-Path $RepoRoot 'scripts\fix_runtimeconfig.ps1') -StageDir $StageDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host '  self-contained publish — skip fix_runtimeconfig' -ForegroundColor Cyan
}

Write-Host '== [6/6] Merge runtime assets (no nested copy) ==' -ForegroundColor Cyan
# data/ 仅打包宏观 JSON；排除 data/training 等大目录
$dataDst = Join-Path $StageDir 'data'
New-Item -ItemType Directory -Force -Path $dataDst | Out-Null
foreach ($df in @('fred_latest.json', 'sentiment.json', 'macro_events.csv', 'macro_calendar.csv')) {
    $src = Join-Path (Join-Path $RepoRoot 'data') $df
    if (Test-Path $src) { Copy-Item -Force $src (Join-Path $dataDst $df) }
}
$macroEventsStaged = Join-Path $dataDst 'macro_events.csv'
if (-not (Test-Path $macroEventsStaged)) { throw 'Staging missing data/macro_events.csv' }
$macroText = Get-Content -LiteralPath $macroEventsStaged -Raw -Encoding UTF8
if ($macroText -notmatch '2026-06-18 02:00,FOMC') {
    throw 'macro_events.csv FOMC 时间错误：应为 2026-06-18 02:00 北京时间'
}
Write-Host '  macro_events.csv: FOMC 2026-06-18 02:00 OK' -ForegroundColor Green
foreach ($sf in @('agent_state_scaler_xauusd.json', 'agent_state_scaler_usoil.json')) {
    $src = Join-Path (Join-Path $RepoRoot 'data') $sf
    if (-not (Test-Path $src)) { throw "Missing agent data: data/$sf" }
    Copy-Item -Force $src (Join-Path $dataDst $sf)
    Write-Host "  agent data: $sf" -ForegroundColor Cyan
}
$macroSrc = Join-Path $RepoRoot 'data\macro'
if (Test-Path $macroSrc) {
    Copy-Item -Recurse -Force $macroSrc (Join-Path $dataDst 'macro')
}

foreach ($item in @('config.json', 'ZhuLong.PythonEngine', 'mql5', 'assets')) {
    $src = Join-Path $RepoRoot $item
    if (-not (Test-Path $src)) { continue }
    if (Test-Path $src -PathType Leaf) {
        Copy-Item -Force $src (Join-Path $StageDir (Split-Path $item -Leaf))
    } else {
        Copy-TreeClean -Src $src -Dst (Join-Path $StageDir (Split-Path $item -Leaf)) -ExcludeDirs @('__pycache__', '.pytest_cache', 'tests')
    }
}
Stage-InstallerModels -DestRoot $StageDir
Stage-ZhulongRuntime -DestRoot $StageDir

Write-Host '== Python agent syntax gate ==' -ForegroundColor Cyan
& (Join-Path $RepoRoot 'scripts\validate_python_agent_bundle.ps1') -Root $StageDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

function Write-WindowsPs1 {
    param([string]$Source, [string]$Dest)
    $raw = [System.IO.File]::ReadAllText($Source)
    $raw = $raw -replace "`r`n", "`n" -replace "`n", "`r`n"
    $enc = New-Object System.Text.UTF8Encoding $true
    [System.IO.File]::WriteAllText($Dest, $raw, $enc)
}

function Write-WindowsBatch {
    param([string]$Source, [string]$Dest)
    $raw = [System.IO.File]::ReadAllText($Source)
    $raw = $raw -replace "`r`n", "`n" -replace "`n", "`r`n"
    $enc = New-Object System.Text.ASCIIEncoding
    [System.IO.File]::WriteAllText($Dest, $raw, $enc)
}

Copy-Item -Force (Join-Path $RepoRoot 'requirements_runtime.txt') (Join-Path $StageDir 'requirements_runtime.txt')
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\install_python_deps.ps1') (Join-Path $StageDir 'install_python_deps.ps1')
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\resolve_system_python.ps1') (Join-Path $StageDir 'resolve_system_python.ps1')
New-Item -ItemType Directory -Force -Path (Join-Path $StageDir 'scripts') | Out-Null
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\check_runtime.ps1') (Join-Path $StageDir 'scripts\check_runtime.ps1')

# System Python mode: drop bundled python_runtime (~200MB installer)
if ($StageDir) {
    $rtDir = Join-Path $StageDir 'python_runtime'
    if ($rtDir -and (Test-Path -LiteralPath $rtDir)) {
        Remove-Item -Recurse -Force $rtDir
        Write-Host '  removed python_runtime from staging (system Python mode)' -ForegroundColor Yellow
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $StageDir 'indicators') | Out-Null
Copy-Item -Force (Join-Path $RepoRoot 'mql5\ZhuLongIndicator.mq5') (Join-Path $StageDir 'indicators\')

if (-not (Test-Path (Join-Path $StageDir 'mql5\Libraries\ZhuLongMt5Pipe.dll'))) {
    throw 'Staging missing mql5/Libraries/ZhuLongMt5Pipe.dll — run build-zhulong-mt5-pipe.ps1 first'
}
$dllSize = (Get-Item (Join-Path $StageDir 'mql5\Libraries\ZhuLongMt5Pipe.dll')).Length
if ($dllSize -lt 10000) {
    throw "Staging ZhuLongMt5Pipe.dll too small ($dllSize bytes)"
}
Write-Host ("  MT5 bridge bundled: ZhuLongMt5Pipe.dll ({0} bytes)" -f $dllSize) -ForegroundColor Green

# 预编译指标 ex5，同事机无需 MetaEditor
$stageMq5 = Join-Path $StageDir 'mql5\ZhuLongIndicator.mq5'
$stageEx5 = Join-Path $StageDir 'mql5\ZhuLongIndicator.ex5'
$metaEditor = @(
    'C:\Program Files\WCG Group MT5 Terminal\metaeditor64.exe',
    'C:\Program Files\MetaTrader 5\metaeditor64.exe',
    'D:\Program Files\MetaTrader 5\metaeditor64.exe'
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($metaEditor -and (Test-Path $stageMq5)) {
    Write-Host '  Precompiling ZhuLongIndicator.ex5 for installer...' -ForegroundColor Cyan
    if (Test-Path $stageEx5) { Remove-Item -Force $stageEx5 }
    $compileProc = Start-Process -FilePath $metaEditor -ArgumentList @("/compile:$stageMq5", '/log') -PassThru -Wait
    if (Test-Path $stageEx5) {
        Copy-Item -Force $stageEx5 (Join-Path $StageDir 'indicators\ZhuLongIndicator.ex5')
        Write-Host ("  ex5 OK ({0} bytes, metaeditor exit={1})" -f (Get-Item $stageEx5).Length, $compileProc.ExitCode) -ForegroundColor Green
    }
    else {
        Write-Warning 'ex5 precompile failed on build machine; end users may need MetaEditor F7'
    }
}
else {
    Write-Warning 'MetaEditor not found — ship mq5 only; run deploy with -Compile on dev machine'
}

Write-WindowsBatch (Join-Path $RepoRoot 'scripts\LaunchZhuLong.cmd') (Join-Path $StageDir 'LaunchZhuLong.cmd')
Write-WindowsBatch (Join-Path $RepoRoot 'scripts\DeployMt5Bridge.cmd') (Join-Path $StageDir 'DeployMt5Bridge.cmd')

$configDst = Join-Path $StageDir 'config'
New-Item -ItemType Directory -Force -Path $configDst | Out-Null
Copy-Item -Force (Join-Path $RepoRoot 'config\causal_graph.yaml') (Join-Path $configDst 'causal_graph.yaml') -ErrorAction SilentlyContinue
Copy-Item -Force (Join-Path $RepoRoot 'config\causal_graph.json') (Join-Path $configDst 'causal_graph.json') -ErrorAction SilentlyContinue
foreach ($cf in @(
        'config_multi_strategy.json', 'config_scheduler.json', 'config_xau_v14.json',
        'config_oil_v14.json', 'config_agent.json'
    )) {
    $src = Join-Path (Join-Path $RepoRoot 'config') $cf
    if (Test-Path $src) { Copy-Item -Force $src (Join-Path $configDst $cf) }
    elseif ($cf -eq 'config_agent.json') { throw "Missing agent config: $src" }
}
$agentCfgPath = Join-Path $configDst 'config_agent.json'
if (Test-Path $agentCfgPath) {
    $agentJson = Get-Content -LiteralPath $agentCfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $agentJson.enabled) {
        $agentJson.enabled = $true
        $agentJson | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $agentCfgPath -Encoding UTF8
        Write-Host '  config_agent.json: forced enabled=true' -ForegroundColor Yellow
    }
}
$scriptsDst = Join-Path $StageDir 'scripts'
New-Item -ItemType Directory -Force -Path $scriptsDst | Out-Null
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\deploy-mt5-indicator.ps1') (Join-Path $scriptsDst 'deploy-mt5-indicator.ps1')
Write-Host '  scripts: runtime only (no training/backtest)' -ForegroundColor Cyan
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\check_runtime.ps1') (Join-Path $scriptsDst 'check_runtime.ps1')
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\fix_runtimeconfig.ps1') (Join-Path $scriptsDst 'fix_runtimeconfig.ps1')
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\install_post_setup.ps1') (Join-Path $scriptsDst 'install_post_setup.ps1')
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\resolve_system_python.ps1') (Join-Path $scriptsDst 'resolve_system_python.ps1')
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\validate_python_agent_bundle.ps1') (Join-Path $scriptsDst 'validate_python_agent_bundle.ps1')
Write-WindowsPs1 (Join-Path $RepoRoot 'scripts\resolve_system_python.ps1') (Join-Path $StageDir 'resolve_system_python.ps1')
$causalCoef = Join-Path $RepoRoot 'models\causal_coef.pkl'
if (Test-Path $causalCoef) {
    Copy-Item -Force $causalCoef (Join-Path $StageDir 'models\causal_coef.pkl')
    $causalMeta = Join-Path $RepoRoot 'models\causal_coef.meta.json'
    if (Test-Path $causalMeta) { Copy-Item -Force $causalMeta (Join-Path $StageDir 'models\causal_coef.meta.json') }
}
# 智能体模型（已训练）
$modelsDir = Join-Path $RepoRoot 'models'
$knPth = Join-Path $modelsDir 'knowledge_net.pth'
$knOnnx = Join-Path $modelsDir 'knowledge_net.onnx'
if ((Test-Path $knPth) -and -not (Test-Path $knOnnx)) {
    Write-Host '  export knowledge_net.onnx (from .pth)' -ForegroundColor Cyan
    $convert = Join-Path $RepoRoot 'scripts\convert_knowledge_net_to_onnx.py'
    if (Test-Path $convert) {
        $py = $env:PYTHON_EXE
        if (-not $py -or -not (Test-Path $py)) { $py = 'python' }
        & $py $convert --model $knPth --out $knOnnx 2>&1 | Out-Host
    }
}
foreach ($agentFile in @('knowledge_net.onnx', 'knowledge_net.meta.json', 'knowledge_scaler.pkl', 'rl_agent_xau.zip')) {
    $src = Join-Path (Join-Path $RepoRoot 'models') $agentFile
    if (-not (Test-Path $src)) {
        throw "Missing agent model: models/$agentFile"
    }
    Copy-Item -Force $src (Join-Path (Join-Path $StageDir 'models') $agentFile)
    Write-Host "  agent model: $agentFile" -ForegroundColor Cyan
}
foreach ($kn2File in @('kn2_trader_v16.pth', 'kn2_trader_v16.meta.json')) {
    $src = Join-Path (Join-Path $RepoRoot 'models') $kn2File
    if (-not (Test-Path $src)) {
        $src = Join-Path (Join-Path $RepoRoot 'data') $kn2File
    }
    if (-not (Test-Path $src)) {
        throw "Missing KN2 V16 model: models/$kn2File (or data/$kn2File)"
    }
    Copy-Item -Force $src (Join-Path (Join-Path $StageDir 'models') $kn2File)
    Write-Host "  KN2 V16 model: $kn2File" -ForegroundColor Cyan
}
foreach ($kn2File in @('kn2_trader.pth', 'kn2_trader.meta.json')) {
    $src = Join-Path (Join-Path $RepoRoot 'models') $kn2File
    if (Test-Path $src) {
        Copy-Item -Force $src (Join-Path (Join-Path $StageDir 'models') $kn2File)
        Write-Host "  KN2 legacy model: $kn2File" -ForegroundColor DarkGray
    }
}
foreach ($v16File in @('horizon_v16.onnx', 'horizon_v16_scaler.pkl', 'horizon_v16.meta.json')) {
    $src = Join-Path (Join-Path $RepoRoot 'models') $v16File
    if (-not (Test-Path $src)) {
        throw "Missing V16 Horizon model: models/$v16File"
    }
    Copy-Item -Force $src (Join-Path (Join-Path $StageDir 'models') $v16File)
    Write-Host "  V16 Horizon: $v16File" -ForegroundColor Cyan
}
Write-Host '== V16 models staged ==' -ForegroundColor Cyan
$rlMeta = Join-Path $RepoRoot 'models\XAUUSD\v16\rl_meta.json'
if (Test-Path $rlMeta) {
    $rlMetaDst = Join-Path (Join-Path $StageDir 'models') 'XAUUSD\v16'
    New-Item -ItemType Directory -Force -Path $rlMetaDst | Out-Null
    Copy-Item -Force $rlMeta (Join-Path $rlMetaDst 'rl_meta.json')
    Write-Host '  V16 RL meta: XAUUSD/v16/rl_meta.json' -ForegroundColor Cyan
}
# USOIL V14 manifest (Stage-InstallerModels already ships inference models)
Copy-Item -Force (Join-Path $RepoRoot 'models\USOIL\manifest.json') (Join-Path (Join-Path $StageDir 'models') 'USOIL\manifest.json') -ErrorAction SilentlyContinue
Copy-Item -Force (Join-Path $RepoRoot 'models\USOIL\acceptance_summary.json') (Join-Path (Join-Path $StageDir 'models') 'USOIL\acceptance_summary.json') -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path (Join-Path $StageDir 'Assets') | Out-Null
$appAssets = Join-Path $RepoRoot 'src\ZhuLong.App\Assets'
Copy-Item -Force (Join-Path $appAssets '*') (Join-Path $StageDir 'Assets\')
Write-Host '  Assets: full icon set copied' -ForegroundColor Cyan

# 安装包仅含运行时 data，剔除误混入的训练/测试大文件
$dataJunk = @('training_data.npz','rl_features_xau.npz','oil_training_data.npz','test1.csv','test1_triple.csv','macro_calendar.csv')
foreach ($jf in $dataJunk) {
    $jp = Join-Path (Join-Path $StageDir 'data') $jf
    if (Test-Path $jp) {
        Remove-Item -Force $jp
        Write-Host "  purged data/$jf" -ForegroundColor Yellow
    }
}

$total = (Get-ChildItem $StageDir -Recurse -File | Measure-Object Length -Sum).Sum
$totalMb = [math]::Round($total / 1MB, 1)
Write-Host ("Staging: {0} MB" -f $totalMb) -ForegroundColor Green
if ($totalMb -gt 450) {
    Write-Warning "Staging ${totalMb} MB - larger than expected ~200-250 MB (check for python_runtime bloat)."
}
if ($totalMb -lt 150) {
    Write-Warning "Staging only ${totalMb} MB - models may be missing."
}

if ($Output -eq 'Folder') {
    Write-Host "Folder only: $StageDir" -ForegroundColor Yellow
    exit 0
}

$iscc = Find-Iscc
if (-not $iscc) {
    Write-Error 'ISCC.exe not found. Install Inno Setup 6.'
}

Get-Process ISCC -ErrorAction SilentlyContinue | Stop-Process -Force
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "== [7/7] Inno Setup: $iscc ==" -ForegroundColor Cyan
$proc = Start-Process -FilePath $iscc -ArgumentList @((Resolve-Path $iss).Path) -Wait -NoNewWindow -PassThru
if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }

$setup = Get-ChildItem $outDir -Filter 'ZhuLong_Setup_v3.1.13.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $setup) {
    $setup = Get-ChildItem $outDir -Filter 'ZhuLong_Setup_v*.exe' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
if ($setup) {
    Write-Host ("Installer OK: {0} ({1} MB)" -f $setup.FullName, [math]::Round($setup.Length / 1MB, 1)) -ForegroundColor Green
}
