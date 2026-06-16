# Deploy retrained models: KnowledgeNet + RL
$ErrorActionPreference = "Continue"

$root = "D:\ZhuLong3_Migration_20260609.zip"
$deploy = "d:\Program Files\ZhuLong"
$models = "$deploy\models"

# Step 1: Kill ZhuLong
Write-Host "=== Killing ZhuLong.exe ==="
$p = Get-Process -Name ZhuLong -ErrorAction SilentlyContinue
if ($p) { $p.Kill(); Start-Sleep 2 }

# Step 2: Copy KnowledgeNet XAUUSD (PTH + ONNX + META + SCALER)
Write-Host "=== Deploy KnowledgeNet XAUUSD ==="
if (Test-Path "$root\models\knowledge_net.pth") {
    Copy-Item -Force "$root\models\knowledge_net.pth" "$models\knowledge_net.pth"
    Copy-Item -Force "$root\models\knowledge_net.meta.json" "$models\knowledge_net.meta.json"
    Copy-Item -Force "$root\models\knowledge_scaler.pkl" "$models\knowledge_scaler.pkl"
    if (Test-Path "$root\models\knowledge_net.onnx") {
        Copy-Item -Force "$root\models\knowledge_net.onnx" "$models\knowledge_net.onnx"
        Write-Host "ONNX exported and deployed"
    }
    Write-Host "XAUUSD KnowledgeNet deployed"
} else { Write-Host "WARNING: XAUUSD KnowledgeNet .pth not found" }

# Step 3: Copy KnowledgeNet USOIL
Write-Host "=== Deploy KnowledgeNet USOIL ==="
if (Test-Path "$root\models\knowledge_net_oil.pth") {
    Copy-Item -Force "$root\models\knowledge_net_oil.pth" "$models\knowledge_net_oil.pth"
    Copy-Item -Force "$root\models\knowledge_net_oil.meta.json" "$models\knowledge_net_oil.meta.json"
    Copy-Item -Force "$root\models\knowledge_scaler_oil.pkl" "$models\knowledge_scaler_oil.pkl"
    Write-Host "USOIL KnowledgeNet deployed"
} else { Write-Host "WARNING: USOIL KnowledgeNet .pth not found" }

# Step 4: Copy RL Agent XAUUSD
Write-Host "=== Deploy RL Agent ==="
if (Test-Path "$root\models\rl_agent_xau.zip") {
    Copy-Item -Force "$root\models\rl_agent_xau.zip" "$models\rl_agent_xau.zip"
    Write-Host "RL Agent XAUUSD deployed"
} else { Write-Host "WARNING: RL Agent .zip not found" }
if (Test-Path "$root\models\rl_agent_oil.zip") {
    Copy-Item -Force "$root\models\rl_agent_oil.zip" "$models\rl_agent_oil.zip"
    Write-Host "RL Agent USOIL deployed"
} else { Write-Host "WARNING: RL Agent USOIL .zip not found" }

# Step 5: Copy state scaler for RL
if (Test-Path "$root\data\agent_state_scaler_xauusd.json") {
    Copy-Item -Force "$root\data\agent_state_scaler_xauusd.json" "$deploy\data\agent_state_scaler_xauusd.json"
    Write-Host "State scaler deployed"
}

# Step 6: Enable trading agent in config
Write-Host "=== Enable TradingAgent in config ==="
$configPath = "$deploy\config.json"
if (Test-Path $configPath) {
    $cfg = Get-Content $configPath -Raw | ConvertFrom-Json
    $cfg.trading_agent.enabled = $true
    $cfg | ConvertTo-Json -Depth 10 | Set-Content $configPath -Encoding UTF8
    Write-Host "trading_agent.enabled = true"
}

# Step 7: Copy zhulong/engine (缺失时补全)
Write-Host "=== Copy zhulong/engine Python module ==="
$engineSrc = "$root\zhulong\engine"
$engineDst = "$deploy\zhulong\engine"
if (Test-Path $engineSrc) {
    if (-not (Test-Path $engineDst)) { New-Item -ItemType Directory -Force -Path $engineDst | Out-Null }
    Copy-Item -Force "$engineSrc\*.py" $engineDst
    Write-Host "zhulong/engine copied"
}

# Step 8: Build C# to pick up any code changes
Write-Host "=== Build C# ==="
$msbuild = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
if (-not (Test-Path $msbuild)) {
    $msbuild = "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
}
if (-not (Test-Path $msbuild)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
        if ($vsPath) { $msbuild = Join-Path $vsPath "MSBuild\Current\Bin\MSBuild.exe" }
    }
}
Set-Location "$root\src"
dotnet build -c Release /p:Platform=x64 --runtime win-x64 /p:SelfContained=false ZhuLong.App\ZhuLong.App.csproj 2>&1 | Select-String 'error CS|Build succeeded'
$buildOut = "$root\src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64"
# 仅复制编译产物（exe/dll/pdb/pri/runtimeconfig/deps），不覆盖模型/Python目录
foreach ($ext in @('.exe', '.dll', '.pdb', '.pri', '.runtimeconfig.json', '.deps.json')) {
    $src = Join-Path $buildOut "ZhuLong$ext"
    if (Test-Path $src) { Copy-Item -Force $src $deploy }
    $src = Join-Path $buildOut "ZhuLong.Core$ext"
    if (Test-Path $src) { Copy-Item -Force $src $deploy }
}
Copy-Item -Force "$buildOut\config.json" "$deploy\config.json"
Write-Host "C# binaries deployed (models/Python preserved)"

# Step 9: Copy zhulong Python modules (确保最新版本)
Write-Host "=== Copy zhulong Python modules ==="
if (Test-Path "$root\zhulong") {
    Copy-Item -Recurse -Force "$root\zhulong\agent" "$deploy\zhulong\"
    Copy-Item -Recurse -Force "$root\zhulong\engine" "$deploy\zhulong\"
    Copy-Item -Recurse -Force "$root\zhulong\utils" "$deploy\zhulong\"
    Write-Host "zhulong Python modules deployed"
}

# Step 10: Also deploy DLL + MQL5 (ensure latest)
$mt5Base = "$env:AppData\MetaQuotes\Terminal"
$mt5Dirs = Get-ChildItem -Path $mt5Base -Directory | Where-Object { $_.Name -ne 'Common' }
$targetMql5 = $null
foreach ($dir in $mt5Dirs) {
    $indicatorPath = Join-Path $dir.FullName "MQL5\Indicators"
    if (Test-Path $indicatorPath) { $targetMql5 = $dir.FullName; break }
}
if (-not $targetMql5) {
    $specific = Join-Path $mt5Base "A52E191D2FFA25A9AE2E3FB78CEE38D2"
    if (Test-Path $specific) { $targetMql5 = $specific }
}
if ($targetMql5) {
    $dllSrc = "$root\native\ZhuLongMt5Pipe\Release\ZhuLongMt5Pipe.dll"
    if (-not (Test-Path $dllSrc)) { $dllSrc = "$root\mql5\Libraries\ZhuLongMt5Pipe.dll" }
    if (Test-Path $dllSrc) { Copy-Item -Force $dllSrc "$targetMql5\MQL5\Libraries\ZhuLongMt5Pipe.dll"; Write-Host "DLL deployed" }
    Copy-Item -Force "$root\mql5\ZhuLongIndicator.mq5" "$targetMql5\MQL5\Indicators\ZhuLongIndicator.mq5"
    Write-Host "MQL5 deployed"
}

# Step 9: Restart
Write-Host "=== Starting ZhuLong.exe ==="
Start-Process -FilePath "$deploy\ZhuLong.exe" -WorkingDirectory "$deploy"
Write-Host "=== DEPLOYMENT COMPLETE ==="
Write-Host ""
Write-Host "Next: In MT5, reload ZhuLongIndicator"
