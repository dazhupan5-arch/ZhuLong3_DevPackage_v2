# Build & Deploy script for ZhuLong Drawing Pipe Fix
# Part of the plan: simplify-csharp -> build-dll -> build-csharp -> deploy -> restart

$ErrorActionPreference = "Stop"
$root = "D:\ZhuLong3_Migration_20260609.zip"

# Step 1: Kill running ZhuLong
Write-Host "=== Step 0: Kill ZhuLong.exe ==="
$p = Get-Process -Name ZhuLong -ErrorAction SilentlyContinue
if ($p) { $p.Kill(); Start-Sleep 3; Write-Host "ZhuLong.exe killed" }
else { Write-Host "ZhuLong.exe not running" }

# Step 2: Build DLL
Write-Host "=== Step 1: Build DLL ==="
$msbuild = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
if (-not (Test-Path $msbuild)) {
    # try alternate paths
    $msbuild = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
}
if (-not (Test-Path $msbuild)) {
    Write-Host "MSBuild not found at expected paths!"
    # Try vswhere
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
        if ($vsPath) {
            $msbuild = Join-Path $vsPath "MSBuild\Current\Bin\MSBuild.exe"
        }
    }
}
Write-Host "MSBuild: $msbuild"
& $msbuild "$root\native\ZhuLongMt5Pipe\ZhuLongMt5Pipe.vcxproj" /p:Configuration=Release /p:Platform=x64 /v:minimal
if ($LASTEXITCODE -ne 0) { throw "DLL build failed with exit code $LASTEXITCODE" }
Write-Host "DLL build OK"

# Step 3: Build C#
Write-Host "=== Step 2: Build C# ==="
Set-Location "$root\src"
dotnet build -c Release /p:Platform=x64 --runtime win-x64 /p:SelfContained=false ZhuLong.App\ZhuLong.App.csproj 2>&1
if ($LASTEXITCODE -ne 0) { throw "C# build failed with exit code $LASTEXITCODE" }
Write-Host "C# build OK"

# Step 4: Deploy DLL to all MT5 terminals (portable + AppData)
Write-Host "=== Step 3: Deploy DLL + MQL5 to MT5 ==="
& (Join-Path $root 'scripts\deploy-mt5-indicator.ps1') -StopMt5First -Compile
if ($LASTEXITCODE -ne 0) { throw "MT5 indicator deploy failed with exit code $LASTEXITCODE" }
Write-Host "DLL/MQL5 deploy OK"

# Step 5: Deploy C# to Program Files
Write-Host "=== Step 4: Deploy C# to Program Files ==="
$buildOut = "$root\src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64"
if (-not (Test-Path $buildOut)) {
    # try alternate path
    $buildOut = "$root\src\out"
}
if (-not (Test-Path $buildOut)) { throw "Build output not found at $buildOut" }
Write-Host "Build output: $buildOut"
robocopy "$buildOut" "d:\Program Files\ZhuLong" /MIR /XF '*.log' /XF '*.pdb' /XF '*.db' /XF '*.bak' /XD 'logs' /np /njh /njs /nc /mt:8
# 确保 MT5 桥接文件在安装目录（build 输出应已含 mql5\，此处双保险）
if (Test-Path "$root\mql5") {
    robocopy "$root\mql5" "d:\Program Files\ZhuLong\mql5" /E /XF '*.exp' /XF '*.lib' /np /njh /njs /nc /mt:8
}
Write-Host "C# deploy OK"

# Step 6: Copy meta.json
Write-Host "=== Copy meta.json ==="
Copy-Item -Force "d:\Program Files\ZhuLong\models\knowledge_net.meta.json" "$buildOut\models\knowledge_net.meta.json"

# Step 7: Restart
Write-Host "=== Step 6: Restart ZhuLong.exe ==="
Start-Process -FilePath "d:\Program Files\ZhuLong\ZhuLong.exe" -WorkingDirectory "d:\Program Files\ZhuLong"
Write-Host "=== ALL DONE ==="

Write-Host ""
Write-Host "IMPORTANT: After deployment, please reload the MT5 indicator:"
Write-Host "  1. In MT5, right-click on chart -> Indicators List"
Write-Host "  2. Select ZhuLongIndicator -> Remove"
Write-Host "  3. Re-add ZhuLongIndicator from Navigator"
