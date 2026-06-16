# 下载 Inno 安装包所需的 VC++ / Windows App Runtime（与 V14 同源）
$ErrorActionPreference = 'Stop'
$redist = Join-Path $PSScriptRoot '..\installer\redist'
New-Item -ItemType Directory -Force -Path $redist | Out-Null

$vc = Join-Path $redist 'VC_redist.x64.exe'
$war = Join-Path $redist 'WindowsAppRuntimeInstall-x64.exe'
$dotnet = Join-Path $redist 'windowsdesktop-runtime-8.0-win-x64.exe'

if (-not (Test-Path $vc)) {
    Write-Host 'Download VC++ redist...'
    Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $vc -UseBasicParsing
}

if (-not (Test-Path $war)) {
    Write-Host 'Download Windows App Runtime 1.8 x64...'
    Invoke-WebRequest -Uri 'https://aka.ms/windowsappsdk/1.8/latest/windowsappruntimeinstall-x64.exe' -OutFile $war -UseBasicParsing
}

if (-not (Test-Path $dotnet)) {
    Write-Host 'Download .NET 8 Desktop Runtime x64...'
    $dotnetUrls = @(
        'https://aka.ms/dotnet/8.0/windowsdesktop-runtime-win-x64.exe',
        'https://builds.dotnet.microsoft.com/dotnet/WindowsDesktop/8.0.11/windowsdesktop-runtime-8.0.11-win-x64.exe'
    )
    $ok = $false
    foreach ($u in $dotnetUrls) {
        try {
            Invoke-WebRequest -Uri $u -OutFile $dotnet -UseBasicParsing
            if ((Get-Item $dotnet).Length -gt 10MB) { $ok = $true; break }
        } catch { Write-Warning "dotnet download failed: $u" }
    }
    if (-not $ok) { throw 'Failed to download .NET 8 Desktop Runtime' }
}

foreach ($f in @($vc, $war, $dotnet)) {
    $sz = (Get-Item $f).Length
    if ($sz -lt 500000) { throw "Bad download: $f ($sz bytes)" }
    Write-Host "OK $f ($([math]::Round($sz/1MB,1)) MB)"
}
