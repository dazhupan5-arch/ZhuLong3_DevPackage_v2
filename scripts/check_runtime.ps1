# 烛龙启动前运行环境检测（扫描本机 .NET / WinUI，与安装盘无关）

param(

    [string] $InstallDir = '',

    [switch] $AutoRepair,

    [switch] $Quiet

)



$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($InstallDir)) {

    $InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path

    if ((Split-Path -Leaf $InstallDir) -eq 'scripts') {

        $InstallDir = Split-Path -Parent $InstallDir

    }

}

$InstallDir = $InstallDir.TrimEnd('\')

if (-not (Test-Path -LiteralPath $InstallDir)) {

    Write-Error "InstallDir not found: $InstallDir"

    exit 1

}

$InstallDir = (Get-Item -LiteralPath $InstallDir).FullName

$RedistDir = Join-Path $InstallDir 'redist'

$AppDataZhuLong = Join-Path $env:APPDATA 'ZhuLong'

function Test-SelfContainedApp {
    param([string] $Dir)
    return (Test-Path -LiteralPath (Join-Path $Dir 'coreclr.dll'))
}

$isSelfContained = Test-SelfContainedApp -Dir $InstallDir



function Get-DotNetInstallRoots {

    $roots = [ordered]@{}

    function Add-Root([string] $Path) {

        if ([string]::IsNullOrWhiteSpace($Path)) { return }

        try {

            $full = (Get-Item -LiteralPath $Path).FullName

        } catch { return }

        if (-not (Test-Path -LiteralPath (Join-Path $full 'dotnet.exe'))) { return }

        if (-not $roots.Contains($full)) { $roots[$full] = $true }

    }



    Add-Root $env:DOTNET_ROOT

    $cache = Join-Path $AppDataZhuLong 'dotnet_root.txt'

    if (Test-Path -LiteralPath $cache) { Add-Root (Get-Content -LiteralPath $cache -Raw).Trim() }



    foreach ($reg in @(

            'HKLM:\SOFTWARE\dotnet\Setup\InstalledVersions\x64',

            'HKLM:\SOFTWARE\WOW6432Node\dotnet\Setup\InstalledVersions\x64'

        )) {

        if (Test-Path $reg) {

            $loc = (Get-ItemProperty -LiteralPath $reg -ErrorAction SilentlyContinue).InstallLocation

            Add-Root $loc

        }

    }



    foreach ($pf in @(${env:ProgramFiles}, ${env:ProgramFiles(x86)})) {

        if ($pf) { Add-Root (Join-Path $pf 'dotnet') }

    }



    try {

        $where = @(where.exe dotnet 2>$null | Select-Object -First 1)

        if ($where.Count -gt 0) { Add-Root (Split-Path -Parent $where[0]) }

    } catch { }



    return @($roots.Keys)

}



function Get-SharedFxVersions {

    param([string] $InstallRoot, [string] $Framework, [string] $MajorPrefix)

    $dir = Join-Path (Join-Path $InstallRoot 'shared') $Framework

    if (-not (Test-Path -LiteralPath $dir)) { return @() }

    return @(Get-ChildItem -LiteralPath $dir -Directory -ErrorAction SilentlyContinue |

        Where-Object { $_.Name -like "$MajorPrefix.*" } |

        Select-Object -ExpandProperty Name |

        Sort-Object -Unique)

}



function Test-DotNet8Desktop {

    foreach ($root in (Get-DotNetInstallRoots)) {

        $desktop = Get-SharedFxVersions -InstallRoot $root -Framework 'Microsoft.WindowsDesktop.App' -MajorPrefix '8.0'

        if ($desktop.Count -gt 0) {

            return @{ Ok = $true; Root = $root; Detail = "根目录 $root；Desktop: $($desktop -join ', ')" }

        }

    }



    foreach ($root in (Get-DotNetInstallRoots)) {

        try {

            $dotnet = Join-Path $root 'dotnet.exe'

            $listed = @(& $dotnet --list-runtimes 2>$null |

                Where-Object { $_ -match 'Microsoft\.WindowsDesktop\.App 8\.0\.' })

            if ($listed.Count -gt 0) {

                return @{ Ok = $true; Root = $root; Detail = ($listed -join '; ') }

            }

        } catch { }

    }



    foreach ($root in (Get-DotNetInstallRoots)) {

        $core = Get-SharedFxVersions -InstallRoot $root -Framework 'Microsoft.NETCore.App' -MajorPrefix '8.0'

        if ($core.Count -gt 0) {

            return @{

                Ok = $false; Root = $root

                Detail = "根目录 $root 仅有 .NET Core ($($core -join ', '))，缺少 WindowsDesktop 桌面框架"

            }

        }

    }



    return @{ Ok = $false; Root = ''; Detail = '未安装 .NET 8 Desktop Runtime' }

}



function Repair-RuntimeConfig {

    param([string] $Dir)

    $path = Join-Path $Dir 'ZhuLong.runtimeconfig.json'

    if (-not (Test-Path -LiteralPath $path)) { return $false }

    try {

        $raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json

        $frameworks = @($raw.runtimeOptions.frameworks)

        $hasDesktop = @($frameworks | Where-Object { $_.name -eq 'Microsoft.WindowsDesktop.App' }).Count -gt 0

        if ($hasDesktop -and $raw.runtimeOptions.rollForward -eq 'LatestPatch') { return $false }

        $fixScript = Join-Path $Dir 'scripts\fix_runtimeconfig.ps1'

        if (Test-Path -LiteralPath $fixScript) {

            & $fixScript -StageDir $Dir

            return $true

        }

        $configProps = @{}

        if ($raw.runtimeOptions.configProperties) {

            $raw.runtimeOptions.configProperties.PSObject.Properties | ForEach-Object {

                $configProps[$_.Name] = $_.Value

            }

        }

        $fixed = [ordered]@{

            runtimeOptions = [ordered]@{

                tfm         = 'net8.0'

                rollForward = 'LatestPatch'

                frameworks  = @(

                    @{ name = 'Microsoft.NETCore.App'; version = '8.0.0' }

                    @{ name = 'Microsoft.WindowsDesktop.App'; version = '8.0.0' }

                )

                configProperties = $configProps

            }

        }

        $json = ($fixed | ConvertTo-Json -Depth 8)

        [System.IO.File]::WriteAllText($path, $json, (New-Object System.Text.UTF8Encoding $false))

        return $true

    } catch {

        Write-Warning "Repair-RuntimeConfig failed: $_"

        return $false

    }

}



if (Repair-RuntimeConfig -Dir $InstallDir) {

    Write-Host "  runtimeconfig 已修复 (WindowsDesktop.App + LatestPatch)" -ForegroundColor Yellow

}



function Test-WinAppRuntimeAppxStore {

    $root = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Appx\AppxAllUserStore\Applications'

    if (-not (Test-Path $root)) { return $null }

    foreach ($app in Get-ChildItem $root -ErrorAction SilentlyContinue) {

        foreach ($sub in Get-ChildItem $app.PSPath -ErrorAction SilentlyContinue) {

            $name = $sub.PSChildName

            if ($name -like 'Microsoft.WindowsAppRuntime.*' -or

                $name -like 'MicrosoftCorporationII.WinAppRuntime.Main.*') {

                return $name

            }

        }

    }

    return $null

}



function Test-WinAppRuntime {

    $paths = @(

        'HKLM:\SOFTWARE\Microsoft\WindowsAppRuntime\Installed',

        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\WindowsAppRuntime\Installed'

    )

    foreach ($p in $paths) {

        if (Test-Path $p) {

            $ver = (Get-ItemProperty $p -ErrorAction SilentlyContinue).Version

            $detail = if ($ver) { "注册表已注册 ($ver)" } else { '注册表已注册' }

            return @{ Ok = $true; Detail = $detail }

        }

    }



    $appxPkg = Test-WinAppRuntimeAppxStore

    if ($appxPkg) {

        return @{ Ok = $true; Detail = "AppX 已安装 ($appxPkg)" }

    }



    $windowsApps = Join-Path ${env:ProgramFiles} 'WindowsApps'

    if (Test-Path $windowsApps) {

        foreach ($pattern in @('Microsoft.WindowsAppRuntime.*', 'MicrosoftCorporationII.WinAppRuntime.Main.*')) {

            $hit = Get-ChildItem $windowsApps -Directory -Filter $pattern -ErrorAction SilentlyContinue |

                Select-Object -First 1

            if ($hit) { return @{ Ok = $true; Detail = "WindowsApps: $($hit.Name)" } }

        }

    }



    foreach ($hive in @(

            'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',

            'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*'

        )) {

        $hit = Get-ItemProperty $hive -ErrorAction SilentlyContinue |

            Where-Object { $_.DisplayName -match 'Windows App Runtime|WindowsAppRuntime' } |

            Select-Object -First 1

        if ($hit) {

            $ver = if ($hit.DisplayVersion) { " $($hit.DisplayVersion)" } else { '' }

            return @{ Ok = $true; Detail = "卸载项: $($hit.DisplayName)$ver" }

        }

    }



    try {

        $prov = @(Get-AppxProvisionedPackage -Online -ErrorAction Stop |

            Where-Object { $_.DisplayName -match 'WindowsAppRuntime|WinAppRuntime\.Main' })

        if ($prov.Count -gt 0) {

            $top = $prov | Sort-Object { [version]$_.Version } -Descending | Select-Object -First 1

            return @{ Ok = $true; Detail = "AppX 预装 ($($top.DisplayName) $($top.Version))" }

        }

    } catch { }



    return @{ Ok = $false; Detail = '未安装 Windows App Runtime (WinUI 3)' }

}



function Test-VcRedist {

    $paths = @(

        'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',

        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'

    )

    foreach ($p in $paths) {

        if (Test-Path $p) { return @{ Ok = $true; Detail = '已安装' } }

    }

    return @{ Ok = $false; Detail = '未检测到 VC++ 2015-2022 x64' }

}



function Install-Redist {

    param([string] $Exe, [string] $InstallArgs = '')

    if (-not (Test-Path $Exe)) { return $false }

    Write-Host "正在安装: $Exe" -ForegroundColor Cyan

    $startParams = @{

        FilePath     = $Exe

        Wait         = $true

        PassThru     = $true

        Verb         = 'RunAs'

    }

    if (-not [string]::IsNullOrWhiteSpace($InstallArgs)) {

        $startParams['ArgumentList'] = $InstallArgs

    }

    $p = Start-Process @startParams

    return $p.ExitCode -eq 0 -or $p.ExitCode -eq 1638 -or $p.ExitCode -eq 3010

}



$dotnetProbe = if ($isSelfContained) {
    @{ Ok = $true; Root = $InstallDir; Detail = '自包含发布 (coreclr.dll 已内置)' }
} else {
    Test-DotNet8Desktop
}

if ($dotnetProbe.Ok -and $dotnetProbe.Root) {

    $env:DOTNET_ROOT = $dotnetProbe.Root

    $env:DOTNET_ROLL_FORWARD = 'LatestPatch'

    $env:DOTNET_MULTILEVEL_LOOKUP = '1'

    if ($env:PATH -notlike "*$($dotnetProbe.Root)*") {

        $env:PATH = "$($dotnetProbe.Root);$($env:PATH)"

    }

    New-Item -ItemType Directory -Force -Path $AppDataZhuLong | Out-Null

    Set-Content -LiteralPath (Join-Path $AppDataZhuLong 'dotnet_root.txt') -Value $dotnetProbe.Root -Encoding ASCII

}



$checks = [ordered]@{

    '.NET 8 Desktop Runtime' = $dotnetProbe

    'Windows App Runtime (WinUI 3)' = (Test-WinAppRuntime)

    'Visual C++ 2015-2022 x64' = (Test-VcRedist)

}



$missing = @($checks.GetEnumerator() | Where-Object { -not $_.Value.Ok })



if ($AutoRepair -and $missing.Count -gt 0) {

    if (-not $checks['.NET 8 Desktop Runtime'].Ok) {

        $exe = Join-Path $RedistDir 'windowsdesktop-runtime-8.0-win-x64.exe'

        if (Install-Redist $exe '/install /quiet /norestart') {

            $checks['.NET 8 Desktop Runtime'] = Test-DotNet8Desktop

            if ($checks['.NET 8 Desktop Runtime'].Ok -and $checks['.NET 8 Desktop Runtime'].Root) {

                $env:DOTNET_ROOT = $checks['.NET 8 Desktop Runtime'].Root

                Set-Content -LiteralPath (Join-Path $AppDataZhuLong 'dotnet_root.txt') -Value $env:DOTNET_ROOT -Encoding ASCII

            }

        }

    }

    if (-not ($checks['Windows App Runtime (WinUI 3)'].Ok)) {

        $exe = Join-Path $RedistDir 'WindowsAppRuntimeInstall-x64.exe'

        if (Install-Redist $exe '--quiet') {

            Start-Sleep -Seconds 3

            $checks['Windows App Runtime (WinUI 3)'] = (Test-WinAppRuntime)

        }

    }

    if (-not ($checks['Visual C++ 2015-2022 x64'].Ok)) {

        $exe = Join-Path $RedistDir 'VC_redist.x64.exe'

        Install-Redist $exe '/install /passive /norestart' | Out-Null

        $checks['Visual C++ 2015-2022 x64'] = (Test-VcRedist)

    }

    $missing = @($checks.GetEnumerator() | Where-Object { -not $_.Value.Ok })

}



New-Item -ItemType Directory -Force -Path $AppDataZhuLong | Out-Null

$logPath = Join-Path $AppDataZhuLong 'startup.log'

$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

$lines = @("[$stamp] [check_runtime] InstallDir=$InstallDir DOTNET_ROOT=$($env:DOTNET_ROOT)")

foreach ($kv in $checks.GetEnumerator()) {

    $flag = if ($kv.Value.Ok) { 'OK' } else { 'MISS' }

    $detail = if ($kv.Value.Detail) { $kv.Value.Detail } else { $kv.Value.Root }

    $lines += "[$stamp] [check_runtime] $flag $($kv.Key): $detail"

}

Add-Content -LiteralPath $logPath -Value ($lines -join [Environment]::NewLine)



if ($missing.Count -eq 0) {

    if (-not $Quiet) { Write-Host '运行环境就绪' -ForegroundColor Green }

    exit 0

}



$msg = @(

    '烛龙检测到运行环境不完整，无法启动 ZhuLong.exe。',

    '',

    '缺少或未识别：'

) + ($missing | ForEach-Object { "  • $($_.Key) — $($_.Value.Detail)" }) + @(

    '',

    "安装目录：$InstallDir",

    "可手动运行 redist 目录下对应安装程序，或重新运行 ZhuLong_Setup。",

    '',

    '说明：.NET 扫描与安装盘无关；若已安装仍提示缺失，请查看 %APPDATA%\ZhuLong\startup.log 中的 DOTNET_ROOT。'

)



if (-not $Quiet) {

    Add-Type -AssemblyName System.Windows.Forms

    [System.Windows.Forms.MessageBox]::Show(

        ($msg -join [Environment]::NewLine),

        '烛龙 ZhuLong — 运行环境',

        [System.Windows.Forms.MessageBoxButtons]::OK,

        [System.Windows.Forms.MessageBoxIcon]::Warning) | Out-Null

}



exit 1

