# 修复 ZhuLong.runtimeconfig.json：WinUI 3 需要 WindowsDesktop.App + rollForward
param(
    [Parameter(Mandatory = $true)]
    [string] $StageDir
)

$path = Join-Path $StageDir 'ZhuLong.runtimeconfig.json'
if (-not (Test-Path $path)) {
    Write-Warning "fix_runtimeconfig: missing $path"
    exit 0
}

if (Test-Path (Join-Path $StageDir 'coreclr.dll')) {
    Write-Host "  runtimeconfig: self-contained app, no patch needed" -ForegroundColor Green
    exit 0
}

$raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
if ($raw.runtimeOptions.includedFrameworks) {
    Write-Host "  runtimeconfig: includedFrameworks (self-contained), no patch needed" -ForegroundColor Green
    exit 0
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
Write-Host "  runtimeconfig: WindowsDesktop.App + LatestPatch" -ForegroundColor Green
