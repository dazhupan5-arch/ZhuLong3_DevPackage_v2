# 备份 %APPDATA%\ZhuLong\trading.db
$ErrorActionPreference = "Stop"
$src = Join-Path $env:APPDATA "ZhuLong\trading.db"
if (-not (Test-Path $src)) {
    Write-Warning "数据库不存在: $src"
    exit 1
}
$destDir = Join-Path $env:APPDATA "ZhuLong\backups"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$dest = Join-Path $destDir "trading_$stamp.db"
Copy-Item -Path $src -Destination $dest -Force
Write-Host "已备份 -> $dest"
