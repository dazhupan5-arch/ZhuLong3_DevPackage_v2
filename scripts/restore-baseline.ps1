# 烛龙 ZhuLong_3 — 回滚到 v1.0.22 基线
#Requires -Version 5.1
param(
    [string] $BackupDir = "D:\trae_projects\_backups\ZhuLong_3_v1.0.22_baseline_20260609",
    [string] $TargetDir = "D:\trae_projects\ZhuLong_3",
    [switch] $Force
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $BackupDir)) {
    Write-Error "找不到基线备份: $BackupDir"
}

$manifest = Join-Path $BackupDir 'BASELINE_MANIFEST.json'
if (Test-Path $manifest) {
    Write-Host "基线: $(Get-Content $manifest -Raw)" -ForegroundColor Cyan
}

if (-not $Force) {
    Write-Host ""
    Write-Host "将把以下内容覆盖到: $TargetDir" -ForegroundColor Yellow
    Write-Host "  来源: $BackupDir"
    Write-Host ""
    $confirm = Read-Host "确认回滚? 输入 yes 继续"
    if ($confirm -ne 'yes') {
        Write-Host "已取消。"
        exit 0
    }
}

Write-Host "结束 ZhuLong 进程…" -ForegroundColor Cyan
Get-Process ZhuLong -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$failed = "${TargetDir}_before_rollback_$stamp"
if (Test-Path $TargetDir) {
    Write-Host "当前目录重命名为: $failed" -ForegroundColor Cyan
    Rename-Item -LiteralPath $TargetDir -NewName (Split-Path $failed -Leaf)
}

Write-Host "复制基线…" -ForegroundColor Cyan
Copy-Item -LiteralPath $BackupDir -Destination $TargetDir -Recurse -Force

Write-Host ""
Write-Host "回滚完成。" -ForegroundColor Green
Write-Host "  工作目录: $TargetDir"
Write-Host "  被替换的旧版: $failed"
Write-Host ""
Write-Host "下一步: 在 $TargetDir 运行 pack-installer 或直接使用 output 中的 v1.0.22 安装包。" -ForegroundColor Green
