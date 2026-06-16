# 将本地仓库推送到 GitHub（需先在网页创建空仓库）
# https://github.com/new  -> 名称 ZhuLong3_DevPackage_v2 -> 不要勾选 README
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$remote = "https://github.com/dazhupan5-arch/ZhuLong3_DevPackage_v2.git"
if (-not (git remote get-url origin 2>$null)) {
    git remote add origin $remote
} else {
    git remote set-url origin $remote
}

Write-Host "Pushing to $remote ..."
git push -u origin main
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done: $remote" -ForegroundColor Green
} else {
    Write-Host @"
Push 失败（网络/鉴权）。可选方案：
  1. 配置 GitHub 凭据后重跑本脚本
  2. 把上级目录 ZhuLong3_DevPackage_v2.bundle 拷到 GPU 机：
       git clone ZhuLong3_DevPackage_v2.bundle ZhuLong3_DevPackage_v2
       cd ZhuLong3_DevPackage_v2
       git remote add origin $remote
       git push -u origin main
"@ -ForegroundColor Yellow
    exit $LASTEXITCODE
}
