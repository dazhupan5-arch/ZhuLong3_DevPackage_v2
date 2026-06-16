# 发布前裁剪 python_runtime：去掉 torch C++ 头文件等推理不需要的内容（显著加速 ISCC）
param(
    [string] $StagingRoot = (Join-Path $PSScriptRoot "..\publish\win-x64")
)

$ErrorActionPreference = "Stop"
$site = Join-Path $StagingRoot "python_runtime\Lib\site-packages"
if (-not (Test-Path $site)) {
    Write-Host "skip: no python_runtime in staging"
    exit 0
}

$removeDirs = @(
    (Join-Path $site "torch\include"),
    (Join-Path $site "torch\share"),
    (Join-Path $site "sympy"),
    (Join-Path $site "networkx"),
    (Join-Path $site "mpmath"),
    (Join-Path $site "jinja2"),
    (Join-Path $site "fsspec"),
    (Join-Path $site "filelock"),
    (Join-Path $site "functorch"),
    (Join-Path $site "torchgen"),
    (Join-Path $site "narwhals")
)

foreach ($d in $removeDirs) {
    if (Test-Path $d) {
        Remove-Item -Recurse -Force $d
        Write-Host "removed dir: $($d.Replace($StagingRoot, ''))"
    }
}

$lib = Join-Path $site "torch\lib"
if (Test-Path $lib) {
    Get-ChildItem $lib -Filter "*.lib" -File | Remove-Item -Force
    Write-Host "removed torch/lib/*.lib"
}

Get-ChildItem $site -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$total = (Get-ChildItem $StagingRoot -Recurse -File | Measure-Object Length -Sum).Sum
$fileCount = (Get-ChildItem $StagingRoot -Recurse -File | Measure-Object).Count
Write-Host "staging after trim: $([math]::Round($total/1MB,1)) MB, $fileCount files"
