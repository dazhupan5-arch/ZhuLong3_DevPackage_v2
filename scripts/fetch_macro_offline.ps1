# 宏观离线数据拉取：FRED + LLM 情绪
Set-Location $PSScriptRoot\..

Write-Host "[1/2] FRED ..."
py -3 ZhuLong.PythonEngine/fetch_fred.py
if ($LASTEXITCODE -ne 0) { Write-Warning "FRED 拉取未完成（可稍后重试）" }

Write-Host "[2/2] LLM 情绪 ..."
py -3 ZhuLong.PythonEngine/fetch_sentiment.py
if ($LASTEXITCODE -ne 0) { Write-Warning "情绪分析未完成" }

Write-Host "完成。C# 启动时会读取 data/fred_latest.json 与 data/sentiment.json"
