# V17 config_agent.json 契约合并：以 config/config_agent_v17.json 为权威。
# 保留 AppData 中 enabled / primary_symbol；V17 路径下 KN2 默认 shadow/off。

. (Join-Path $PSScriptRoot "Merge-V16AgentConfig.ps1")

function Merge-V17AgentConfig {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [string]$SourcePath = "",
        [double]$DirectionMinScore = 0.0
    )

    $devRoot = Split-Path $PSScriptRoot -Parent
    if ([string]::IsNullOrWhiteSpace($SourcePath)) {
        $SourcePath = Join-Path $devRoot "config\config_agent_v17.json"
    }
    if (-not (Test-Path $SourcePath)) {
        throw "Missing V17 source config: $SourcePath"
    }

    Merge-V16AgentConfig -TargetPath $TargetPath -SourcePath $SourcePath

    $cfg = Get-Content $TargetPath -Raw -Encoding UTF8 | ConvertFrom-Json

    if (-not $cfg.architecture) {
        $cfg | Add-Member -NotePropertyName architecture -NotePropertyValue (@{}) -Force
    }
    $cfg.architecture.version = "v17"

    if ($DirectionMinScore -gt 0) {
        if ($cfg.architecture.direction_scorer) {
            $cfg.architecture.direction_scorer.min_abs_score = $DirectionMinScore
        }
        if ($cfg.trader_mind) {
            $cfg.trader_mind.min_confidence = $DirectionMinScore
        }
        if ($cfg.cognition) {
            $cfg.cognition.direction_threshold = $DirectionMinScore
        }
    }

    $json = $cfg | ConvertTo-Json -Depth 25
    [System.IO.File]::WriteAllText($TargetPath, $json, [System.Text.UTF8Encoding]::new($false))

    $ds = if ($cfg.architecture.direction_scorer.model_path) { $cfg.architecture.direction_scorer.model_path } else { "(missing)" }
    $lg = if ($cfg.architecture.location_gate.model_path) { $cfg.architecture.location_gate.model_path } else { "(missing)" }
    Write-Host "Merged V17 config_agent.json → $TargetPath" -ForegroundColor Green
    Write-Host "  architecture.version=v17 direction_scorer=$ds location_gate=$lg" -ForegroundColor Cyan
}
