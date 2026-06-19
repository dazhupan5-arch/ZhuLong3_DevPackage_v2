# V16 config_agent.json 契约合并：以仓库 config 为权威，保留 AppData 中 KN2 LIVE 状态。
# 禁止 deploy 脚本用精简模板覆写，导致 execution_composer / trading_env 丢失。

function Merge-V16AgentConfig {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [string]$SourcePath = "",
        [double]$HorizonMinConfidence = 0.0
    )

    $devRoot = Split-Path $PSScriptRoot -Parent
    if ([string]::IsNullOrWhiteSpace($SourcePath)) {
        $SourcePath = Join-Path $devRoot "config\config_agent.json"
    }
    if (-not (Test-Path $SourcePath)) {
        throw "Missing source config: $SourcePath"
    }

    $newCfg = Get-Content $SourcePath -Raw -Encoding UTF8 | ConvertFrom-Json

    $mergeKeys = @(
        "architecture",
        "execution_gates",
        "execution_composer",
        "execution_composer_v17",
        "trader_mind",
        "rl_inference",
        "trading_env",
        "cognition",
        "knowledge_net",
        "rl",
        "symbols",
        "counterfactual",
        "meta_learning",
        "meta_finetune",
        "attribution",
        "structure_analyzer",
        "adaptation_trigger",
        "trader_memory",
        "use_rl",
        "primary_symbol",
        "signal_expiry_minutes",
        "fallback_strategy",
        "state_file",
        "state_scaler_path"
    )

    if (Test-Path $TargetPath) {
        $cfg = Get-Content $TargetPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $kn2WasLive = $false
        if ($cfg.kn2) {
            $kn2WasLive = [bool]$cfg.kn2.enabled -and -not [bool]$cfg.kn2.shadow_mode
        }

        foreach ($prop in $mergeKeys) {
            if ($null -ne $newCfg.$prop) {
                $cfg | Add-Member -NotePropertyName $prop -NotePropertyValue $newCfg.$prop -Force
            }
        }

        if ($newCfg.kn2) {
            $cfg | Add-Member -NotePropertyName kn2 -NotePropertyValue $newCfg.kn2 -Force
            if ($kn2WasLive) {
                $cfg.kn2.enabled = $true
                $cfg.kn2.shadow_mode = $false
            }
        }

        if ($null -eq $cfg.enabled) {
            $cfg | Add-Member -NotePropertyName enabled -NotePropertyValue $true -Force
        }
    }
    else {
        $cfg = $newCfg
        $targetDir = Split-Path $TargetPath -Parent
        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
        }
    }

    if ($HorizonMinConfidence -gt 0) {
        if ($cfg.architecture -and $cfg.architecture.horizon_predictor) {
            $cfg.architecture.horizon_predictor.min_direction_confidence = $HorizonMinConfidence
        }
        if ($cfg.trader_mind) {
            $cfg.trader_mind.min_confidence = $HorizonMinConfidence
        }
        if ($cfg.cognition) {
            $cfg.cognition.direction_threshold = $HorizonMinConfidence
        }
    }

    $json = $cfg | ConvertTo-Json -Depth 25
    [System.IO.File]::WriteAllText($TargetPath, $json, [System.Text.UTF8Encoding]::new($false))

    $kn2Mode = if ($cfg.kn2.enabled -and -not $cfg.kn2.shadow_mode) { "LIVE" } elseif ($cfg.kn2.shadow_mode) { "shadow" } else { "off" }
    Write-Host "Merged config_agent.json → $TargetPath (execution_composer + trading_env preserved; KN2=$kn2Mode)" -ForegroundColor Green
}
