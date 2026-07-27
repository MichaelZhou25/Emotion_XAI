param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$experiments = @(
    @{
        Name = "SEED-IV session 1"
        Config = "configs/seediv_hyperbolic_node_edge_v8_full100.yaml"
        Summary = "results/seed_iv/hyperbolic_node_edge/v8_full100/summary.json"
    },
    @{
        Name = "SEED-IV session 2"
        Config = "configs/seediv_hyperbolic_node_edge_v8_session2_full100.yaml"
        Summary = "results/seed_iv/hyperbolic_node_edge/v8_session2_full100/summary.json"
    },
    @{
        Name = "SEED-IV session 3"
        Config = "configs/seediv_hyperbolic_node_edge_v8_session3_full100.yaml"
        Summary = "results/seed_iv/hyperbolic_node_edge/v8_session3_full100/summary.json"
    },
    @{
        Name = "SEED session 1"
        Config = "configs/seed_hyperbolic_node_edge_v8_full100.yaml"
        Summary = "results/seed/hyperbolic_node_edge/v8_session1_full100/summary.json"
    },
    @{
        Name = "SEED session 2"
        Config = "configs/seed_hyperbolic_node_edge_v8_session2_full100.yaml"
        Summary = "results/seed/hyperbolic_node_edge/v8_session2_full100/summary.json"
    },
    @{
        Name = "SEED session 3"
        Config = "configs/seed_hyperbolic_node_edge_v8_session3_full100.yaml"
        Summary = "results/seed/hyperbolic_node_edge/v8_session3_full100/summary.json"
    }
)

foreach ($experiment in $experiments) {
    if ((Test-Path $experiment.Summary) -and -not $Force) {
        Write-Host "[skip] $($experiment.Name): summary already exists"
        continue
    }
    Write-Host "[run] $($experiment.Name)"
    python train.py --config $experiment.Config
    if ($LASTEXITCODE -ne 0) {
        throw "$($experiment.Name) failed with exit code $LASTEXITCODE"
    }
}
