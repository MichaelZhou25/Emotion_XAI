param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$configs = @(
    "configs/tuning/seediv_v8_phase1_base.yaml",
    "configs/tuning/seediv_v8_phase1_lr3.yaml",
    "configs/tuning/seediv_v8_phase1_regularized.yaml",
    "configs/tuning/seediv_v8_phase1_batch256.yaml",
    "configs/tuning/seediv_v8_phase1_grl_ramp.yaml",
    "configs/tuning/seediv_v8_phase1_hpcl.yaml",
    "configs/tuning/seediv_v8_phase1_endpoint025.yaml",
    "configs/tuning/seediv_v8_phase1_endpoint075.yaml",
    "configs/tuning/seed_v8_phase1_base.yaml",
    "configs/tuning/seed_v8_phase1_lr3.yaml",
    "configs/tuning/seed_v8_phase1_regularized.yaml",
    "configs/tuning/seed_v8_phase1_batch256.yaml",
    "configs/tuning/seed_v8_phase1_grl_ramp.yaml",
    "configs/tuning/seed_v8_phase1_hpcl.yaml",
    "configs/tuning/seed_v8_phase1_endpoint025.yaml",
    "configs/tuning/seed_v8_phase1_endpoint075.yaml"
)

foreach ($config in $configs) {
    $resolved = python -c "from utils.config import load_config; print(load_config('$config')['logging']['save_dir'])"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not load $config"
    }
    $summary = Join-Path $resolved.Trim() "summary.json"
    if ((Test-Path $summary) -and -not $Force) {
        Write-Host "[skip] $config"
        continue
    }
    Write-Host "[run] $config"
    python train.py --config $config
    if ($LASTEXITCODE -ne 0) {
        throw "$config failed with exit code $LASTEXITCODE"
    }
}
