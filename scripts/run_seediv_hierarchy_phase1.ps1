param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$configs = @(
    "configs/tuning/seediv_hier_child03_s1.yaml",
    "configs/tuning/seediv_hier_child03_s2.yaml",
    "configs/tuning/seediv_hier_child03_s3.yaml",
    "configs/tuning/seediv_hier_root02_child03_s1.yaml",
    "configs/tuning/seediv_hier_root02_child03_s2.yaml",
    "configs/tuning/seediv_hier_root02_child03_s3.yaml",
    "configs/tuning/seediv_hier_depth15_s1.yaml",
    "configs/tuning/seediv_hier_depth15_s2.yaml",
    "configs/tuning/seediv_hier_depth15_s3.yaml",
    "configs/tuning/seediv_hier_balance_s1.yaml",
    "configs/tuning/seediv_hier_balance_s2.yaml",
    "configs/tuning/seediv_hier_balance_s3.yaml",
    "configs/tuning/seediv_hier_depth15_balance_s1.yaml",
    "configs/tuning/seediv_hier_depth15_balance_s2.yaml",
    "configs/tuning/seediv_hier_depth15_balance_s3.yaml"
)

foreach ($config in $configs) {
    $saveDir = python -c "from utils.config import load_config; print(load_config('$config')['logging']['save_dir'])"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not load $config"
    }
    $summary = Join-Path $saveDir.Trim() "summary.json"
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
