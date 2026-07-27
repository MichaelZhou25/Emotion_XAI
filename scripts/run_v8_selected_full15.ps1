param(
    [int]$Phase2Pid = 0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($Phase2Pid -gt 0 -and (Get-Process -Id $Phase2Pid -ErrorAction SilentlyContinue)) {
    Write-Host "[wait] phase-2 controller PID $Phase2Pid"
    Wait-Process -Id $Phase2Pid
}

function Get-MeanAccuracy {
    param([string[]]$Summaries)
    $values = foreach ($summary in $Summaries) {
        if (-not (Test-Path $summary)) {
            throw "Missing required phase-2 result: $summary"
        }
        $result = Get-Content $summary -Raw | ConvertFrom-Json
        [double]$result.acc.mean
    }
    return [double](($values | Measure-Object -Average).Average)
}

$seedCandidates = [ordered]@{
    endpoint025 = Get-MeanAccuracy @(
        "results/tuning/v8_phase1/seed/endpoint025/summary.json",
        "results/tuning/v8_phase2/seed/endpoint025_s2/summary.json",
        "results/tuning/v8_phase2/seed/endpoint025_s3/summary.json"
    )
    grl_ramp = Get-MeanAccuracy @(
        "results/tuning/v8_phase1/seed/grl_ramp/summary.json",
        "results/tuning/v8_phase2/seed/grl_ramp_s2/summary.json",
        "results/tuning/v8_phase2/seed/grl_ramp_s3/summary.json"
    )
    combo = Get-MeanAccuracy @(
        "results/tuning/v8_phase2/seed/combo_s1/summary.json",
        "results/tuning/v8_phase2/seed/combo_s2/summary.json",
        "results/tuning/v8_phase2/seed/combo_s3/summary.json"
    )
}

foreach ($entry in $seedCandidates.GetEnumerator()) {
    Write-Host ("[score] SEED {0}: {1:P2}" -f $entry.Key, $entry.Value)
}

$selected = $seedCandidates.GetEnumerator() |
    Sort-Object -Property Value -Descending |
    Select-Object -First 1

$seedConfigByName = @{
    endpoint025 = "configs/tuning/seed_v8_phase3_endpoint025_s1_full100.yaml"
    grl_ramp = "configs/tuning/seed_v8_phase3_grl_ramp_s1_full100.yaml"
    combo = "configs/tuning/seed_v8_phase3_combo_s1_full100.yaml"
}

$jobs = @(
    @{
        Name = "SEED-IV endpoint025 full15"
        Config = "configs/tuning/seediv_v8_phase3_endpoint025_s1_full100.yaml"
    },
    @{
        Name = "SEED $($selected.Key) full15"
        Config = $seedConfigByName[$selected.Key]
    }
)

foreach ($job in $jobs) {
    $saveDir = python -c "from utils.config import load_config; print(load_config('$($job.Config)')['logging']['save_dir'])"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not load $($job.Config)"
    }
    $summary = Join-Path $saveDir.Trim() "summary.json"
    if ((Test-Path $summary) -and -not $Force) {
        Write-Host "[skip] $($job.Name)"
        continue
    }
    Write-Host "[run] $($job.Name)"
    python train.py --config $job.Config
    if ($LASTEXITCODE -ne 0) {
        throw "$($job.Name) failed with exit code $LASTEXITCODE"
    }
}
