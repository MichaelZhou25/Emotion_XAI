$ErrorActionPreference = 'Stop'

foreach ($session in @(1, 2, 3)) {
    Write-Output "[SEED-V strict LOSO] starting session $session"
    & python train.py `
        --config configs/seedv_strict_dg_loso_v8_full100.yaml `
        --sessions $session
    if ($LASTEXITCODE -ne 0) {
        throw "SEED-V session $session failed with exit code $LASTEXITCODE"
    }
    Write-Output "[SEED-V strict LOSO] completed session $session"
}
