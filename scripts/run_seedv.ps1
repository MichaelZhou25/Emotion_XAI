$ErrorActionPreference = 'Stop'

& python train.py --config configs/seedv.yaml
if ($LASTEXITCODE -ne 0) {
    throw "SEED-V paper-aligned three-fold run failed with exit code $LASTEXITCODE"
}
