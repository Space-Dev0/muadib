# One-cell, fixed-budget multiscale learnability probe; not a Stage 0 replication.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
Set-Location (Join-Path $root 'mosaic-gpu-81300')
$python = Join-Path $root '.venv\Scripts\python.exe'
& $python -m exposure.mosaic_multiscale_probe --dataset outputs/mosaic-data-gpu-feasibility-81300 --output outputs/mosaic-multiscale-probe-81322 --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Multiscale learnability probe failed; preserve partial outputs.' }
