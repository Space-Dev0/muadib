# Bounded weight-only warm start; not exact resume or a MOSAIC replication.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
Set-Location (Join-Path $root 'mosaic-gpu-81300')
$python = Join-Path $root '.venv\Scripts\python.exe'
$dataset = 'outputs/mosaic-data-gpu-feasibility-81300'
$checkpoint = 'outputs/mosaic-multiscale-probe-81322/multiscale-20000.pt'
$output = 'outputs/mosaic-warmstart-probe-81350'
if (Test-Path $output) { throw 'Output exists; preserve it and choose a new run identifier.' }
& $python -m exposure.mosaic_warmstart_probe --dataset $dataset --checkpoint $checkpoint --output $output --additional-updates 30000 --save-every 10000 --seed 81350 --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Warm-start probe failed; preserve partial outputs.' }
