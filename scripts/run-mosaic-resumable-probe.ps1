# One-off continuation from weights; new optimizer, then resumable run checkpoints.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
Set-Location (Join-Path $root 'mosaic-gpu-81300')
$python = Join-Path $root '.venv\Scripts\python.exe'
$dataset = 'outputs/mosaic-data-gpu-feasibility-81300'
$parent = 'outputs/mosaic-warmstart-probe-81350/warmstart-plus-30000.pt'
$output = 'outputs/mosaic-resumable-probe-81360'
if (Test-Path $output) { throw 'Output exists; preserve it and resume explicitly.' }
& $python -m exposure.mosaic_resumable_probe --dataset $dataset --parent $parent --output $output --steps 30000 --interval 10000 --seed 81360 --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Probe failed; preserve checkpoints and inspect logs.' }
