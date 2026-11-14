$ErrorActionPreference = 'Stop'
$root=Join-Path $env:USERPROFILE 'muadib'
Set-Location (Join-Path $root 'mosaic-gpu-81300')
$python=Join-Path $root '.venv\Scripts\python.exe'
& $python -u -m exposure.mosaic_pilotunet_warmstart warmstart --source outputs/mosaic-pilotunet-probe-81373/pilotunet-002000.pt --dataset outputs/mosaic-data-gpu-feasibility-81300 --output outputs/mosaic-pilotunet-warmstart-81377 --seed 81377 --steps 6000 --device cuda --preview-seeds 81380 --snapshot-steps 5000 8000
if ($LASTEXITCODE -ne 0) { throw 'Warm-start failed; preserve partial outputs.' }
