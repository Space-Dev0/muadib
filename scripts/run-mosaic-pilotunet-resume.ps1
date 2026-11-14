$ErrorActionPreference = 'Stop'
$root=Join-Path $env:USERPROFILE 'muadib'
Set-Location (Join-Path $root 'mosaic-gpu-81300')
$python=Join-Path $root '.venv\Scripts\python.exe'
& $python -u -m exposure.mosaic_pilotunet_warmstart resume --source outputs/mosaic-pilotunet-warmstart-81377/warmstart-008000.pt --dataset outputs/mosaic-data-gpu-feasibility-81300 --output outputs/mosaic-pilotunet-resume-81378 --steps 12000 --device cuda --snapshot-steps 14000 20000
if ($LASTEXITCODE -ne 0) { throw 'PilotUNet resume failed; preserve partial outputs.' }
