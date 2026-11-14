$ErrorActionPreference = 'Stop'
$root=Join-Path $env:USERPROFILE 'muadib'
Set-Location (Join-Path $root 'mosaic-gpu-81300')
$python=Join-Path $root '.venv\Scripts\python.exe'
& $python -u -m exposure.mosaic_guidance_probe --checkpoint outputs/mosaic-pilotunet-resume-81379/warmstart-030000.pt --output outputs/mosaic-guidance-30k-81385 --seeds 81380 81381 --guidance 1 2 --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Guidance probe failed; preserve partial outputs.' }
