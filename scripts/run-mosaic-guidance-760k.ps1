# Bounded inference only; does not resume training.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
Set-Location (Join-Path $root 'mosaic-gpu-81300')
$python = Join-Path $root '.venv\Scripts\python.exe'
$checkpoint = 'outputs/mosaic-pilotunet-continuous-81391/segment-760000-609378c832/warmstart-760000.pt'
$output = 'outputs/mosaic-guidance-760k-matched-81410'
if (Test-Path $output) { throw 'Output exists; preserve it and choose a new directory.' }
& $python -u -m exposure.mosaic_guidance_probe --checkpoint $checkpoint --output $output --seeds 81380 81381 --guidance 1 2 --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Sampling failed; preserve partial outputs.' }
