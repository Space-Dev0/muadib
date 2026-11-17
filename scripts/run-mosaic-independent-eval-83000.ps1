# Wait for the second training block; sample and package without human input.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$workspace = Join-Path $root 'mosaic-gpu-81300'
$python = Join-Path $root '.venv\Scripts\python.exe'
$log = Join-Path $root 'mosaic-independent-eval-83000.log'
Set-Location $workspace
while ((Get-ScheduledTask MuadibMosaicIndependent82600).State -in @('Running','Queued')) { Start-Sleep -Seconds 60 }
& $python -u -m exposure.mosaic_independent_eval --workspace $workspace --output (Join-Path $workspace 'outputs\mosaic-independent-eval-500k-83000') --review (Join-Path $workspace 'outputs\mosaic-independent-review-500k-83000') --device cuda *>> $log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Disable-ScheduledTask MuadibMosaicIndependentEval83000 | Out-Null
