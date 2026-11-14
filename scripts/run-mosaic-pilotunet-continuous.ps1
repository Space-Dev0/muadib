# Unlimited supervised continuation after the current bounded task finishes.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$workspace = Join-Path $root 'mosaic-gpu-81300'
$control = Join-Path $root 'mosaic-pilotunet-continuous-81391'
$python = Join-Path $root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $control | Out-Null
Set-Location $workspace
Start-Transcript -Path (Join-Path $control 'supervisor.log') -Append | Out-Null
try {
    & $python -u -m exposure.mosaic_continuous_supervisor --workspace $workspace --control $control
    if ($LASTEXITCODE -ne 0) { throw "Supervisor exited with code $LASTEXITCODE; inspect the log." }
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
