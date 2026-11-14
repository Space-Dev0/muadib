# Bounded overnight continuation of the 30k low-balanced feasibility model.
# New checkpoints include optimizer and RNG state; this is not a Stage 0 run.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$workspace = Join-Path $root 'mosaic-gpu-81300'
$python = Join-Path $root '.venv\Scripts\python.exe'
$source = Join-Path $workspace 'outputs\mosaic-pilotunet-resume-81379\warmstart-030000.pt'
$output = Join-Path $workspace 'outputs\mosaic-pilotunet-overnight-81390'
$log = Join-Path $root 'mosaic-pilotunet-overnight-81390.log'
if (-not (Test-Path $source) -or -not (Test-Path $python)) { throw 'Missing parent checkpoint or Python environment.' }
if ((Get-FileHash -Algorithm SHA256 $source).Hash.ToLowerInvariant() -ne 'b3405f731ff44963fb1704687108e90499812bb21657410cbd78dab051ecca8c') { throw 'Parent checkpoint hash mismatch.' }
if (Test-Path $output) { throw 'Output exists; preserve it and resume from a verified complete checkpoint in a new directory.' }
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class MuadibOvernightPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$continuous = [uint32]2147483648
$systemRequired = [uint32]1
$state = [MuadibOvernightPower]::SetThreadExecutionState($continuous -bor $systemRequired)
if ($state -eq 0) { throw 'Could not request Windows to stay awake.' }
try {
    Set-Location $workspace
    Start-Transcript -Path $log -Append | Out-Null
    & $python -u -m exposure.mosaic_pilotunet_warmstart resume --source $source --dataset outputs/mosaic-data-gpu-feasibility-81300 --output $output --steps 120000 --device cuda --snapshot-steps 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE; preserve outputs and inspect the log." }
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    [void][MuadibOvernightPower]::SetThreadExecutionState($continuous)
}
