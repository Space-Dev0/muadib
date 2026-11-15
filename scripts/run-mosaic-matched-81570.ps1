# Bounded four-cell exploratory run; separate from the stopped 760k supervisor.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$workspace = Join-Path $root 'mosaic-gpu-81300'
$python = Join-Path $root '.venv\Scripts\python.exe'
$dataset = 'outputs/mosaic-data-gpu-feasibility-81300'
$log = Join-Path $root 'mosaic-matched-81570.log'
Set-Location $workspace
if (-not (Test-Path $python) -or -not (Test-Path "$dataset/manifest.json")) { throw 'Missing Python or dataset.' }
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class MuadibMatchedPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$state = [MuadibMatchedPower]::SetThreadExecutionState([uint32]2147483649)
if ($state -eq 0) { throw 'Could not request the system to stay awake.' }
try {
    Start-Transcript -Path $log -Append | Out-Null
    foreach ($cell in @(@('low','balanced'), @('low','imbalanced'), @('high','balanced'), @('high','imbalanced'))) {
        $complexity = $cell[0]; $imbalance = $cell[1]
        $base = "outputs/mosaic-matched-81570-$complexity-$imbalance"
        $initial = "$base-initial"; $continued = "$base-continued"
        if ((Test-Path $initial) -or (Test-Path $continued)) { throw "Output already exists for $complexity/$imbalance; preserve it." }
        & $python -u -m exposure.mosaic_matched_cell --complexity $complexity --imbalance $imbalance --trainer probe bounded --dataset $dataset --output $initial --seed 81570 --steps 1000 --batch-size 4 --width 16 --timesteps 1000 --lr 0.0001 --dropout 0.1 --ema-decay 0.999 --preview-seeds 81580 --device cuda
        if ($LASTEXITCODE -ne 0) { throw "Initial run failed for $complexity/$imbalance." }
        $checkpoint = "$initial/pilotunet-001000.pt"
        & $python -u -m exposure.mosaic_matched_cell --complexity $complexity --imbalance $imbalance --trainer warmstart warmstart --source $checkpoint --dataset $dataset --output $continued --seed 81571 --steps 149000 --snapshot-steps 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 --device cuda
        if ($LASTEXITCODE -ne 0) { throw "Continuation failed for $complexity/$imbalance; preserve checkpoints." }
    }
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    [void][MuadibMatchedPower]::SetThreadExecutionState([uint32]2147483648)
}
