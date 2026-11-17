$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$workspace = Join-Path $root 'mosaic-gpu-81300'
$python = Join-Path $root '.venv\Scripts\python.exe'
$output = Join-Path $workspace 'outputs\mosaic-matched-eval-500k-82000'
$log = Join-Path $root 'mosaic-matched-eval-82000.log'
if (Test-Path $output) { throw 'Evaluation output already exists; preserve it.' }
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class MuadibEvalPower {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
if ([MuadibEvalPower]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw 'Could not prevent idle sleep.' }
try {
    Set-Location $workspace
    Start-Transcript -Path $log -Append | Out-Null
    & $python -u -m exposure.mosaic_matched_eval --workspace $workspace --output $output --device cuda
    if ($LASTEXITCODE -ne 0) { throw 'Inference failed; preserve partial output.' }
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    [void][MuadibEvalPower]::SetThreadExecutionState([uint32]2147483648)
}
