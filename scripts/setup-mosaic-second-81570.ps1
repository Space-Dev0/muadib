# One-time second-laptop setup; may be retried after network loss.
$ErrorActionPreference = 'Stop'
$env:UV_HTTP_TIMEOUT = '180'
$root = Join-Path $env:USERPROFILE 'muadib'
$workspace = Join-Path $root 'mosaic-gpu-81300'
$python = Join-Path $root '.venv-official\Scripts\python.exe'
$uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
$installer = Join-Path $root 'install-mosaic-distributed-81570.ps1'
$log = Join-Path $root 'mosaic-second-setup-81570.log'
if (Get-ScheduledTask -TaskName 'MuadibMosaicDistributed81570-low' -ErrorAction SilentlyContinue) {
    Write-Output 'Low-cell task is already installed; setup is complete.'
    return
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class MuadibSetupPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
if ([MuadibSetupPower]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw 'Could not request prevention of idle sleep.' }
try {
    Start-Transcript -Path $log -Append | Out-Null
    if (-not (Test-Path $python) -or -not (Test-Path $uv)) { throw 'Missing approved Python environment or uv.' }
    & $uv pip install --python $python 'torch==2.9.1' --index-url 'https://download.pytorch.org/whl/cu128'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed; task will retry.' }
    $wheels = Join-Path $root 'wheels'
    if ((Test-Path (Join-Path $wheels 'numpy-2.5.4-cp313-cp313-win_amd64.whl')) -and
        (Test-Path (Join-Path $wheels 'pillow-13.0.0-cp313-cp313-win_amd64.whl')) -and
        (Test-Path (Join-Path $wheels 'scipy-1.18.1-cp313-cp313-win_amd64.whl'))) {
        & $uv pip install --python $python --no-index --find-links $wheels 'numpy==2.5.4' 'pillow==13.0.0' 'scipy==1.18.1'
    } else {
        & $uv pip install --python $python 'numpy==2.5.4' 'pillow==13.0.0' 'scipy==1.18.1'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed; task will retry.' }
    Set-Location $workspace
    & $python -c "import torch; assert torch.__version__ == '2.9.1+cu128' and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
    & $python -c "from pathlib import Path; from exposure.mosaic_distributed_supervisor import from_report; print(from_report(Path('outputs/mosaic-matched-81570-low-balanced-continued'), 'low-balanced')); print(from_report(Path('outputs/mosaic-matched-81570-low-imbalanced-continued'), 'low-imbalanced'))"
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint preflight failed.' }
    $complete = Get-ChildItem 'outputs' -Directory -Filter 'mosaic-distributed-migration-smoke-81570-*' | Where-Object { Test-Path (Join-Path $_.FullName 'report.json') } | Select-Object -First 1
    if ($complete) { $smoke = $complete.FullName } else {
        $smoke = 'outputs/mosaic-distributed-migration-smoke-81570-' + [guid]::NewGuid().ToString('N').Substring(0,10)
        & $python -u -m exposure.mosaic_matched_cell --complexity low --imbalance balanced --trainer warmstart resume --source outputs/mosaic-matched-81570-low-balanced-continued/warmstart-150000.pt --dataset outputs/mosaic-data-gpu-feasibility-81300 --output $smoke --steps 1 --snapshot-steps 150001 --device cuda
        if ($LASTEXITCODE -ne 0) { throw 'Migration smoke test failed; preserve partial output and investigate.' }
    }
    & $python -c "from pathlib import Path; from exposure.mosaic_distributed_supervisor import from_report; print(from_report(Path(r'$smoke'), 'low-balanced'))"
    if ($LASTEXITCODE -ne 0) { throw 'Migration smoke report validation failed.' }
    if (-not (Get-ScheduledTask -TaskName 'MuadibMosaicDistributed81570-low' -ErrorAction SilentlyContinue)) {
        & $installer -Role low
        if (-not $?) { throw 'Could not install low-cell supervisor.' }
    }
    Write-Output 'Setup verified; low-cell scheduled task is installed.'
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    [void][MuadibSetupPower]::SetThreadExecutionState([uint32]2147483648)
}
