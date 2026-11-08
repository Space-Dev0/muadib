# Run on the Windows GPU laptop: powershell -ExecutionPolicy Bypass -File .\resume-gpu-training.ps1
# Stop with Ctrl+C. Restart the script after sleep or reboot if it stopped.
$ErrorActionPreference = 'Stop'
$source = Join-Path $env:USERPROFILE 'muadib\source'
$python = Join-Path $env:USERPROFILE 'muadib\.venv\Scripts\python.exe'
if (-not (Test-Path $source) -or -not (Test-Path $python)) {
    throw 'Expected muadib\source and muadib\.venv under your Windows user directory.'
}
Set-Location $source

$mutex = New-Object System.Threading.Mutex($false, 'Local\MuadibVelocityTraining')
$ownsMutex = $false
try {
    try { $ownsMutex = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) { throw 'Another copy of this training script is already running.' }

    # Also guard against a training process started outside this script.
    $running = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'exposure\.velocity_online\s+(resume|train)'
    }
    if ($running) { throw 'An online velocity training process is already running. Stop this script and check that process.' }

    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    while ($true) {
        $checkpoints = @(Get-ChildItem 'outputs' -Filter 'velocity-online-v1-*-79c331f.pt' | Where-Object {
            $_.BaseName -match '^velocity-online-v1-\d+-79c331f$'
        } | Sort-Object { [int64]($_.BaseName -replace '^velocity-online-v1-(\d+)-79c331f$', '$1') } -Descending)
        if ($checkpoints.Count -eq 0) { throw 'No 79c331f online velocity checkpoint found in outputs.' }
        $latest = $checkpoints[0]
        $step = [int64]($latest.BaseName -replace '^velocity-online-v1-(\d+)-79c331f$', '$1')
        # Refuse a truncated, mislabeled, or non-CUDA checkpoint instead of silently falling back.
        & $python -c "import sys, torch; from exposure.velocity_online import _resume_parts; c=torch.load(sys.argv[1], map_location='cpu', weights_only=True); cfg, _, _ = _resume_parts(c); assert cfg['steps']==int(sys.argv[2]) and cfg['device']=='cuda', 'Checkpoint steps or device mismatch'" $latest.FullName $step
        if ($LASTEXITCODE -ne 0) { throw "Checkpoint validation failed: $($latest.FullName). Preserve it and investigate before restarting." }
        $next = $step + 10000
        $target = Join-Path (Join-Path $source 'outputs') "velocity-online-v1-$next-79c331f.pt"
        if (Test-Path $target) { throw "Target already exists: $target" }
        Write-Host "Resuming $step to $next updates on CUDA. Stop with Ctrl+C."
        & $python -m exposure.velocity_online resume --checkpoint $latest.FullName --steps 10000 --output $target
        if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE. Existing checkpoints have been retained." }
        if (-not (Test-Path $target)) { throw "Training reported success but did not create $target" }
        Write-Host "Saved $target"
    }
} finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
