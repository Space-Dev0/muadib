# Generate 40 fresh checker-validation pilot images from the existing four-rule checkpoint.
# Rerun after interruption; never update the checkpoint-compatible source tree.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
Set-Location (Join-Path $root 'source')
$python = Join-Path $root '.venv\Scripts\python.exe'
$checkpoint = 'outputs\velocity-online-v1-800000-79c331f.pt'
$dir = 'outputs\audit-validation-800000-seeds-30000-30009'
if (-not (Test-Path $checkpoint) -or -not (Test-Path $python)) { throw 'Missing checkpoint or Python environment.' }
$mutex = New-Object System.Threading.Mutex($false, 'Local\MuadibPilotAuditGeneration')
$owns = $false
try {
    try { $owns = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $owns = $true }
    if (-not $owns) { throw 'Another pilot audit generation process is running.' }
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($rule in @('red', 'square', 'red_left_of_square', 'blue_above_circle')) {
        foreach ($seed in 30000..30009) {
            $path = Join-Path $dir "$rule-$seed.png"
            if (Test-Path $path) {
                # A crashed writer may leave an incomplete PNG: refuse to skip it.
                & $python -c "from PIL import Image; import sys; im=Image.open(sys.argv[1]); im.verify()" $path
                if ($LASTEXITCODE -ne 0) { throw "Existing image is invalid: $path" }
                continue
            }
            & $python -m exposure.velocity_online sample --checkpoint $checkpoint --rule $rule --seed $seed --device cuda --output $path
            if ($LASTEXITCODE -ne 0) { throw "Generation failed: $rule $seed" }
            Write-Output "Saved $path"
        }
    }
    Write-Output 'Completed 40 fresh validation images.'
} finally {
    if ($owns) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
