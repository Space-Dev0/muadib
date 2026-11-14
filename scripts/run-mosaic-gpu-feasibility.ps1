# One bounded MOSAIC-inspired feasibility run, not the research gate.
# Requires the isolated mosaic-gpu-81300 directory and the existing GPU venv.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$dir = Join-Path $root 'mosaic-gpu-81300'
$python = Join-Path $root '.venv\Scripts\python.exe'
Set-Location $dir
$dataset = 'outputs/mosaic-data-gpu-feasibility-81300'
$checkpointDir = 'outputs/mosaic-gpu-feasibility-checkpoints-81302'
$report = 'outputs/mosaic-gpu-feasibility-81302.json'
$sampleDir = 'outputs/mosaic-gpu-feasibility-samples-81302'
if ((Test-Path $checkpointDir) -or (Test-Path $report) -or (Test-Path $sampleDir)) {
    throw 'Output already exists: preserve it and choose a new run identifier.'
}
& $python -m exposure.mosaic_toy_train --dataset $dataset --seed 81302 --steps 5000 --batch-size 4 --width 16 --device cuda --output $report --checkpoint-dir $checkpointDir
if ($LASTEXITCODE -ne 0) { throw 'Feasibility training failed; inspect partial checkpoints.' }
New-Item -ItemType Directory -Path $sampleDir | Out-Null
foreach ($cell in @('low-balanced', 'low-imbalanced', 'high-balanced', 'high-imbalanced')) {
    foreach ($label in @('red', 'blue')) {
        foreach ($seed in @(81310, 81311)) {
            & $python -m exposure.mosaic_toy_sample --checkpoint "$checkpointDir/$cell.pt" --label $label --seed $seed --steps 50 --device cuda --output "$sampleDir/$cell-$label-$seed.png"
            if ($LASTEXITCODE -ne 0) { throw "Sampling failed: $cell $label $seed" }
        }
    }
}
Write-Output 'Completed bounded training and 16 exploratory samples; this is not a MOSAIC replication.'
