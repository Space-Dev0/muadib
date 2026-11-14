# Remaining three cells of a bounded exploratory 2x2, not a replication gate.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
Set-Location (Join-Path $root 'mosaic-gpu-81300')
$python = Join-Path $root '.venv\Scripts\python.exe'
foreach ($cell in @(@('low','imbalanced'), @('high','balanced'), @('high','imbalanced'))) {
    $complexity = $cell[0]
    $imbalance = $cell[1]
    $output = "outputs/mosaic-multiscale-81322-$complexity-$imbalance"
    & $python -m exposure.mosaic_multiscale_probe --dataset outputs/mosaic-data-gpu-feasibility-81300 --output $output --device cuda --complexity $complexity --imbalance $imbalance
    if ($LASTEXITCODE -ne 0) { throw "Run failed for $complexity/$imbalance; preserve partial outputs." }
}
Write-Output 'Completed remaining three exploratory cells.'
