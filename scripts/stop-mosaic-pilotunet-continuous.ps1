# Persistently stop the continuous supervisor and its predecessor.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$control = Join-Path $root 'mosaic-pilotunet-continuous-81391'
New-Item -ItemType Directory -Force -Path $control | Out-Null
Set-Content -LiteralPath (Join-Path $control 'STOP') -Value 'Stopped by user'
Disable-ScheduledTask -TaskName 'MuadibPilotUNetContinuous81391' -ErrorAction SilentlyContinue | Out-Null
$prior = Get-ScheduledTask -TaskName 'MuadibPilotUNetOvernight81390' -ErrorAction SilentlyContinue
if ($prior -and $prior.State -eq 'Running') { Stop-ScheduledTask -TaskName 'MuadibPilotUNetOvernight81390' }
# Windows may leave the venv launcher and its child running after stopping a task.
# Match only the known predecessor launcher and output path; never kill by name alone.
$python = Join-Path $root '.venv\Scripts\python.exe'
$trainer = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {
    $_.ExecutablePath -ieq $python -and
    $_.CommandLine -like '*exposure.mosaic_pilotunet_warmstart*' -and
    $_.CommandLine -like '*mosaic-pilotunet-overnight-81390*'
})
if ($trainer.Count -gt 1) { throw 'Multiple matching predecessor launchers; inspect processes manually.' }
if ($trainer.Count -eq 1) {
    & taskkill.exe /PID $trainer[0].ProcessId /T /F
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the predecessor process tree; inspect processes manually.' }
}
Write-Output 'Stop requested. The supervisor should terminate its active segment within a few seconds; partial files are preserved.'
