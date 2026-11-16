param(
    [Parameter(Mandatory=$true)][ValidateSet('low','high')][string]$Role,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$workspace = Join-Path $root 'mosaic-gpu-81300'
$python = if ($PythonPath) { $PythonPath } elseif ($Role -eq 'low') { Join-Path $root '.venv-official\Scripts\python.exe' } else { Join-Path $root '.venv\Scripts\python.exe' }
$name = "MuadibMosaicDistributed81570-$Role"
$runner = Join-Path $root "run-mosaic-distributed-81570-$Role.ps1"
if (-not (Test-Path $python) -or -not (Test-Path (Join-Path $workspace 'exposure\mosaic_distributed_supervisor.py'))) { throw 'Missing project or Python.' }
if ((Test-Path $runner) -or (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)) { throw 'Runner or task already exists; inspect first.' }
$cells = if ($Role -eq 'low') { 'low-balanced low-imbalanced' } else { 'high-balanced high-imbalanced' }
$wait = if ($Role -eq 'high') { '--wait-task MuadibMosaicMatched81570' } else { '' }
$log = Join-Path $root "mosaic-distributed-81570-$Role.log"
$body = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$workspace'
& '$python' -u -m exposure.mosaic_distributed_supervisor --workspace '$workspace' --cells $cells --target 500000 $wait *>> '$log'
if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
"@
Set-Content -Path $runner -Value $body -Encoding UTF8
$exe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$action = New-ScheduledTaskAction -Execute $exe -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`""
$account = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Interactive -RunLevel Limited
$login = New-ScheduledTaskTrigger -AtLogOn -User $account
$timer = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 365)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
Register-ScheduledTask -TaskName $name -Action $action -Trigger @($login,$timer) -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $name
Get-ScheduledTask -TaskName $name | Select-Object TaskName, State
