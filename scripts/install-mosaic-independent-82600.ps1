# Independent model-seed block on the existing fixed four-cell dataset.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$workspace = Join-Path $root 'mosaic-gpu-81300'
$python = Join-Path $root '.venv\Scripts\python.exe'
$name = 'MuadibMosaicIndependent82600'
$runner = Join-Path $root 'run-mosaic-independent-82600.ps1'
$log = Join-Path $root 'mosaic-independent-82600.log'
if (-not (Test-Path $python) -or -not (Test-Path (Join-Path $workspace 'exposure\mosaic_distributed_supervisor.py'))) { throw 'Missing Python or supervisor.' }
if ((Test-Path $runner) -or (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)) { throw 'Runner or task already exists; preserve it.' }
$body = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$workspace'
& '$python' -u -m exposure.mosaic_distributed_supervisor --workspace '$workspace' --run-id 82600 --cells low-balanced low-imbalanced high-balanced high-imbalanced --target 500000 *>> '$log'
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
Get-ScheduledTask -TaskName $name | Select-Object TaskName,State
