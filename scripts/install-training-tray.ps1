# Run once as the signed-in Windows user after copying both tray and trainer scripts
# to %USERPROFILE%\muadib. No password is stored; task runs in the interactive session.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'muadib'
$tray = Join-Path $root 'training-tray.ps1'
$trainer = Join-Path $root 'resume-gpu-training.ps1'
if (-not (Test-Path $tray) -or -not (Test-Path $trainer)) { throw 'Copy training-tray.ps1 and resume-gpu-training.ps1 into %USERPROFILE%\muadib first.' }
$name = 'Muadib GPU Training Tray'
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { throw "Task '$name' already exists; inspect it before changing it." }
$exe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$action = New-ScheduledTaskAction -Execute $exe -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$tray`""
$account = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $account
$principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $name
Write-Host "Installed and started '$name'. Look for Muadib in the notification area."
