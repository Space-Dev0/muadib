# Windows tray controller for resume-gpu-training.ps1. Install at sign-in with
# install-training-tray.ps1. Pausing discards updates since the last checkpoint.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$root = Join-Path $env:USERPROFILE 'muadib'
$trainer = Join-Path $root 'resume-gpu-training.ps1'
$state = Join-Path $root 'training-paused'
$stdout = Join-Path $root 'training-tray.log'
$stderr = Join-Path $root 'training-tray-errors.log'
if (-not (Test-Path $trainer)) { throw "Missing training script: $trainer" }
$mutex = New-Object System.Threading.Mutex($false, 'Local\MuadibTrainingTray')
$ownsMutex = $false
try {
    try { $ownsMutex = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) { exit 0 }
    $script:child = $null
    $script:failed = $false
    $icon = New-Object System.Windows.Forms.NotifyIcon
    $icon.Icon = [System.Drawing.SystemIcons]::Application
    $icon.Text = 'Muadib training'
    $menu = New-Object System.Windows.Forms.ContextMenuStrip
    $status = $menu.Items.Add('Starting...')
    $status.Enabled = $false
    $toggle = $menu.Items.Add('Pause training')
    $exitItem = $menu.Items.Add('Exit tray')
    $icon.ContextMenuStrip = $menu
    $icon.Visible = $true

    function Stop-Training {
        if ($null -ne $script:child -and -not $script:child.HasExited) {
            & "$env:SystemRoot\System32\taskkill.exe" /PID $script:child.Id /T /F | Out-Null
            $script:child.WaitForExit()
        }
        $script:child = $null
    }
    function Update-Training {
        if (Test-Path $state) {
            $status.Text = 'Paused'
            $toggle.Text = 'Resume training'
            $icon.Text = 'Muadib training: paused'
            return
        }
        $toggle.Text = 'Pause training'
        if ($null -ne $script:child -and $script:child.HasExited) {
            $script:failed = $true
            $script:child = $null
        }
        if ($script:failed) {
            $status.Text = 'Stopped: check training-tray-errors.log'
            $toggle.Text = 'Retry training'
            $icon.Text = 'Muadib training: stopped'
            return
        }
        if ($null -eq $script:child) {
            # Separate files avoid the identical redirection-path restriction.
            $script:child = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',("`"$trainer`"")) -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        }
        $status.Text = 'Training (checkpoint every 10,000 updates)'
        $icon.Text = 'Muadib training: running'
    }
    $toggle.Add_Click({
        if (Test-Path $state) {
            Remove-Item $state
            $script:failed = $false
        } elseif ($script:failed) {
            $script:failed = $false
        } else {
            New-Item -ItemType File -Path $state -Force | Out-Null
            Stop-Training
        }
        Update-Training
    })
    $exitItem.Add_Click({
        New-Item -ItemType File -Path $state -Force | Out-Null
        Stop-Training
        [System.Windows.Forms.Application]::ExitThread()
    })
    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 30000
    $timer.Add_Tick({ Update-Training })
    Update-Training
    $timer.Start()
    [System.Windows.Forms.Application]::Run()
} finally {
    if ($timer) { $timer.Stop(); $timer.Dispose() }
    if ($icon) { $icon.Visible = $false; $icon.Dispose() }
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
