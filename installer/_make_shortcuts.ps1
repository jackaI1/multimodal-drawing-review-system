param([Parameter(Mandatory = $true)][string]$AppDir)
# Create the "RedlineCheck" (start) and "Stop RedlineCheck" Desktop shortcuts.
$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$installer = Join-Path $AppDir 'installer'

$run = $ws.CreateShortcut((Join-Path $desktop 'RedlineCheck.lnk'))
$run.TargetPath = Join-Path $installer 'run_redlinecheck.bat'
$run.WorkingDirectory = $AppDir
$run.IconLocation = "$env:SystemRoot\System32\shell32.dll,167"   # blue page icon
$run.Description = 'Start RedlineCheck (opens in your web browser)'
$run.Save()

$stop = $ws.CreateShortcut((Join-Path $desktop 'Stop RedlineCheck.lnk'))
$stop.TargetPath = Join-Path $installer 'stop_redlinecheck.bat'
$stop.WorkingDirectory = $AppDir
$stop.IconLocation = "$env:SystemRoot\System32\shell32.dll,131"   # red stop icon
$stop.Description = 'Stop RedlineCheck'
$stop.Save()

Write-Host '  [ok] Desktop shortcuts created.'
