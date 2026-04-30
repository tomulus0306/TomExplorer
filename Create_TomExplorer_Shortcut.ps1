$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$shortcutPath = Join-Path $projectRoot 'TomExplorer.lnk'
$targetPath = Join-Path $projectRoot 'Start_TomExplorer.bat'
$iconPath = Join-Path $projectRoot 'TomExplorer.ico'

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.WorkingDirectory = $projectRoot
if (Test-Path $iconPath) {
    $shortcut.IconLocation = $iconPath
}
$shortcut.Save()

Write-Host "Shortcut created:" $shortcutPath