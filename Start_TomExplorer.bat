@echo off
setlocal
cd /d "%~dp0"
set "TOMEXPLORER_HOST=127.0.0.1"
set "PYTHONWARNINGS=ignore::SyntaxWarning"
title TomExplorer Launcher
set "TOMEXPLORER_PYTHON=%LocalAppData%\Programs\Python\Python314\python.exe"
if not exist "%TOMEXPLORER_PYTHON%" (
	echo Python 3.14 wurde unter "%LocalAppData%\Programs\Python\Python314\python.exe" nicht gefunden.
	echo Bitte Python installieren oder den Startpfad in Start_TomExplorer.bat anpassen.
	pause
	exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$processes = Get-CimInstance Win32_Process; foreach ($process in $processes) { if ($process.CommandLine -match 'tomexplorer_app.py') { try { Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop } catch {} } }"

set "TOMEXPLORER_PORT="
for /f %%P in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$chosen = $null; foreach ($port in 8050..8099) { try { $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse('127.0.0.1'), $port); $listener.Start(); $listener.Stop(); $chosen = $port; break } catch { if ($listener) { try { $listener.Stop() } catch {} } } }; if ($null -eq $chosen) { exit 1 }; Write-Output $chosen"') do set "TOMEXPLORER_PORT=%%P"

if not defined TOMEXPLORER_PORT (
	echo Kein freier lokaler Port zwischen 8050 und 8099 gefunden.
	pause
	exit /b 1
)

title TomExplorer Server
echo TomExplorer startet auf http://%TOMEXPLORER_HOST%:%TOMEXPLORER_PORT%/
"%TOMEXPLORER_PYTHON%" tomexplorer_app.py
