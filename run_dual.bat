@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Starting detector...
REM Close any existing detector/ui windows from previous runs
taskkill /F /T /FI "WINDOWTITLE eq Detector" >nul 2>&1
taskkill /F /T /FI "WINDOWTITLE eq UI" >nul 2>&1

start "Detector" /D "%SCRIPT_DIR%" cmd /k ".\venv\Scripts\python detector_server.py"

REM Give the detector a moment to bind TCP
timeout /t 2 /nobreak >nul

echo Starting UI...
start "UI" /D "%SCRIPT_DIR%" cmd /k ".\venv\Scripts\python ui_client.py"

REM Watchdog: restart if any window is closed
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$log = Join-Path (Get-Location) 'watchdog.log';" ^
  "Add-Content -Path $log -Value ('[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] Watchdog started');" ^
  "while ($true) {" ^
  "  $det = Get-Process | Where-Object { $_.MainWindowTitle -eq 'Detector' };" ^
  "  $ui  = Get-Process | Where-Object { $_.MainWindowTitle -eq 'UI' };" ^
  "  if (-not $det -or -not $ui) {" ^
  "    Add-Content -Path $log -Value ('[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] Restarting');" ^
  "    if ($det) { $det | Stop-Process -Force }" ^
  "    if ($ui)  { $ui  | Stop-Process -Force }" ^
  "    Start-Sleep -Seconds 1;" ^
  "    Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', '.\\venv\\Scripts\\python detector_server.py' -WindowStyle Normal -WorkingDirectory (Get-Location) -PassThru | ForEach-Object { $_.MainWindowTitle = 'Detector' };" ^
  "    Start-Sleep -Seconds 2;" ^
  "    Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', '.\\venv\\Scripts\\python ui_client.py' -WindowStyle Normal -WorkingDirectory (Get-Location) -PassThru | ForEach-Object { $_.MainWindowTitle = 'UI' };" ^
  "  }" ^
  "  Start-Sleep -Seconds 5;" ^
  "}"

endlocal
