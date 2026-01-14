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

REM Schedule daily restart at 09:00
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$dir = (Get-Location).Path;" ^
  "$now = Get-Date;" ^
  "$target = (Get-Date -Hour 9 -Minute 0 -Second 0);" ^
  "if ($now -ge $target) { $target = $target.AddDays(1) }" ^
  "$sleep = ($target - $now).TotalMilliseconds;" ^
  "Start-Sleep -Milliseconds $sleep;" ^
  "taskkill /F /T /FI 'WINDOWTITLE eq Detector' > $null 2>&1;" ^
  "taskkill /F /T /FI 'WINDOWTITLE eq UI' > $null 2>&1;" ^
  "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'run_dual.bat' -WorkingDirectory $dir;"

endlocal
