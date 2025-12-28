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

endlocal
