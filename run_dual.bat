@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "RESTART_HHMM=11:21"
set "RESTART_DELAY_SEC=30"
set "TASK_NAME=SberSmileDailyRestart"
set "TASK_SCRIPT=%SCRIPT_DIR%restart_task.bat"
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

REM Create/update a Windows Scheduled Task for daily restart at %RESTART_HHMM% with %RESTART_DELAY_SEC%s delay
(
  echo @echo off
  echo set "SCRIPT_DIR=%SCRIPT_DIR%"
  echo timeout /t %RESTART_DELAY_SEC% /nobreak ^>nul
  echo taskkill /F /T /FI "WINDOWTITLE eq Detector*" ^>nul 2^>^&1
  echo taskkill /F /T /FI "WINDOWTITLE eq UI*" ^>nul 2^>^&1
  echo for /f %%%%P in ^('powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process ^| Where-Object { $_.CommandLine -match ''detector_server\.py^|ui_client\.py'' } ^| ForEach-Object { $_.ProcessId }"'^) do taskkill /F /PID %%%%P ^>nul 2^>^&1
  echo timeout /t 2 /nobreak ^>nul
  echo start "" /D "%SCRIPT_DIR%" cmd /c "run_dual.bat"
) > "%TASK_SCRIPT%"

schtasks /Create /F /SC DAILY /TN "%TASK_NAME%" /ST %RESTART_HHMM% /RL LIMITED /TR "\"%TASK_SCRIPT%\"" >nul
echo Scheduled Task "%TASK_NAME%" set for %RESTART_HHMM% (delay %RESTART_DELAY_SEC%s).
