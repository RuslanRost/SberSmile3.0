@echo off
set "SCRIPT_DIR=C:\Users\User\Documents\GitHub\smile_detection-main\SberSmile3.0\"
timeout /t 30 /nobreak >nul
taskkill /F /T /FI "WINDOWTITLE eq Detector*" >nul 2>&1
taskkill /F /T /FI "WINDOWTITLE eq UI*" >nul 2>&1
for /f %%P in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process ^| Where-Object { $_.CommandLine -match ''detector_server\.py^|ui_client\.py'' } ^| ForEach-Object { $_.ProcessId }"') do taskkill /F /PID %%P >nul 2>&1
timeout /t 2 /nobreak >nul
start "" /D "C:\Users\User\Documents\GitHub\smile_detection-main\SberSmile3.0\" cmd /c "run_dual.bat"
