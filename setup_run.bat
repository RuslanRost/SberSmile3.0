@echo off
setlocal enabledelayedexpansion

REM Resolve script directory (repo root if run from there)
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM 0) Ensure Git is installed
where /q git.exe
if errorlevel 1 (
  echo Installing Git...
  winget install --id Git.Git --accept-source-agreements --accept-package-agreements --silent
) else (
  echo Git found.
)

REM 1) Ensure Visual C++ Build Tools are installed (silent install)
where /q cl.exe
if errorlevel 1 (
  echo Installing Visual C++ Build Tools...
  winget install --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements --accept-package-agreements --silent --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait"
) else (
  echo Visual C++ Build Tools found.
)

REM 2) Ensure Python 3.10 is installed (uses py launcher)
py -3.10 -V >nul 2>&1
if errorlevel 1 (
  echo Installing Python 3.10 via winget...
  winget install --id Python.Python.3.10 --accept-source-agreements --accept-package-agreements --silent
)

REM 3) Sync repository (pull latest)
if exist ".git" (
  echo Pulling latest changes...
  git pull
) else (
  echo Warning: .git not found, skipping git pull.
)

REM 4) Create venv with Python 3.10
if not exist "venv\Scripts\python.exe" (
  echo Creating venv...
  py -3.10 -m venv venv
)

REM 5) Install dependencies
echo Installing dependencies...
.\venv\Scripts\python -m pip install --upgrade pip wheel
.\venv\Scripts\python -m pip install -r requirements.txt
.\venv\Scripts\python -m pip install --no-cache-dir --no-build-isolation dlib==19.24.0

endlocal
