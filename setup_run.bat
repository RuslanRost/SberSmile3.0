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

REM 2) Ensure CMake is installed (from cmake.org)
cmake --version >nul 2>&1
if errorlevel 1 (
  echo Installing CMake from cmake.org...
  set "CMAKE_VER=3.26.4"
  set "CMAKE_EXE=%TEMP%\cmake-%CMAKE_VER%-windows-x86_64.msi"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri https://github.com/Kitware/CMake/releases/download/v%CMAKE_VER%/cmake-%CMAKE_VER%-windows-x86_64.msi -OutFile '%CMAKE_EXE%'" || (
    echo Download failed, trying winget...
    winget install --id Kitware.CMake --accept-source-agreements --accept-package-agreements --silent
    goto :cmake_check
  )
  msiexec /i "%CMAKE_EXE%" /qn ADD_CMAKE_TO_PATH=System
)
:cmake_check
cmake --version >nul 2>&1
if errorlevel 1 (
  echo CMake installation failed or not in PATH. Please install from https://cmake.org/download/ and reopen the terminal.
  pause
  goto :eof
)

REM 3) Ensure Python 3.10 is installed (uses py launcher)
py -3.10 -V >nul 2>&1
if errorlevel 1 (
  echo Installing Python 3.10 via winget...
  winget install --id Python.Python.3.10 --accept-source-agreements --accept-package-agreements --silent
)

REM 4) Sync repository (pull latest)
if exist ".git" (
  echo Pulling latest changes...
  git pull
) else (
  echo Warning: .git not found, skipping git pull.
)

REM 5) Create venv with Python 3.10
if not exist "venv\Scripts\python.exe" (
  echo Creating venv...
  py -3.10 -m venv venv
)

REM 6) Install dependencies
echo Installing dependencies...
.\venv\Scripts\python -m pip install --upgrade pip wheel
if errorlevel 1 goto :error

REM Install requirements except dlib to avoid build isolation issues
set "REQ_TMP=%TEMP%\\requirements-nodlib.txt"
findstr /V /I "dlib" requirements.txt > "%REQ_TMP%"
.\venv\Scripts\python -m pip install -r "%REQ_TMP%"
if errorlevel 1 goto :error

REM Install dlib with no build isolation so it sees cmake/numpy
.\venv\Scripts\python -m pip install --no-cache-dir --no-build-isolation dlib==19.24.0
if errorlevel 1 goto :error

echo Setup completed successfully.
goto :eof

:error
echo.
echo Setup failed. Check the output above.
pause

endlocal
