@echo off
rem ============================================================
rem  FVA LLM configuration helper.
rem  ASCII only + CRLF: Chinese text inside a .bat breaks cmd's
rem  parser when the console codepage differs from the file.
rem ============================================================
chcp 65001 >nul 2>nul
setlocal EnableExtensions
cd /d "%~dp0"
title FVA - Configure LLM

echo ==================================================
echo    Configure the large language model
echo ==================================================
echo.
echo This will write FVA_LLM_API_KEY / FVA_LLM_MODEL / FVA_LLM_BASE_URL
echo into the .env file in this folder, then verify the connection.
echo.

call :pick_python
if errorlevel 1 goto :need_deps

%PY% -m fva.cli setup
echo.
pause
exit /b 0

rem ------------------------------------------------------------
:pick_python
set "PY="
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import fva" >nul 2>nul
    if not errorlevel 1 (
        set "PY=.venv\Scripts\python.exe"
        exit /b 0
    )
)
python -c "import fva" >nul 2>nul
if not errorlevel 1 (
    set "PY=python"
    exit /b 0
)
py -3 -c "import fva" >nul 2>nul
if not errorlevel 1 (
    set "PY=py -3"
    exit /b 0
)
exit /b 1

rem ------------------------------------------------------------
:need_deps
echo [ERROR] No usable Python environment was found.
echo.
echo Run the following once in this folder:
echo     python -m pip install -r requirements.txt
echo then double-click this file again.
echo.
pause
exit /b 1
