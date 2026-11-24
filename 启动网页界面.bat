@echo off
rem ============================================================
rem  FVA launcher (ASCII only on purpose: Chinese text inside a
rem  .bat file breaks cmd's parser when the console codepage
rem  differs from the file encoding. All user-facing Chinese
rem  messages are printed by the Python program instead.)
rem ============================================================
chcp 65001 >nul 2>nul
setlocal EnableExtensions
cd /d "%~dp0"
title FVA - Web UI

echo ==================================================
echo    Function Vulnerability Agent (FVA)
echo ==================================================
echo.

call :pick_python
if errorlevel 1 goto :need_deps

if not exist "cache\kb_index.pkl.gz" (
    echo [1/2] Building the knowledge base index for the first run ...
    %PY% -m fva.cli build-kb
    if errorlevel 1 goto :failed
) else (
    echo [1/2] Knowledge base index: ready
)

echo [2/2] Starting the web UI. Your browser will open http://localhost:8501
echo       Close this window to stop the server.
echo.
%PY% -m fva.cli ui
if errorlevel 1 goto :failed

echo.
echo UI stopped.
pause
exit /b 0

rem ------------------------------------------------------------
:pick_python
set "PY="
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import streamlit, fva" >nul 2>nul
    if not errorlevel 1 (
        set "PY=.venv\Scripts\python.exe"
        exit /b 0
    )
)
python -c "import streamlit, fva" >nul 2>nul
if not errorlevel 1 (
    set "PY=python"
    exit /b 0
)
py -3 -c "import streamlit, fva" >nul 2>nul
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
echo (If you use a virtual environment, make sure .venv\Scripts\python.exe exists.)
echo.
pause
exit /b 1

:failed
echo.
echo [ERROR] The last step failed. Please send us the messages above.
echo.
pause
exit /b 1
