@echo off
rem ============================================================
rem  FVA batch evaluation launcher.
rem  ASCII only: Chinese text inside a .bat breaks cmd's parser
rem  when the console codepage differs from the file encoding.
rem ============================================================
chcp 65001 >nul 2>nul
setlocal EnableExtensions
cd /d "%~dp0"
title FVA - Batch evaluation

echo ==================================================
echo    Batch detection + course evaluation
echo ==================================================
echo.

call :pick_python
if errorlevel 1 goto :need_deps

if not exist "cache\kb_index.pkl.gz" (
    echo [setup] Building the knowledge base index ...
    %PY% -m fva.cli build-kb
    if errorlevel 1 goto :failed
)

echo [1/2] Batch detection (resumable: finished samples are skipped)
echo       Extra arguments are passed to the batch command, e.g.
echo         %~nx0 --mode offline --no-repo
echo.
%PY% -m fva.cli batch --workers 2 %*
if errorlevel 1 goto :failed

echo.
echo [2/2] Evaluating with the course metric
%PY% -m fva.cli eval
if errorlevel 1 goto :failed

echo.
echo Done.
echo   predictions : results\result.jsonl
echo   evaluation  : results\eval.json
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

:failed
echo.
echo [ERROR] The last step failed. Please send us the messages above.
echo.
pause
exit /b 1
