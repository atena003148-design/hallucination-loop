@echo off
title Hallucination Loop Engine
echo =========================================
echo Starting Hallucination Loop Engine...
echo =========================================
cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo Warning: Virtual environment 'venv' not found. Using global python.
)

:loop
echo.
echo [%time%] Running python main.py...
echo =========================================
python main.py
set EXIT_CODE=%ERRORLEVEL%

echo.
echo =========================================
if %EXIT_CODE% EQU 0 (
    echo [%time%] Completed successfully. Restarting in 15 seconds...
    timeout /t 15 /nobreak
) else (
    echo [%time%] Exited with error (code=%EXIT_CODE%).
    echo.
    echo Press any key to retry, or close this window to stop.
    pause >nul
    echo Retrying in 10 seconds...
    timeout /t 10 /nobreak
)
echo Press Ctrl+C to stop the loop.
echo =========================================
goto loop
