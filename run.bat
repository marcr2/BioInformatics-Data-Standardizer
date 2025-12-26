@echo off
echo Starting BIDS - Bioinformatics Data Standardizer...
echo.

:: Check if venv exists
if not exist venv (
    echo ERROR: Virtual environment not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

:: Activate venv and run
call venv\Scripts\activate.bat
python main.py

if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
