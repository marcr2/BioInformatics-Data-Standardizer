@echo off
echo ========================================
echo  BIDS - Bioinformatics Data Standardizer
echo  Installation Script
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

:: Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)
echo.

:: Activate and install
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo.

:: Create .env from example
if not exist .env (
    copy .env.example .env
    echo Created .env file - please add your API keys!
) else (
    echo .env file already exists.
)

:: Create schemas directory
if not exist schemas mkdir schemas

:: Create data directories
if not exist data mkdir data
if not exist data\input mkdir data\input
if not exist data\output mkdir data\output

echo.
echo ========================================
echo  Installation complete!
echo.
echo  Next steps:
echo  1. Edit .env file and add your API keys
echo  2. Run: run.bat
echo ========================================
pause
