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

:: Install PyTorch with CUDA support first (if available)
echo.
echo Checking for CUDA support...
python -c "import subprocess; import sys; result = subprocess.run(['nvidia-smi'], capture_output=True); sys.exit(0 if result.returncode == 0 else 1)" 2>nul
if errorlevel 0 (
    echo NVIDIA GPU detected! Installing PyTorch with CUDA support...
    echo Installing PyTorch with CUDA 12.8 (supports Blackwell/RTX 50-series GPUs)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo Warning: Failed to install CUDA version, falling back to CPU version...
        pip install torch torchvision torchaudio
    )
) else (
    echo No NVIDIA GPU detected or nvidia-smi not found.
    echo Installing CPU-only PyTorch (will be slow for LLM inference)...
    pip install torch torchvision torchaudio
)

:: Install remaining dependencies
echo.
echo Installing other dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo.

:: Note: No .env file needed - system is 100% local!

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
echo  1. Run: run.bat
echo  2. On first launch, the system will download
echo     a local LLM model (~8-16GB, one-time download)
echo  3. No API keys needed - everything runs locally!
echo.
echo  Note: GPU with 8GB+ VRAM recommended for best performance
echo ========================================
pause
