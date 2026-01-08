@echo off
echo ========================================
echo  BIDS - Bioinformatics Data Standardizer
echo  Installation Script
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
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
:: Verify activation by checking pip
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment or pip not available.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

:: Install core dependencies FIRST (always installed)
echo.
echo ========================================
echo  Installing Core Dependencies
echo ========================================
echo.
echo Installing base dependencies...
pip install pandas>=2.0.0 numpy>=1.24.0
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install pandas/numpy.
    pause
    exit /b 1
)

pip install python-magic-bin>=0.4.14
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install python-magic-bin.
    pause
    exit /b 1
)

pip install py7zr>=0.20.0 rarfile>=4.1 openpyxl>=3.1.0 pyarrow>=14.0.0
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install archive/file format libraries.
    pause
    exit /b 1
)

pip install scikit-learn>=1.3.0 chromadb>=0.4.0
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install scikit-learn/chromadb.
    pause
    exit /b 1
)

pip install dearpygui>=1.10.0 vispy>=0.14.0 PyQt5>=5.15.0
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install GUI libraries.
    pause
    exit /b 1
)

pip install python-dotenv>=1.0.0
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install python-dotenv.
    pause
    exit /b 1
)

echo.
echo Core dependencies installed successfully!
echo.

:: Ask user if they want to install LLM support
echo ========================================
echo  LLM Support Installation (Optional)
echo ========================================
echo.
echo  The application can work in two modes:
echo  1. Rules-based only (no LLM required) - Already installed!
echo  2. Full mode with LLM support (requires ~8-16GB for model download)
echo.
set /p INSTALL_LLM="Do you want to install LLM support? (Y/N): "

:: Initialize LLM flag (default to 0 - no LLM)
set INSTALL_LLM_FLAG=0

if /i "%INSTALL_LLM%"=="Y" (
    set INSTALL_LLM_FLAG=1
    echo.
    echo Installing LLM support...
    echo.
    
    :: Install PyTorch with CUDA support first (if available)
    echo Checking for CUDA support...
    python -c "import subprocess; import sys; result = subprocess.run(['nvidia-smi'], capture_output=True); sys.exit(0 if result.returncode == 0 else 1)" 2>nul
    if %errorlevel% equ 0 (
        echo NVIDIA GPU detected! Installing PyTorch with CUDA support...
        echo Installing PyTorch with CUDA 12.8 (supports Blackwell/RTX 50-series GPUs)...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        if %errorlevel% neq 0 (
            echo Warning: Failed to install CUDA version, falling back to CPU version...
            pip install torch torchvision torchaudio
            if %errorlevel% neq 0 (
                echo ERROR: Failed to install PyTorch.
                echo Warning: LLM installation failed, but core system is installed.
                echo You can install LLM support later from Preferences in the application.
                set INSTALL_LLM_FLAG=0
                goto :finish_installation
            )
        )
    ) else (
        echo No NVIDIA GPU detected or nvidia-smi not found.
        echo Installing CPU-only PyTorch (will be slow for LLM inference)...
        pip install torch torchvision torchaudio
        if %errorlevel% neq 0 (
            echo ERROR: Failed to install PyTorch.
            echo Warning: LLM installation failed, but core system is installed.
            echo You can install LLM support later from Preferences in the application.
            set INSTALL_LLM_FLAG=0
            goto :finish_installation
        )
    )
    
    :: Install LLM dependencies
    echo.
    echo Installing LLM dependencies...
    pip install transformers>=4.35.0 accelerate>=0.24.0 bitsandbytes>=0.41.0 sentencepiece>=0.1.99 protobuf>=3.20.0
    if %errorlevel% neq 0 (
        echo.
        echo Warning: Failed to install some LLM dependencies.
        echo Core system is installed and functional.
        echo You can install LLM support later from Preferences in the application.
        set INSTALL_LLM_FLAG=0
    ) else (
        echo.
        echo LLM dependencies installed successfully!
    )
) else (
    set INSTALL_LLM_FLAG=0
    echo.
    echo Skipping LLM support installation.
    echo You can install LLM support later from Preferences in the application.
    echo.
)

:finish_installation
echo.

:: Create required directories
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
if "%INSTALL_LLM_FLAG%"=="1" (
    echo  Next steps:
    echo  1. Run: run.bat
    echo  2. On first launch, the system will download
    echo     a local LLM model (~8-16GB, one-time download)
    echo  3. No API keys needed - everything runs locally!
    echo.
    echo  Note: GPU with 8GB+ VRAM recommended for best performance
) else (
    echo  Next steps:
    echo  1. Run: run.bat
    echo  2. The application will run in rules-based mode
    echo  3. To enable LLM support later, go to Preferences
    echo     and click "Install LLM" button
    echo.
    echo  Note: Rules-based processing works without LLM,
    echo        but LLM support enables advanced features.
)
echo ========================================
pause
