#!/bin/bash
set -e  # Exit on error

echo "========================================"
echo " BIDS - Bioinformatics Data Standardizer"
echo " Installation Script"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.10+"
    echo "Download from: https://www.python.org/downloads/"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Check Python version (3.10+)
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi
echo ""

# Activate and install
echo "Installing dependencies..."
source venv/bin/activate
python -m pip install --upgrade pip

# Ask user if they want to install LLM support
echo ""
echo "========================================"
echo " LLM Support Installation"
echo "========================================"
echo ""
echo " The application can work in two modes:"
echo "  1. Rules-based only (no LLM required)"
echo "  2. Full mode with LLM support (requires ~8-16GB for model download)"
echo ""
read -p "Do you want to install LLM support? (y/n): " INSTALL_LLM
INSTALL_LLM=$(echo "$INSTALL_LLM" | tr '[:upper:]' '[:lower:]')

if [ "$INSTALL_LLM" = "y" ] || [ "$INSTALL_LLM" = "yes" ]; then
    INSTALL_LLM_FLAG=1
    echo ""
    echo "Installing LLM support..."
    echo ""
    
    # Install PyTorch with CUDA support first (if available)
    echo "Checking for CUDA support..."
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected! Installing PyTorch with CUDA support..."
        echo "Installing PyTorch with CUDA 12.8 (supports Blackwell/RTX 50-series GPUs)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || {
            echo "Warning: Failed to install CUDA version, falling back to CPU version..."
            pip install torch torchvision torchaudio
        }
    else
        echo "No NVIDIA GPU detected or nvidia-smi not found."
        echo "Installing CPU-only PyTorch (will be slow for LLM inference)..."
        pip install torch torchvision torchaudio
    fi
else
    INSTALL_LLM_FLAG=0
    echo ""
    echo "Skipping LLM support installation."
    echo "You can install LLM support later from Preferences in the application."
    echo ""
fi

# Install core dependencies (always installed)
echo ""
echo "Installing core dependencies..."
pip install pandas>=2.0.0 numpy>=1.24.0
pip install python-magic>=0.4.14
pip install py7zr>=0.20.0 rarfile>=4.1 openpyxl>=3.1.0 pyarrow>=14.0.0
pip install scikit-learn>=1.3.0 chromadb>=0.4.0
pip install dearpygui>=1.10.0 vispy>=0.14.0 PyQt5>=5.15.0
pip install python-dotenv>=1.0.0

# Install LLM dependencies only if user chose to install them
if [ "$INSTALL_LLM_FLAG" = "1" ]; then
    echo ""
    echo "Installing LLM dependencies..."
    pip install transformers>=4.35.0 accelerate>=0.24.0 bitsandbytes>=0.41.0 sentencepiece>=0.1.99 protobuf>=3.20.0
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install dependencies."
    exit 1
fi
echo ""

# Note: No .env file needed - system is 100% local!

# Create schemas directory
mkdir -p schemas

# Create data directories
mkdir -p data/input
mkdir -p data/output

echo ""
echo "========================================"
echo " Installation complete!"
echo ""
if [ "$INSTALL_LLM_FLAG" = "1" ]; then
    echo " Next steps:"
    echo "  1. Run: ./run.sh"
    echo "  2. On first launch, the system will download"
    echo "     a local LLM model (~8-16GB, one-time download)"
    echo "  3. No API keys needed - everything runs locally!"
    echo ""
    echo " Note: GPU with 8GB+ VRAM recommended for best performance"
else
    echo " Next steps:"
    echo "  1. Run: ./run.sh"
    echo "  2. The application will run in rules-based mode"
    echo "  3. To enable LLM support later, go to Preferences"
    echo "     and click \"Install LLM\" button"
    echo ""
    echo " Note: Rules-based processing works without LLM,"
    echo "       but LLM support enables advanced features."
fi
echo "========================================"

