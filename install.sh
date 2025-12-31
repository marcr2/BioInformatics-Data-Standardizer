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

# Install PyTorch with CUDA support first (if available)
echo ""
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

# Install remaining dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

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
echo " Next steps:"
echo "  1. Run: ./run.sh"
echo "  2. On first launch, the system will download"
echo "     a local LLM model (~8-16GB, one-time download)"
echo "  3. No API keys needed - everything runs locally!"
echo ""
echo " Note: GPU with 8GB+ VRAM recommended for best performance"
echo "========================================"

