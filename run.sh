#!/bin/bash

echo "Starting BIDS - Bioinformatics Data Standardizer..."
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set HF_HOME to models directory in project root (works for any user/path)
export HF_HOME="$SCRIPT_DIR/models"
mkdir -p "$HF_HOME"
echo "Using HuggingFace model directory: $HF_HOME"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run ./install.sh first."
    exit 1
fi

# Activate venv and run
source venv/bin/activate
python main.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with an error."
    exit 1
fi

