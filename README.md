# BIDS - Bioinformatics Data Standardizer

A powerful system for cleaning and standardizing messy clinical/bioinformatics data into custom CSV schemas.

## Features

- **Universal Ingestion**: Accepts any file type (tar, gz, zip, rar, 7z, nested archives) and extracts tabular data
- **AI-Powered Format Detection**: Uses local LLM to parse unknown file formats
- **Noise-Matching RAG**: Learns from past fixes using vector similarity search
- **100% Local & Private**: All LLM processing runs locally on your GPU - no API calls, no data leaves your machine
- **Custom Schema Editor**: Build your own output CSV structure via GUI
- **Modern GUI**: Dear PyGui-powered interface with real-time processing

## Installation

### Prerequisites
- Python 3.10 or higher
- Windows OS (for install.bat)
- NVIDIA GPU with 8GB+ VRAM recommended (16GB+ for best quality)
  - GPU is optional but highly recommended for performance
  - CPU-only mode is supported but will be slow

### Quick Start

1. **Clone/Download** this repository
2. **Run** `install.bat` to set up the environment
   - This will install all dependencies including PyTorch and transformers
   - On first run, the system will automatically download a local LLM model (~8-16GB)
   - Model selection is automatic based on your GPU memory
3. **(Optional)** For best quality models (Llama 3.1), set HuggingFace token:
   - Get a free token from https://huggingface.co/settings/tokens
   - Set environment variable: `set HUGGINGFACE_TOKEN=your_token_here`
   - Or add to `.env` file: `HUGGINGFACE_TOKEN=your_token_here`
   - Note: Most models (Mistral, Qwen, Phi-3) work without authentication
4. **Launch** with `run.bat`
   - No API keys needed - everything runs locally!
   - First launch may take a few minutes to download the model

## Project Structure

```
BIDS/
├── install.bat           # Windows installation script
├── run.bat               # Launch script
├── requirements.txt      # Python dependencies
├── main.py               # Entry point
├── schemas/              # Saved custom schemas
└── src/
    ├── ingestion.py      # SmartIngestor + FormatScout
    ├── noise_generator.py
    ├── vectorizer.py     # TF-IDF fingerprinting
    ├── vector_store.py   # ChromaDB wrapper
    ├── agents.py         # Diagnostic + Fixing agents
    ├── schema_manager.py # IPA + Custom schemas
    ├── gui/              # Dear PyGui interface
    └── utils/            # LLM clients, file handlers
```

## Usage

1. **Add Files**: Drag & drop or browse for input files
2. **Preview**: Review detected columns and data types
3. **Select Schema**: Choose IPA Standard or create a custom schema
4. **Diagnose**: Run AI diagnostics to identify data issues
5. **Fix & Export**: Generate and apply fixes, export cleaned data

### Custom Schemas
Create your own via the Schema Editor with:
- Custom column names and types
- Column mapping from input
- Transformation rules

## License

GPL-3.0 License - See LICENSE file

## AI use
Built using Cursor. See instructions.md for original prompting.
