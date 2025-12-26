# BIDS - Bioinformatics Data Standardizer

A powerful system for cleaning and standardizing messy clinical/bioinformatics data into IPA (Ingenuity Pathway Analysis) format or custom CSV schemas.

## Features

- **Universal Ingestion**: Accepts any file type (tar, gz, zip, rar, 7z, nested archives) and extracts tabular data
- **AI-Powered Format Detection**: Uses Gemini LLM to parse unknown file formats
- **Noise-Matching RAG**: Learns from past fixes using vector similarity search
- **Multi-LLM Architecture**: Gemini for diagnostics, Claude Opus for fix generation
- **Custom Schema Editor**: Build your own output CSV structure via GUI
- **Modern GUI**: Dear PyGui-powered interface with real-time processing

## Installation

### Prerequisites
- Python 3.10 or higher
- Windows OS (for install.bat)

### Quick Start

1. **Clone/Download** this repository
2. **Run** `install.bat` to set up the environment
3. **Edit** `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_gemini_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```
4. **Launch** with `run.bat`

## Project Structure

```
BIDS/
├── install.bat           # Windows installation script
├── run.bat               # Launch script
├── requirements.txt      # Python dependencies
├── .env                  # API keys (create from .env.example)
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

## Output Schemas

### IPA Standard
- `GeneSymbol` (string): Gene identifier
- `FoldChange` (float): Expression fold change
- `PValue` (float): Statistical significance

### Custom Schemas
Create your own via the Schema Editor with:
- Custom column names and types
- Column mapping from input
- Transformation rules

## License

MIT License - See LICENSE file
