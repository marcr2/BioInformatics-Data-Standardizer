"""
BIDS - Bioinformatics Data Standardizer

Main entry point for the BIDS application.

Usage:
    python main.py           # Launch GUI
    python main.py --cli     # Run in CLI mode (for scripting)
    python main.py --help    # Show help
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import dearpygui
    except ImportError:
        missing.append("dearpygui")
    
    try:
        import chromadb
    except ImportError:
        missing.append("chromadb")
    
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    
    try:
        import magic
    except ImportError:
        missing.append("python-magic-bin")
    
    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease run: pip install -r requirements.txt")
        return False
    
    return True


def check_api_keys():
    """Check if API keys are configured."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    warnings = []
    
    if not os.getenv("GOOGLE_API_KEY"):
        warnings.append("GOOGLE_API_KEY not set (needed for diagnostics)")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        warnings.append("ANTHROPIC_API_KEY not set (needed for fix generation)")
    
    if warnings:
        print("API Key Warnings:")
        for warn in warnings:
            print(f"  - {warn}")
        print("\nYou can configure API keys in the Settings menu or .env file.")
        print()
    
    return len(warnings) == 0


def run_gui():
    """Launch the BIDS GUI application."""
    print("Starting BIDS - Bioinformatics Data Standardizer...")
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check API keys (warning only)
    check_api_keys()
    
    # Launch GUI
    try:
        from src.gui.app import BIDSApp
        app = BIDSApp()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_cli(args):
    """Run BIDS in CLI mode for scripting."""
    from src.ingestion import SmartIngestor
    from src.schema_manager import SchemaManager
    from src.agents import AgentOrchestrator
    
    print("BIDS CLI Mode")
    print("=" * 40)
    
    if not args.input:
        print("Error: --input file required")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Ingest file
    print(f"Loading: {input_path}")
    ingestor = SmartIngestor(use_llm_fallback=not args.no_llm)
    result = ingestor.ingest(input_path)
    
    if not result.dataframes:
        print(f"Error: Could not load data from file")
        for error in result.errors:
            print(f"  - {error}")
        sys.exit(1)
    
    df = result.dataframes[0]
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Get schema
    schema_manager = SchemaManager("schemas")
    schema_name = args.schema or "IPA Standard"
    schema = schema_manager.get_schema(schema_name)
    
    if not schema:
        print(f"Error: Schema '{schema_name}' not found")
        print(f"Available schemas: {', '.join(schema_manager.list_schemas())}")
        sys.exit(1)
    
    print(f"Using schema: {schema_name}")
    
    # Process
    if args.diagnose_only:
        from src.agents import DiagnosticAgent
        agent = DiagnosticAgent()
        diagnosis = agent.diagnose(df, schema, schema_manager)
        
        print(f"\nDiagnosis Result:")
        print(f"  Valid: {diagnosis.is_valid}")
        print(f"  Quality Score: {diagnosis.quality_score:.2f}")
        print(f"  Issues: {len(diagnosis.issues)}")
        
        for issue in diagnosis.issues:
            print(f"\n  [{issue.get('severity', 'info').upper()}] {issue.get('column', 'unknown')}")
            print(f"    {issue.get('description', 'No description')}")
    else:
        # Full process
        print("Running full processing pipeline...")
        orchestrator = AgentOrchestrator(
            vector_store_path="data/vector_store",
            schemas_dir="schemas"
        )
        
        result = orchestrator.process(
            df,
            schema_name=schema_name,
            auto_fix=True,
            max_attempts=args.max_attempts
        )
        
        print(f"\nProcessing Result:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Fix Attempts: {len(result.get('fix_attempts', []))}")
        
        # Export result
        if result.get("final_df") is not None and args.output:
            output_path = Path(args.output)
            final_df = result["final_df"]
            
            if output_path.suffix == ".csv":
                final_df.to_csv(output_path, index=False)
            elif output_path.suffix == ".xlsx":
                final_df.to_excel(output_path, index=False)
            elif output_path.suffix == ".parquet":
                final_df.to_parquet(output_path, index=False)
            else:
                final_df.to_csv(output_path, index=False)
            
            print(f"Exported to: {output_path}")
    
    print("\nDone!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BIDS - Bioinformatics Data Standardizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Launch GUI
  python main.py --cli -i data.csv        # Diagnose file
  python main.py --cli -i data.csv -o out.csv  # Process and export
        """
    )
    
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode (no GUI)"
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input file path (CLI mode)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (CLI mode)"
    )
    
    parser.add_argument(
        "-s", "--schema",
        type=str,
        default="IPA Standard",
        help="Schema name to use (default: 'IPA Standard')"
    )
    
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Only run diagnosis, don't attempt fixes"
    )
    
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum fix attempts (default: 3)"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM fallback for file parsing"
    )
    
    args = parser.parse_args()
    
    if args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()

