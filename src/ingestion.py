"""
Universal Ingestion Engine for BIDS

SmartIngestor: Three-layer file ingestion system
- Layer 1: Magic byte detection
- Layer 2: Standard extractors (zip, tar, gz, rar, 7z)
- Layer 3: LLM-powered FormatScout fallback
"""

import io
import warnings
import unicodedata
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Suppress SyntaxWarning from python-magic-bin package
warnings.filterwarnings("ignore", category=SyntaxWarning, module="magic")
import magic

from .utils.file_handlers import (
    ArchiveExtractor, 
    ExtractedFile, 
    read_file_header,
    get_file_info
)
from .utils.llm_client import LocalLLMClient, get_gemini_client


class IngestionStatus(Enum):
    """Status of ingestion attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some files extracted but not all
    FALLBACK_USED = "fallback_used"  # LLM was needed
    FAILED = "failed"


@dataclass
class IngestionResult:
    """Result of ingestion operation."""
    status: IngestionStatus
    dataframes: List[pd.DataFrame]
    source_files: List[str]
    errors: List[str]
    metadata: Dict[str, Any]


class FormatScout:
    """
    LLM-powered format detection and parsing for unknown file types.
    
    Uses local LLM to:
    1. Identify unknown file formats
    2. Generate Python parsing code
    3. Execute parsing to extract DataFrames
    """
    
    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        """
        Initialize FormatScout.
        
        Args:
            llm_client: Local LLM client (creates one lazily if not provided)
        """
        # Store client directly if provided, otherwise None for lazy loading
        self._llm_client = llm_client
        self._sandbox_globals = {
            'pd': pd,
            'io': io,
            'Path': Path,
        }
    
    @property
    def llm(self) -> LocalLLMClient:
        """Lazy-load LLM client only when accessed."""
        if self._llm_client is None:
            self._llm_client = get_gemini_client()
        return self._llm_client
    
    @llm.setter
    def llm(self, value: Optional[LocalLLMClient]) -> None:
        """Set LLM client directly."""
        self._llm_client = value
    
    def identify_format(self, file_path: Path) -> Dict[str, Any]:
        """
        Use LLM to identify an unknown file format.
        
        Args:
            file_path: Path to the unknown file
            
        Returns:
            Dict with format identification results
        """
        header = read_file_header(file_path, 1024)
        return self.llm.identify_file_format(header, file_path.name)
    
    def parse_unknown_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Attempt to parse an unknown file using LLM-generated code.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        # Get format identification
        format_info = self.identify_format(file_path)
        
        if not format_info.get('is_tabular'):
            return None
        
        # Try the suggested Python snippet
        python_code = format_info.get('python_snippet')
        if python_code:
            df = self._execute_parsing_code(python_code, file_path)
            if df is not None:
                return df
        
        # If that fails, ask for a more detailed parsing solution
        return self._generate_and_execute_parser(file_path, format_info)
    
    def _execute_parsing_code(
        self, 
        code: str, 
        file_path: Path
    ) -> Optional[pd.DataFrame]:
        """
        Execute LLM-generated parsing code in a sandbox.
        
        Args:
            code: Python code to execute
            file_path: Path to the file being parsed
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            # Prepare sandbox environment
            sandbox = self._sandbox_globals.copy()
            sandbox['file_path'] = str(file_path)
            sandbox['__builtins__'] = {
                'open': open,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'print': print,
            }
            
            # Execute the code
            exec(code, sandbox)
            
            # Look for a DataFrame result
            if 'df' in sandbox and isinstance(sandbox['df'], pd.DataFrame):
                return sandbox['df']
            if 'result' in sandbox and isinstance(sandbox['result'], pd.DataFrame):
                return sandbox['result']
            
            return None
            
        except Exception as e:
            print(f"FormatScout execution error: {e}")
            return None
    
    def _generate_and_execute_parser(
        self, 
        file_path: Path, 
        format_info: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Generate a custom parser for a difficult file.
        
        Args:
            file_path: Path to the file
            format_info: Previously identified format information
            
        Returns:
            DataFrame if successful, None otherwise
        """
        # Read file content sample
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(4096)
        except Exception:
            with open(file_path, 'rb') as f:
                sample = f.read(4096).decode('utf-8', errors='replace')
        
        prompt = f"""Generate Python code to parse this file into a pandas DataFrame.

File info:
- Name: {file_path.name}
- Detected format: {format_info.get('format', 'unknown')}
- Parsing hint: {format_info.get('parsing_hint', 'none')}

File content sample:
```
{sample[:2000]}
```

Requirements:
1. The code should read from the variable `file_path` (a string)
2. Store the result in a variable called `df`
3. Handle encoding issues gracefully
4. Skip header rows if necessary
5. Clean column names (remove whitespace, special chars)

Generate ONLY the Python code, no explanations."""

        code = self.llm.generate_code(prompt)
        return self._execute_parsing_code(code, file_path)


class SmartIngestor:
    """
    Universal file ingestion engine with three-layer architecture.
    
    Layer 1: Magic byte detection for file type identification
    Layer 2: Standard extractors for common archive formats
    Layer 3: FormatScout LLM fallback for unknown formats
    """
    
    # Standard tabular file readers with encoding handling
    TABULAR_READERS = {
        '.csv': lambda p: SmartIngestor._read_csv_with_encoding(p),
        '.tsv': lambda p: SmartIngestor._read_csv_with_encoding(p, sep='\t'),
        '.xlsx': lambda p: pd.read_excel(p, engine='openpyxl'),
        '.xls': lambda p: pd.read_excel(p),
        '.parquet': lambda p: pd.read_parquet(p),
        '.json': lambda p: pd.read_json(p),
    }
    
    # MIME type to reader mapping
    MIME_READERS = {
        'text/csv': lambda p: SmartIngestor._read_csv_with_encoding(p),
        'text/tab-separated-values': lambda p: SmartIngestor._read_csv_with_encoding(p, sep='\t'),
        'application/vnd.ms-excel': lambda p: pd.read_excel(p),
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 
            lambda p: pd.read_excel(p, engine='openpyxl'),
    }
    
    def __init__(
        self, 
        use_llm_fallback: bool = True,
        gemini_client: Optional[LocalLLMClient] = None
    ):
        """
        Initialize SmartIngestor.
        
        Args:
            use_llm_fallback: Whether to use LLM for unknown formats
            gemini_client: Optional pre-configured local LLM client
        """
        self.use_llm_fallback = use_llm_fallback
        self.extractor = ArchiveExtractor()
        self.format_scout = FormatScout(gemini_client) if use_llm_fallback else None
    
    def ingest(self, file_path: Union[str, Path]) -> IngestionResult:
        """
        Ingest a file and extract all tabular data.
        
        Args:
            file_path: Path to the file (can be archive or direct data file)
            
        Returns:
            IngestionResult with DataFrames and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return IngestionResult(
                status=IngestionStatus.FAILED,
                dataframes=[],
                source_files=[],
                errors=[f"File not found: {file_path}"],
                metadata={}
            )
        
        dataframes: List[pd.DataFrame] = []
        source_files: List[str] = []
        errors: List[str] = []
        used_fallback = False
        
        # Layer 1 & 2: Extract archives and identify files
        extracted = self.extractor.extract(file_path)
        
        if not extracted:
            # Single file, not an archive
            extracted = [ExtractedFile(
                path=file_path,
                original_name=file_path.name,
                size=file_path.stat().st_size,
                mime_type=magic.from_file(str(file_path), mime=True),
                is_tabular=True  # Assume and try
            )]
        
        # Process each extracted file
        for ext_file in extracted:
            if not ext_file.is_tabular:
                continue
            
            df, error, fallback = self._read_tabular_file(ext_file)
            
            if df is not None:
                dataframes.append(df)
                source_files.append(ext_file.original_name)
                if fallback:
                    used_fallback = True
            elif error:
                errors.append(f"{ext_file.original_name}: {error}")
        
        # Determine status
        if not dataframes:
            status = IngestionStatus.FAILED
        elif errors:
            status = IngestionStatus.PARTIAL
        elif used_fallback:
            status = IngestionStatus.FALLBACK_USED
        else:
            status = IngestionStatus.SUCCESS
        
        return IngestionResult(
            status=status,
            dataframes=dataframes,
            source_files=source_files,
            errors=errors,
            metadata={
                "total_extracted": len(extracted),
                "tabular_files": len(dataframes),
                "used_llm_fallback": used_fallback
            }
        )
    
    def _read_tabular_file(
        self, 
        ext_file: ExtractedFile
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], bool]:
        """
        Read a tabular file using standard methods or LLM fallback.
        
        Args:
            ext_file: ExtractedFile to read
            
        Returns:
            Tuple of (DataFrame or None, error message or None, used_fallback)
        """
        file_path = ext_file.path
        used_fallback = False
        
        # Try standard readers based on extension
        suffix = file_path.suffix.lower()
        if suffix in self.TABULAR_READERS:
            try:
                df = self.TABULAR_READERS[suffix](file_path)
                return df, None, False
            except Exception as e:
                pass  # Fall through to next method
        
        # Try based on MIME type
        if ext_file.mime_type in self.MIME_READERS:
            try:
                df = self.MIME_READERS[ext_file.mime_type](file_path)
                return df, None, False
            except Exception:
                pass
        
        # Try generic CSV/TSV detection
        df = self._try_generic_tabular(file_path)
        if df is not None:
            return df, None, False
        
        # Layer 3: LLM Fallback
        if self.use_llm_fallback and self.format_scout:
            try:
                df = self.format_scout.parse_unknown_file(file_path)
                if df is not None:
                    return df, None, True
            except Exception as e:
                return None, f"LLM fallback failed: {e}", True
        
        return None, "Could not parse file with any method", False
    
    def _try_generic_tabular(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Try to read file as generic tabular data.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        # Try common delimiters
        delimiters = [',', '\t', ';', '|']
        
        for delim in delimiters:
            try:
                df = SmartIngestor._read_csv_with_encoding(file_path, sep=delim)
                # Check if it looks like valid tabular data
                if len(df.columns) > 1 and len(df) > 0:
                    return df
            except Exception:
                continue
        
        return None
    
    @staticmethod
    def _read_csv_with_encoding(file_path: Path, sep=',', **kwargs) -> pd.DataFrame:
        """
        Read CSV file with encoding detection and non-ASCII character replacement.
        
        Args:
            file_path: Path to CSV file
            sep: Separator character
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with non-ASCII characters replaced
        """
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    encoding=encoding,
                    on_bad_lines='skip',
                    **kwargs
                )
                used_encoding = encoding
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                # Other errors, try next encoding
                continue
        
        if df is None:
            # Last resort: read with errors='replace' to handle any encoding issues
            try:
                # Read file with error handling
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    df = pd.read_csv(
                        f,
                        sep=sep,
                        on_bad_lines='skip',
                        **kwargs
                    )
            except Exception:
                # Final fallback: use latin-1 which can read any byte
                try:
                    df = pd.read_csv(
                        file_path,
                        sep=sep,
                        encoding='latin-1',
                        on_bad_lines='skip',
                        **kwargs
                    )
                except Exception:
                    # Absolute last resort
                    with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
                        df = pd.read_csv(
                            f,
                            sep=sep,
                            on_bad_lines='skip',
                            **kwargs
                        )
        
        # Replace non-ASCII characters in string columns
        df = SmartIngestor._replace_non_ascii(df)
        
        return df
    
    @staticmethod
    def _replace_non_ascii(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace non-ASCII characters with ASCII equivalents in string columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with non-ASCII characters replaced
        """
        result = df.copy()
        
        for col in result.select_dtypes(include=['object']).columns:
            # Store original NaN mask before conversion
            nan_mask = result[col].isna()
            
            # Convert to string, handling NaN values
            col_str = result[col].astype(str)
            
            # Replace non-ASCII characters
            def replace_char(s):
                # Skip if it was originally NaN (now 'nan' string)
                if s == 'nan':
                    return s
                try:
                    # Normalize to NFKD (decomposed form) and remove non-ASCII
                    normalized = unicodedata.normalize('NFKD', str(s))
                    # Keep only ASCII characters
                    ascii_only = normalized.encode('ascii', 'ignore').decode('ascii')
                    return ascii_only
                except Exception:
                    # If replacement fails, return original
                    return s
            
            col_str = col_str.apply(replace_char)
            
            # Restore original NaN values
            col_str[nan_mask] = pd.NA
            
            result[col] = col_str
        
        return result
    
    def ingest_multiple(
        self, 
        file_paths: List[Union[str, Path]]
    ) -> List[IngestionResult]:
        """
        Ingest multiple files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of IngestionResult objects
        """
        return [self.ingest(fp) for fp in file_paths]
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        self.extractor.cleanup()
    
    def get_file_preview(
        self, 
        file_path: Union[str, Path], 
        max_rows: int = 10
    ) -> Dict[str, Any]:
        """
        Get a preview of file contents without full ingestion.
        
        Args:
            file_path: Path to the file
            max_rows: Maximum rows to preview
            
        Returns:
            Dict with preview information
        """
        file_path = Path(file_path)
        info = get_file_info(file_path)
        
        result = self.ingest(file_path)
        
        if result.dataframes:
            df = result.dataframes[0]
            info.update({
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "shape": df.shape,
                "preview": df.head(max_rows).to_dict('records'),
                "null_counts": df.isnull().sum().to_dict()
            })
        
        info["ingestion_status"] = result.status.value
        info["errors"] = result.errors
        
        return info
