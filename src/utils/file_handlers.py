"""
Archive Extraction Utilities for BIDS

Handles extraction of various archive formats including:
- ZIP, TAR, GZ, TAR.GZ
- RAR, 7Z
- Nested archives (recursive extraction)
"""

import os
import shutil
import tempfile
import tarfile
import zipfile
import gzip
from pathlib import Path
from typing import List, Optional, Tuple, Generator
from dataclasses import dataclass
from enum import Enum
import magic


class ArchiveType(Enum):
    """Supported archive types."""
    ZIP = "zip"
    TAR = "tar"
    GZIP = "gzip"
    TAR_GZ = "tar.gz"
    RAR = "rar"
    SEVEN_ZIP = "7z"
    UNKNOWN = "unknown"
    NOT_ARCHIVE = "not_archive"


@dataclass
class ExtractedFile:
    """Represents an extracted file."""
    path: Path
    original_name: str
    size: int
    mime_type: str
    is_tabular: bool


class ArchiveExtractor:
    """
    Universal archive extractor with recursive extraction support.
    
    Handles nested archives (e.g., data.tar.gz containing more .zip files)
    and identifies tabular data files.
    """
    
    # MIME types for tabular data
    TABULAR_MIMES = {
        'text/csv',
        'text/tab-separated-values',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/x-parquet',
        'text/plain',  # Could be CSV/TSV
    }
    
    # File extensions for tabular data
    TABULAR_EXTENSIONS = {'.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.txt'}
    
    def __init__(self, temp_dir: Optional[Path] = None, max_depth: int = 5):
        """
        Initialize the archive extractor.
        
        Args:
            temp_dir: Directory for temporary extraction (default: system temp)
            max_depth: Maximum recursion depth for nested archives
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "bids_extract"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_depth = max_depth
        self._extraction_id = 0
    
    def detect_archive_type(self, file_path: Path) -> ArchiveType:
        """
        Detect the type of archive using magic bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            ArchiveType enum value
        """
        try:
            mime = magic.from_file(str(file_path), mime=True)
            
            # Check MIME type
            if mime == 'application/zip':
                return ArchiveType.ZIP
            elif mime == 'application/x-tar':
                return ArchiveType.TAR
            elif mime == 'application/gzip' or mime == 'application/x-gzip':
                # Check if it's a tar.gz
                if str(file_path).endswith('.tar.gz') or str(file_path).endswith('.tgz'):
                    return ArchiveType.TAR_GZ
                return ArchiveType.GZIP
            elif mime == 'application/x-rar' or mime == 'application/x-rar-compressed':
                return ArchiveType.RAR
            elif mime == 'application/x-7z-compressed':
                return ArchiveType.SEVEN_ZIP
            
            # Fallback to extension check
            suffix = file_path.suffix.lower()
            if suffix == '.zip':
                return ArchiveType.ZIP
            elif suffix == '.tar':
                return ArchiveType.TAR
            elif suffix == '.gz':
                if '.tar' in file_path.stem.lower():
                    return ArchiveType.TAR_GZ
                return ArchiveType.GZIP
            elif suffix == '.tgz':
                return ArchiveType.TAR_GZ
            elif suffix == '.rar':
                return ArchiveType.RAR
            elif suffix == '.7z':
                return ArchiveType.SEVEN_ZIP
            
            return ArchiveType.NOT_ARCHIVE
            
        except Exception:
            return ArchiveType.UNKNOWN
    
    def extract(self, file_path: Path, depth: int = 0) -> List[ExtractedFile]:
        """
        Extract an archive recursively.
        
        Args:
            file_path: Path to the archive
            depth: Current recursion depth
            
        Returns:
            List of ExtractedFile objects
        """
        if depth >= self.max_depth:
            return []
        
        self._extraction_id += 1
        extract_dir = self.temp_dir / f"extract_{self._extraction_id}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        archive_type = self.detect_archive_type(file_path)
        extracted_files: List[ExtractedFile] = []
        
        try:
            # Extract based on type
            if archive_type == ArchiveType.ZIP:
                self._extract_zip(file_path, extract_dir)
            elif archive_type == ArchiveType.TAR:
                self._extract_tar(file_path, extract_dir)
            elif archive_type == ArchiveType.GZIP:
                self._extract_gzip(file_path, extract_dir)
            elif archive_type == ArchiveType.TAR_GZ:
                self._extract_tar_gz(file_path, extract_dir)
            elif archive_type == ArchiveType.RAR:
                self._extract_rar(file_path, extract_dir)
            elif archive_type == ArchiveType.SEVEN_ZIP:
                self._extract_7z(file_path, extract_dir)
            else:
                # Not an archive, check if it's tabular
                return self._handle_non_archive(file_path)
            
            # Process extracted files
            for extracted_path in self._walk_extracted(extract_dir):
                # Check if it's another archive (recursive)
                nested_type = self.detect_archive_type(extracted_path)
                if nested_type not in (ArchiveType.NOT_ARCHIVE, ArchiveType.UNKNOWN):
                    nested_files = self.extract(extracted_path, depth + 1)
                    extracted_files.extend(nested_files)
                else:
                    # Check if tabular
                    file_info = self._create_file_info(extracted_path)
                    if file_info:
                        extracted_files.append(file_info)
        
        except Exception as e:
            print(f"Extraction error for {file_path}: {e}")
        
        return extracted_files
    
    def _extract_zip(self, file_path: Path, extract_dir: Path) -> None:
        """Extract ZIP archive."""
        with zipfile.ZipFile(file_path, 'r') as zf:
            zf.extractall(extract_dir)
    
    def _extract_tar(self, file_path: Path, extract_dir: Path) -> None:
        """Extract TAR archive."""
        with tarfile.open(file_path, 'r:') as tf:
            tf.extractall(extract_dir, filter='data')
    
    def _extract_gzip(self, file_path: Path, extract_dir: Path) -> None:
        """Extract GZIP file."""
        output_name = file_path.stem  # Remove .gz extension
        output_path = extract_dir / output_name
        
        with gzip.open(file_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def _extract_tar_gz(self, file_path: Path, extract_dir: Path) -> None:
        """Extract TAR.GZ archive."""
        with tarfile.open(file_path, 'r:gz') as tf:
            tf.extractall(extract_dir, filter='data')
    
    def _extract_rar(self, file_path: Path, extract_dir: Path) -> None:
        """Extract RAR archive."""
        import rarfile
        with rarfile.RarFile(file_path, 'r') as rf:
            rf.extractall(extract_dir)
    
    def _extract_7z(self, file_path: Path, extract_dir: Path) -> None:
        """Extract 7Z archive."""
        import py7zr
        with py7zr.SevenZipFile(file_path, mode='r') as szf:
            szf.extractall(path=extract_dir)
    
    def _walk_extracted(self, directory: Path) -> Generator[Path, None, None]:
        """Walk through extracted directory and yield file paths."""
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files:
                if not filename.startswith('.'):
                    yield Path(root) / filename
    
    def _handle_non_archive(self, file_path: Path) -> List[ExtractedFile]:
        """Handle a non-archive file."""
        file_info = self._create_file_info(file_path)
        return [file_info] if file_info else []
    
    def _create_file_info(self, file_path: Path) -> Optional[ExtractedFile]:
        """Create ExtractedFile info for a file."""
        try:
            mime = magic.from_file(str(file_path), mime=True)
            is_tabular = self._is_tabular(file_path, mime)
            
            return ExtractedFile(
                path=file_path,
                original_name=file_path.name,
                size=file_path.stat().st_size,
                mime_type=mime,
                is_tabular=is_tabular
            )
        except Exception:
            return None
    
    def _is_tabular(self, file_path: Path, mime_type: str) -> bool:
        """Check if file is likely tabular data."""
        # Check MIME type
        if mime_type in self.TABULAR_MIMES:
            return True
        
        # Check extension
        if file_path.suffix.lower() in self.TABULAR_EXTENSIONS:
            return True
        
        # For text/plain, peek at content
        if mime_type == 'text/plain':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_lines = [f.readline() for _ in range(3)]
                    content = ''.join(first_lines)
                    # Check for CSV/TSV patterns
                    if ',' in content or '\t' in content:
                        return True
            except Exception:
                pass
        
        return False
    
    def cleanup(self) -> None:
        """Clean up temporary extraction directory."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def get_tabular_files(self, file_path: Path) -> List[ExtractedFile]:
        """
        Extract and return only tabular data files.
        
        Args:
            file_path: Path to file or archive
            
        Returns:
            List of ExtractedFile objects that are tabular
        """
        all_files = self.extract(file_path)
        return [f for f in all_files if f.is_tabular]


def read_file_header(file_path: Path, num_bytes: int = 1024) -> bytes:
    """
    Read the first N bytes of a file.
    
    Args:
        file_path: Path to the file
        num_bytes: Number of bytes to read
        
    Returns:
        Bytes from file header
    """
    with open(file_path, 'rb') as f:
        return f.read(num_bytes)


def get_file_info(file_path: Path) -> dict:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict with file information
    """
    try:
        mime = magic.from_file(str(file_path), mime=True)
        stat = file_path.stat()
        
        return {
            "path": str(file_path),
            "name": file_path.name,
            "extension": file_path.suffix,
            "mime_type": mime,
            "size": stat.st_size,
            "size_human": _human_readable_size(stat.st_size)
        }
    except Exception as e:
        return {
            "path": str(file_path),
            "name": file_path.name,
            "error": str(e)
        }


def _human_readable_size(size: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
