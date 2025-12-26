"""
BIDS Utilities Module

Contains helper functions and clients for:
- LLM interactions (Gemini, Claude)
- File handling and archive extraction
"""

# Lazy imports to avoid circular dependencies
def get_gemini_client(*args, **kwargs):
    from .llm_client import get_gemini_client as _get
    return _get(*args, **kwargs)

def get_claude_client(*args, **kwargs):
    from .llm_client import get_claude_client as _get
    return _get(*args, **kwargs)

def get_archive_extractor(*args, **kwargs):
    from .file_handlers import ArchiveExtractor
    return ArchiveExtractor(*args, **kwargs)

__all__ = [
    "get_gemini_client",
    "get_claude_client",
    "get_archive_extractor"
]
