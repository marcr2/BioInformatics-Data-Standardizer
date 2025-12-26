"""
BIDS - Bioinformatics Data Standardizer

A system for cleaning and standardizing messy clinical data
into IPA format or custom CSV schemas.
"""

__version__ = "1.0.0"
__author__ = "BIDS Team"

# Core components (lazy imports to avoid dependency issues)
def get_ingestor(*args, **kwargs):
    """Get a SmartIngestor instance."""
    from .ingestion import SmartIngestor
    return SmartIngestor(*args, **kwargs)

def get_schema_manager(*args, **kwargs):
    """Get a SchemaManager instance."""
    from .schema_manager import get_schema_manager as _get
    return _get(*args, **kwargs)

def get_orchestrator(*args, **kwargs):
    """Get an AgentOrchestrator instance."""
    from .agents import AgentOrchestrator
    return AgentOrchestrator(*args, **kwargs)

__all__ = [
    "__version__",
    "__author__",
    "get_ingestor",
    "get_schema_manager", 
    "get_orchestrator"
]
