"""
LLM Availability Check Utility

Provides functions to check if LLM libraries are installed and available.
"""

from typing import Optional


def is_llm_available() -> bool:
    """
    Check if transformers library is installed and can be imported.
    
    Returns:
        True if transformers can be imported, False otherwise
    """
    try:
        import transformers
        return True
    except ImportError:
        return False


def can_load_model() -> bool:
    """
    Check if a model can actually be loaded (more expensive check).
    This attempts to import all required LLM dependencies.
    
    Returns:
        True if all LLM dependencies can be imported, False otherwise
    """
    try:
        import transformers
        import accelerate
        import torch
        # Try to import BitsAndBytesConfig (may fail if bitsandbytes not available)
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            # bitsandbytes is optional (only needed for quantization)
            pass
        return True
    except ImportError:
        return False


def get_llm_status() -> dict:
    """
    Get detailed status of LLM installation.
    
    Returns:
        Dictionary with status information:
        - available: bool - Whether LLM is available
        - transformers: bool - Whether transformers is installed
        - torch: bool - Whether torch is installed
        - accelerate: bool - Whether accelerate is installed
        - bitsandbytes: bool - Whether bitsandbytes is installed (optional)
        - message: str - Human-readable status message
    """
    status = {
        "available": False,
        "transformers": False,
        "torch": False,
        "accelerate": False,
        "bitsandbytes": False,
        "message": "LLM not installed"
    }
    
    try:
        import transformers
        status["transformers"] = True
    except ImportError:
        status["message"] = "transformers library not installed"
        return status
    
    try:
        import torch
        status["torch"] = True
    except ImportError:
        status["message"] = "PyTorch not installed"
        return status
    
    try:
        import accelerate
        status["accelerate"] = True
    except ImportError:
        status["message"] = "accelerate library not installed"
        return status
    
    try:
        import bitsandbytes
        status["bitsandbytes"] = True
    except ImportError:
        # bitsandbytes is optional, so we don't fail if it's missing
        pass
    
    # All required components are available
    status["available"] = True
    status["message"] = "LLM support available"
    
    return status

