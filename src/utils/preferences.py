"""
Preferences Manager for BIDS

Handles saving and loading user preferences, including GPU acceleration settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class PreferencesManager:
    """Manages application preferences with persistence."""
    
    DEFAULT_PREFS = {
        "gpu_acceleration": True,  # Enable GPU by default if available
        "max_fix_attempts": 3,
        "auto_fix": True,
        "default_schema": "IPA Standard",
        "llm_rewrite_enabled": False,  # Optional LLM-rewrite step (structural reorganization only)
        "processing_mode": "auto",  # "auto" (use LLM if available), "rules_only" (never use LLM), "llm_required" (require LLM)
        "llm_installed": False,  # Track if LLM dependencies are installed (updated on check)
        "map_canvas_width": None,  # Auto-detect if None
        "map_canvas_height": None,  # Auto-detect if None
        "map_auto_fit": True,  # Auto-fit map to screen on startup
        "window_width": None,  # Auto-detect if None
        "window_height": None,  # Auto-detect if None
        "resolution_preset": "auto",  # "auto", "1920x1080", "1366x768", "2560x1440", "3840x2160", "custom"
        "max_tokens_percentage": 50,  # Max tokens as percentage of GPU memory (default 50%)
    }
    
    # Estimated tokens per GB of GPU memory (conservative estimate for 4-bit quantized models)
    TOKENS_PER_GB = 500
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize preferences manager.
        
        Args:
            config_file: Path to preferences file (default: .bids_prefs.json in project root)
        """
        if config_file is None:
            # Default to project root
            config_file = Path(__file__).parent.parent.parent / ".bids_prefs.json"
        
        self.config_file = Path(config_file)
        self._prefs: Dict[str, Any] = {}
        # Store loaded model memory info (not persisted, runtime only)
        self._loaded_model_id: Optional[str] = None
        self._loaded_model_memory_gb: Optional[float] = None
        self.load()
    
    def load(self) -> None:
        """Load preferences from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self._prefs = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key, value in self.DEFAULT_PREFS.items():
                    if key not in self._prefs:
                        self._prefs[key] = value
            except Exception:
                # If loading fails, use defaults
                self._prefs = self.DEFAULT_PREFS.copy()
        else:
            # First run - use defaults but check GPU availability
            self._prefs = self.DEFAULT_PREFS.copy()
            # Auto-detect GPU and set preference
            self._prefs["gpu_acceleration"] = self._check_gpu_available()
            # Check LLM availability
            try:
                from .llm_check import is_llm_available
                self._prefs["llm_installed"] = is_llm_available()
            except Exception:
                self._prefs["llm_installed"] = False
            self.save()
    
    def save(self) -> None:
        """Save preferences to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._prefs, f, indent=2)
        except Exception:
            pass  # Silently fail if can't save
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        return self._prefs.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a preference value and save."""
        self._prefs[key] = value
        self.save()
    
    def get_all(self) -> Dict[str, Any]:
        """Get all preferences."""
        return self._prefs.copy()
    
    def update(self, prefs: Dict[str, Any]) -> None:
        """Update multiple preferences at once."""
        self._prefs.update(prefs)
        self.save()
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except (ImportError, Exception):
            return False
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        gpu_pref = self.get("gpu_acceleration", True)
        # If preference says GPU is enabled, verify it's actually available
        if gpu_pref:
            return self._check_gpu_available()
        return False
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        info = {
            "available": False,
            "device_name": None,
            "memory_gb": None,
            "preference_enabled": self.get("gpu_acceleration", True)
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                info["available"] = True
                info["device_name"] = torch.cuda.get_device_name(0)
                info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except (ImportError, Exception):
            pass
        
        return info
    
    def get_screen_resolution(self) -> Dict[str, int]:
        """Get the current screen resolution."""
        try:
            import tkinter as tk
            root = tk.Tk()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return {"width": width, "height": height}
        except Exception:
            # Fallback to common resolution
            return {"width": 1920, "height": 1080}
    
    def get_resolution_presets(self) -> Dict[str, Dict[str, int]]:
        """Get available resolution presets."""
        screen = self.get_screen_resolution()
        screen_width = screen["width"]
        screen_height = screen["height"]
        
        # Calculate presets based on screen size
        presets = {
            "auto": {
                "width": int(screen_width * 0.85),  # 85% of screen width
                "height": int(screen_height * 0.85),  # 85% of screen height
                "label": f"Auto ({int(screen_width * 0.85)}x{int(screen_height * 0.85)})"
            },
            "1920x1080": {"width": 1920, "height": 1080, "label": "1920x1080 (Full HD)"},
            "1366x768": {"width": 1366, "height": 768, "label": "1366x768 (HD)"},
            "2560x1440": {"width": 2560, "height": 1440, "label": "2560x1440 (QHD)"},
            "3840x2160": {"width": 3840, "height": 2160, "label": "3840x2160 (4K)"},
            "1280x720": {"width": 1280, "height": 720, "label": "1280x720 (HD)"},
            "1600x900": {"width": 1600, "height": 900, "label": "1600x900"},
            "custom": {"width": None, "height": None, "label": "Custom"}
        }
        
        return presets
    
    def get_window_size(self) -> Tuple[int, int]:
        """Get the window size based on preferences."""
        preset = self.get("resolution_preset", "auto")
        
        # First, check if we have saved width/height values (for both custom and presets)
        saved_width = self.get("window_width")
        saved_height = self.get("window_height")
        
        if saved_width and saved_height:
            # Use saved values if available (this works for both custom and preset selections)
            return (saved_width, saved_height)
        
        # If no saved values, calculate from preset
        presets = self.get_resolution_presets()
        
        if preset == "custom":
            # Custom preset but no saved values - fallback to auto
            preset = "auto"
        
        if preset in presets:
            preset_data = presets[preset]
            return (preset_data["width"], preset_data["height"])
        
        # Default fallback
        return (1400, 900)
    
    def register_model_memory(self, model_id: str, memory_gb: float) -> None:
        """
        Register the GPU memory usage of a loaded model.
        
        Args:
            model_id: The model identifier
            memory_gb: Memory usage in GB
        """
        self._loaded_model_id = model_id
        self._loaded_model_memory_gb = memory_gb
    
    def get_model_memory_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model's memory usage.
        
        Returns:
            Dict with model_id, memory_gb, and base_percentage
        """
        gpu_info = self.get_gpu_info()
        if not gpu_info["available"] or not gpu_info["memory_gb"]:
            return {
                "model_id": None,
                "memory_gb": None,
                "base_percentage": None
            }
        
        if self._loaded_model_memory_gb is not None:
            base_percentage = (self._loaded_model_memory_gb / gpu_info["memory_gb"]) * 100
            return {
                "model_id": self._loaded_model_id,
                "memory_gb": self._loaded_model_memory_gb,
                "base_percentage": base_percentage
            }
        
        return {
            "model_id": None,
            "memory_gb": None,
            "base_percentage": None
        }
    
    def get_max_tokens(self) -> int:
        """
        Calculate max tokens based on GPU memory and percentage setting.
        Uses model memory as base, then calculates from remaining memory.
        
        Returns:
            Maximum tokens to use for generation
        """
        gpu_info = self.get_gpu_info()
        percentage = self.get("max_tokens_percentage", 50)
        
        if gpu_info["available"] and gpu_info["memory_gb"]:
            total_memory_gb = gpu_info["memory_gb"]
            
            # If model is loaded, use remaining memory after model
            if self._loaded_model_memory_gb is not None:
                available_memory_gb = total_memory_gb - self._loaded_model_memory_gb
                # Ensure we don't go negative
                available_memory_gb = max(0.1, available_memory_gb)
                # Calculate tokens from available memory (after model)
                max_tokens = int(available_memory_gb * (percentage / 100) * self.TOKENS_PER_GB)
            else:
                # Fallback: use total memory if model not loaded yet
                max_tokens = int(total_memory_gb * (percentage / 100) * self.TOKENS_PER_GB)
            
            # Clamp to reasonable bounds
            return max(1024, min(max_tokens, 16384))
        else:
            # Fallback for CPU
            return 4096
    
    def set_max_tokens_percentage(self, percentage: float) -> None:
        """
        Set the max tokens percentage.
        
        Args:
            percentage: Percentage of GPU memory to use (0-100)
        """
        # Clamp to valid range
        percentage = max(10, min(100, percentage))
        self.set("max_tokens_percentage", percentage)
    
    def get_max_tokens_info(self) -> Dict[str, Any]:
        """
        Get detailed information about max tokens settings.
        
        Returns:
            Dict with percentage, calculated tokens, GPU memory info, and model memory info
        """
        gpu_info = self.get_gpu_info()
        percentage = self.get("max_tokens_percentage", 50)
        max_tokens = self.get_max_tokens()
        model_info = self.get_model_memory_info()
        
        result = {
            "percentage": percentage,
            "max_tokens": max_tokens,
            "gpu_available": gpu_info["available"],
            "gpu_memory_gb": gpu_info.get("memory_gb"),
            "tokens_per_gb": self.TOKENS_PER_GB,
            "model_id": model_info.get("model_id"),
            "model_memory_gb": model_info.get("memory_gb"),
            "base_percentage": model_info.get("base_percentage")
        }
        
        # Calculate available memory for tokens
        if gpu_info.get("memory_gb") and model_info.get("memory_gb"):
            result["available_memory_gb"] = gpu_info["memory_gb"] - model_info["memory_gb"]
        else:
            result["available_memory_gb"] = None
        
        return result
    
    def is_llm_available(self) -> bool:
        """
        Check if LLM is available (installed and can be used).
        
        Returns:
            True if LLM is available, False otherwise
        """
        try:
            from .llm_check import is_llm_available
            available = is_llm_available()
            # Update the preference to reflect current status
            self.set("llm_installed", available)
            return available
        except Exception:
            self.set("llm_installed", False)
            return False
    
    def get_processing_mode(self) -> str:
        """
        Get the current processing mode.
        
        Returns:
            Processing mode: "auto", "rules_only", or "llm_required"
        """
        return self.get("processing_mode", "auto")
    
    def set_processing_mode(self, mode: str) -> None:
        """
        Set the processing mode.
        
        Args:
            mode: Processing mode ("auto", "rules_only", or "llm_required")
        """
        if mode not in ["auto", "rules_only", "llm_required"]:
            raise ValueError(f"Invalid processing mode: {mode}")
        self.set("processing_mode", mode)
    
    def should_use_llm(self) -> bool:
        """
        Determine if LLM should be used based on processing mode and availability.
        
        Returns:
            True if LLM should be used, False otherwise
        """
        mode = self.get_processing_mode()
        llm_available = self.is_llm_available()
        
        if mode == "rules_only":
            return False
        elif mode == "llm_required":
            return llm_available  # Will raise error in orchestrator if False
        else:  # "auto"
            return llm_available


# Global preferences instance
_preferences_instance: Optional[PreferencesManager] = None


def get_preferences() -> PreferencesManager:
    """Get the global preferences instance."""
    global _preferences_instance
    if _preferences_instance is None:
        _preferences_instance = PreferencesManager()
    return _preferences_instance

