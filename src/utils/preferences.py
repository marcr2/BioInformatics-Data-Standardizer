"""
Preferences Manager for BIDS

Handles saving and loading user preferences, including GPU acceleration settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch


class PreferencesManager:
    """Manages application preferences with persistence."""
    
    DEFAULT_PREFS = {
        "gpu_acceleration": True,  # Enable GPU by default if available
        "max_fix_attempts": 3,
        "auto_fix": True,
        "default_schema": "IPA Standard",
        "map_canvas_width": None,  # Auto-detect if None
        "map_canvas_height": None,  # Auto-detect if None
        "map_auto_fit": True,  # Auto-fit map to screen on startup
        "window_width": None,  # Auto-detect if None
        "window_height": None,  # Auto-detect if None
        "resolution_preset": "auto",  # "auto", "1920x1080", "1366x768", "2560x1440", "3840x2160", "custom"
    }
    
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
            return torch.cuda.is_available()
        except Exception:
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
            if torch.cuda.is_available():
                info["available"] = True
                info["device_name"] = torch.cuda.get_device_name(0)
                info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
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


# Global preferences instance
_preferences_instance: Optional[PreferencesManager] = None


def get_preferences() -> PreferencesManager:
    """Get the global preferences instance."""
    global _preferences_instance
    if _preferences_instance is None:
        _preferences_instance = PreferencesManager()
    return _preferences_instance

