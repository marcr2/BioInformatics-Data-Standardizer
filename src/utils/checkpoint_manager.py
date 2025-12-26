"""
Checkpoint Manager for BIDS

Handles saving and loading checkpoints at different stages of the processing cycle.
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd


class ProcessingStage:
    """Enumeration of processing stages."""
    IDLE = "idle"
    PREPROCESSING = "preprocessing"  # Rules-based preprocessing (before LLM)
    LOADING_LLM = "loading_llm"
    DIAGNOSING = "diagnosing"
    FIX_ATTEMPT = "fix_attempt"
    VALIDATING = "validating"
    COMPLETE = "complete"
    ERROR = "error"


class CheckpointManager:
    """Manages checkpoints during data processing."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            cache_dir: Directory for cache storage (default: cache/ in project root)
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.cache_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_checkpoint_id: Optional[str] = None
        self.checkpoint_history: List[Dict[str, Any]] = []
    
    def create_checkpoint(
        self,
        stage: str,
        df: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a checkpoint at a given stage.
        
        Args:
            stage: Processing stage name
            df: DataFrame to save (optional)
            metadata: Additional metadata to save
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        checkpoint_dir = self.checkpoints_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_info = {
            "id": checkpoint_id,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save DataFrame if provided
        if df is not None:
            df_path = checkpoint_dir / "dataframe.pkl"
            with open(df_path, 'wb') as f:
                pickle.dump(df, f)
            checkpoint_info["has_dataframe"] = True
            checkpoint_info["dataframe_shape"] = df.shape
        else:
            checkpoint_info["has_dataframe"] = False
        
        # Save checkpoint info
        info_path = checkpoint_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        self.current_checkpoint_id = checkpoint_id
        self.checkpoint_history.append(checkpoint_info)
        
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint by ID.
        
        Args:
            checkpoint_id: ID of checkpoint to load
            
        Returns:
            Dict with checkpoint data, or None if not found
        """
        checkpoint_dir = self.checkpoints_dir / checkpoint_id
        
        if not checkpoint_dir.exists():
            return None
        
        # Load info
        info_path = checkpoint_dir / "info.json"
        if not info_path.exists():
            return None
        
        with open(info_path, 'r') as f:
            checkpoint_info = json.load(f)
        
        # Load DataFrame if available
        df_path = checkpoint_dir / "dataframe.pkl"
        if df_path.exists() and checkpoint_info.get("has_dataframe"):
            with open(df_path, 'rb') as f:
                checkpoint_info["dataframe"] = pickle.load(f)
        
        return checkpoint_info
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint.
        
        Returns:
            Latest checkpoint data, or None if no checkpoints exist
        """
        if not self.checkpoint_history:
            return None
        
        latest = self.checkpoint_history[-1]
        return self.load_checkpoint(latest["id"])
    
    def get_checkpoint_by_stage(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint for a specific stage.
        
        Args:
            stage: Stage name to search for
            
        Returns:
            Latest checkpoint for that stage, or None if not found
        """
        # Search history in reverse order
        for checkpoint_info in reversed(self.checkpoint_history):
            if checkpoint_info["stage"] == stage:
                return self.load_checkpoint(checkpoint_info["id"])
        
        return None
    
    def clear_cache(self) -> None:
        """Clear all cached checkpoints."""
        import shutil
        
        if self.checkpoints_dir.exists():
            shutil.rmtree(self.checkpoints_dir)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_checkpoint_id = None
        self.checkpoint_history = []
    
    def get_cache_size(self) -> int:
        """
        Get total size of cache directory in bytes.
        
        Returns:
            Size in bytes
        """
        total_size = 0
        if self.cache_dir.exists():
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size
    
    def get_checkpoint_count(self) -> int:
        """Get number of checkpoints."""
        return len(self.checkpoint_history)

