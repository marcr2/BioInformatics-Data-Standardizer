"""
Data State Panel for BIDS GUI

Shows the state of data at the last checkpoint.
"""

import dearpygui.dearpygui as dpg
from typing import Optional, Dict, Any
import pandas as pd
from ..utils.checkpoint_manager import CheckpointManager


class DataStatePanel:
    """
    Panel showing data state at last checkpoint.
    
    Features:
    - Display DataFrame from last checkpoint
    - Show checkpoint metadata
    - Allow navigation between checkpoints
    """
    
    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        """
        Initialize data state panel.
        
        Args:
            checkpoint_manager: Checkpoint manager instance
        """
        self.checkpoint_manager = checkpoint_manager
        self.current_checkpoint: Optional[Dict[str, Any]] = None
        
        # UI tags
        self.checkpoint_info_tag: Optional[int] = None
        self.data_table_tag: Optional[int] = None
        self.data_container_tag: Optional[int] = None
    
    def create(self) -> None:
        """Create the data state panel UI."""
        dpg.add_text("Data State at Last Checkpoint", color=(100, 149, 237))
        dpg.add_spacer(height=10)
        
        # Checkpoint selection and info
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Refresh",
                callback=self._refresh_checkpoint,
                width=100
            )
            dpg.add_spacer(width=10)
            dpg.add_button(
                label="Load Latest",
                callback=self._load_latest,
                width=100
            )
        
        dpg.add_spacer(height=10)
        
        # Checkpoint metadata
        with dpg.group():
            self.checkpoint_info_tag = dpg.add_text(
                "No checkpoint loaded",
                color=(149, 165, 166),
                wrap=-1
            )
        
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)
        
        # Data display
        with dpg.child_window(height=-1, border=True) as container:
            self.data_container_tag = container
            dpg.add_text(
                "Load a checkpoint to view data",
                color=(149, 165, 166),
                parent=container
            )
        
        # Load latest checkpoint on creation
        self._load_latest()
    
    def _refresh_checkpoint(self) -> None:
        """Refresh the current checkpoint display."""
        if self.current_checkpoint and self.checkpoint_manager:
            checkpoint_id = self.current_checkpoint.get("id")
            if checkpoint_id:
                self.current_checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_id)
                self._update_display()
    
    def _load_latest(self) -> None:
        """Load the latest checkpoint."""
        if not self.checkpoint_manager:
            if self.checkpoint_info_tag:
                dpg.set_value(
                    self.checkpoint_info_tag,
                    "Checkpoint manager not available"
                )
            return
        
        latest = self.checkpoint_manager.get_latest_checkpoint()
        if latest:
            self.current_checkpoint = latest
            self._update_display()
        else:
            self.current_checkpoint = None
            if self.checkpoint_info_tag:
                dpg.set_value(
                    self.checkpoint_info_tag,
                    "No checkpoints available"
                )
            if self.data_container_tag:
                dpg.delete_item(self.data_container_tag, children_only=True)
                dpg.add_text(
                    "No checkpoint data to display",
                    color=(149, 165, 166),
                    parent=self.data_container_tag
                )
    
    def _update_display(self) -> None:
        """Update the display with current checkpoint data."""
        if not self.current_checkpoint:
            return
        
        # Update checkpoint info
        if self.checkpoint_info_tag:
            info = self.current_checkpoint
            info_text = (
                f"Checkpoint ID: {info.get('id', 'Unknown')}\n"
                f"Stage: {info.get('stage', 'Unknown')}\n"
                f"Timestamp: {info.get('timestamp', 'Unknown')}\n"
            )
            
            if info.get("has_dataframe"):
                shape = info.get("dataframe_shape", (0, 0))
                info_text += f"DataFrame Shape: {shape[0]} rows × {shape[1]} columns\n"
            
            metadata = info.get("metadata", {})
            if metadata:
                info_text += "\nMetadata:\n"
                for key, value in metadata.items():
                    info_text += f"  {key}: {value}\n"
            
            dpg.set_value(self.checkpoint_info_tag, info_text)
        
        # Update data display
        if self.data_container_tag:
            dpg.delete_item(self.data_container_tag, children_only=True)
            
            df = self.current_checkpoint.get("dataframe")
            if df is not None and isinstance(df, pd.DataFrame):
                self._display_dataframe(df)
            else:
                dpg.add_text(
                    "No DataFrame in this checkpoint",
                    color=(149, 165, 166),
                    parent=self.data_container_tag
                )
    
    def _display_dataframe(self, df: pd.DataFrame) -> None:
        """Display a DataFrame in the panel."""
        if df.empty:
            dpg.add_text(
                "DataFrame is empty",
                color=(149, 165, 166),
                parent=self.data_container_tag
            )
            return
        
        # Show summary info
        with dpg.group(parent=self.data_container_tag):
            dpg.add_text(
                f"Showing {len(df)} rows, {len(df.columns)} columns",
                color=(149, 165, 166)
            )
            dpg.add_spacer(height=5)
            
            # Show column names
            dpg.add_text("Columns:", color=(100, 149, 237))
            columns_text = ", ".join(df.columns.tolist()[:20])
            if len(df.columns) > 20:
                columns_text += f" ... (+{len(df.columns) - 20} more)"
            dpg.add_text(columns_text, wrap=-1, color=(236, 240, 241))
            
            dpg.add_spacer(height=10)
            
            # Show preview (first 100 rows)
            preview_rows = min(100, len(df))
            dpg.add_text(
                f"Preview (first {preview_rows} rows):",
                color=(100, 149, 237)
            )
            dpg.add_spacer(height=5)
            
            # Create table
            with dpg.table(
                header_row=True,
                resizable=True,
                policy=dpg.mvTable_SizingStretchProp,
                borders_innerH=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_outerV=True,
                parent=self.data_container_tag
            ):
                # Add columns
                for col in df.columns:
                    dpg.add_table_column(label=str(col)[:30], width_fixed=False)
                
                # Add rows (limit to preview)
                for idx in range(preview_rows):
                    with dpg.table_row():
                        for col in df.columns:
                            value = df.iloc[idx, df.columns.get_loc(col)]
                            # Convert to string, handle NaN
                            if pd.isna(value):
                                display_value = "NaN"
                            else:
                                display_value = str(value)[:50]  # Truncate long values
                            dpg.add_text(display_value)
            
            if len(df) > preview_rows:
                dpg.add_text(
                    f"... and {len(df) - preview_rows} more rows",
                    color=(149, 165, 166)
                )

