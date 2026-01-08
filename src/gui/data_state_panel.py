"""
Data State Panel for BIDS GUI

Shows the state of data at the last checkpoint.
"""

import dearpygui.dearpygui as dpg
from typing import Optional, Dict, Any
from pathlib import Path
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
    
    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None, main_app=None):
        """
        Initialize data state panel.
        
        Args:
            checkpoint_manager: Checkpoint manager instance
            main_app: Reference to main app for accessing final dataset dialog
        """
        self.checkpoint_manager = checkpoint_manager
        self.main_app = main_app
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
            dpg.add_spacer(width=10)
            dpg.add_button(
                label="Export",
                callback=self._show_export_dialog,
                width=100
            )
            dpg.add_spacer(width=10)
            dpg.add_button(
                label="View Final Dataset",
                callback=self._view_final_dataset,
                width=130
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
                info_text += f"DataFrame Shape: {shape[0]} rows x {shape[1]} columns\n"
            
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
    
    def _show_export_dialog(self) -> None:
        """Show the export file dialog."""
        if not self.current_checkpoint:
            self._show_error("No checkpoint loaded")
            return
        
        df = self.current_checkpoint.get("dataframe")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            self._show_error("No data to export")
            return
        
        # Ensure exports directory exists
        exports_dir = Path("exports")
        exports_dir.mkdir(exist_ok=True)
        
        # Create save dialog
        export_dialog_tag = "data_state_export_dialog"
        if dpg.does_item_exist(export_dialog_tag):
            dpg.delete_item(export_dialog_tag)
        
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self._on_export_dialog_ok,
            cancel_callback=lambda s, a: dpg.delete_item(export_dialog_tag),
            width=700,
            height=400,
            modal=True,
            tag=export_dialog_tag,
            default_filename="data_state_export.csv",
            default_path=str(exports_dir.absolute())
        ):
            dpg.add_file_extension(".*", color=(255, 255, 255))
            dpg.add_file_extension(".csv", color=(0, 255, 0))
            dpg.add_file_extension(".tsv", color=(0, 255, 0))
            dpg.add_file_extension(".xlsx", color=(0, 200, 255))
            dpg.add_file_extension(".parquet", color=(255, 200, 0))
            dpg.add_file_extension(".json", color=(200, 200, 255))
    
    def _on_export_dialog_ok(self, sender, app_data) -> None:
        """Handle export dialog OK."""
        file_path = app_data.get("file_path_name", "")
        
        if not file_path:
            return
        
        self._export_to_file(file_path)
        dpg.delete_item("data_state_export_dialog")
    
    def _export_to_file(self, file_path: str) -> None:
        """Export DataFrame to file."""
        if not self.current_checkpoint:
            return
        
        df = self.current_checkpoint.get("dataframe")
        if df is None or not isinstance(df, pd.DataFrame):
            return
        
        try:
            # Ensure exports directory exists
            exports_dir = Path("exports")
            exports_dir.mkdir(exist_ok=True)
            
            path = Path(file_path)
            # If path is not in exports directory, move it there
            if "exports" not in str(path.parent).replace("\\", "/"):
                path = exports_dir / path.name
            
            suffix = path.suffix.lower()
            
            if suffix == ".csv":
                df.to_csv(path, index=False)
            elif suffix == ".tsv":
                df.to_csv(path, sep='\t', index=False)
            elif suffix == ".xlsx":
                df.to_excel(path, index=False, engine='openpyxl')
            elif suffix == ".parquet":
                df.to_parquet(path, index=False)
            elif suffix == ".json":
                df.to_json(path, orient='records', indent=2)
            else:
                # Default to CSV
                df.to_csv(path, index=False)
            
            self._show_success(f"Exported to {path.name}")
            
        except Exception as e:
            self._show_error(f"Export failed: {str(e)}")
    
    def _show_error(self, message: str) -> None:
        """Show error popup."""
        popup_tag = "data_state_error_popup"
        if dpg.does_item_exist(popup_tag):
            dpg.delete_item(popup_tag)
        
        with dpg.window(
            label="Error",
            modal=True,
            width=300,
            height=100,
            pos=[550, 400],
            on_close=lambda: dpg.delete_item(popup_tag)
        ) as popup:
            dpg.set_item_alias(popup, popup_tag)
            dpg.add_text(message, wrap=-1, color=(231, 76, 60))
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="OK",
                callback=lambda: dpg.delete_item(popup_tag),
                width=-1
            )
    
    def _view_final_dataset(self) -> None:
        """Open the final dataset dialog."""
        if not self.main_app:
            self._show_error("Main app reference not available")
            return
        
        # Check if final dataset exists
        if self.main_app.final_dataset is not None:
            self.main_app.show_final_dataset_dialog()
        elif (hasattr(self.main_app, 'process_panel') and 
              self.main_app.process_panel and 
              self.main_app.process_panel.final_dataset is not None):
            # Get from process panel
            self.main_app.final_dataset = self.main_app.process_panel.final_dataset.copy()
            self.main_app.show_final_dataset_dialog()
        else:
            self._show_error("No final dataset available. Please run processing first.")
    
    def _show_success(self, message: str) -> None:
        """Show success popup."""
        popup_tag = "data_state_success_popup"
        if dpg.does_item_exist(popup_tag):
            dpg.delete_item(popup_tag)
        
        with dpg.window(
            label="Success",
            modal=True,
            width=300,
            height=100,
            pos=[550, 400],
            on_close=lambda: dpg.delete_item(popup_tag)
        ) as popup:
            dpg.set_item_alias(popup, popup_tag)
            dpg.add_text(message, wrap=-1, color=(46, 204, 113))
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="OK",
                callback=lambda: dpg.delete_item(popup_tag),
                width=-1
            )

