"""
Final Dataset Dialog for BIDS GUI

Displays the final processed dataset in a dialog window.
"""

import dearpygui.dearpygui as dpg
from typing import Optional
from pathlib import Path
import pandas as pd


class FinalDatasetDialog:
    """
    Dialog window showing the final processed dataset.
    
    Features:
    - Scrollable table display
    - Export functionality
    - Can be reopened from data state tab
    """
    
    MAX_PREVIEW_ROWS = 1000  # Show more rows in dialog
    
    def __init__(self):
        """Initialize FinalDatasetDialog."""
        self.dialog_tag: Optional[str] = None
        self.current_df: Optional[pd.DataFrame] = None
        self.table_container_tag: Optional[int] = None
    
    def show(self, df: pd.DataFrame) -> None:
        """
        Show the dialog with the given DataFrame.
        
        Args:
            df: DataFrame to display
        """
        self.current_df = df
        
        # Close existing dialog if open
        if self.dialog_tag and dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)
        
        # Create dialog window
        self.dialog_tag = "final_dataset_dialog"
        
        with dpg.window(
            label="Final Dataset",
            tag=self.dialog_tag,
            width=1000,
            height=700,
            pos=[200, 100],
            on_close=self._on_close
        ):
            # Header with stats
            with dpg.group(horizontal=True):
                dpg.add_text("Final Processed Dataset", color=(100, 149, 237))
                dpg.add_spacer(width=20)
                stats_text = f"Rows: {len(df):,} | Columns: {len(df.columns)}"
                null_count = df.isnull().sum().sum()
                if null_count > 0:
                    stats_text += f" | Nulls: {null_count:,}"
                dpg.add_text(stats_text, color=(149, 165, 166))
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Export button
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Export",
                    callback=self._show_export_dialog,
                    width=100
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Close",
                    callback=self._on_close,
                    width=100
                )
            
            dpg.add_spacer(height=10)
            
            # Table container with scrollbars
            # Calculate available height: window (700) - header (~80) - buttons (~50) - spacing (~40) = ~530
            with dpg.child_window(
                height=530,
                border=True,
                horizontal_scrollbar=True
            ) as container:
                self.table_container_tag = container
                self._build_table(df)
    
    def _build_table(self, df: pd.DataFrame) -> None:
        """Build the data table display."""
        # Clear any existing content
        try:
            dpg.delete_item(self.table_container_tag, children_only=True)
        except Exception:
            pass
        
        if df is None or len(df) == 0:
            dpg.add_text(
                "No data to display",
                parent=self.table_container_tag,
                color=(149, 165, 166)
            )
            return
        
        # Limit rows for display
        display_df = df.head(self.MAX_PREVIEW_ROWS)
        columns = list(df.columns)
        
        # Create table with proper settings
        with dpg.table(
            parent=self.table_container_tag,
            header_row=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            scrollX=True,
            scrollY=True,
            freeze_rows=1,
            resizable=True,
            policy=dpg.mvTable_SizingFixedFit
        ):
            # Add columns
            for col in columns:
                dpg.add_table_column(
                    label=str(col),
                    width_fixed=False
                )
            
            # Add rows
            for idx, row in display_df.iterrows():
                with dpg.table_row():
                    for col in columns:
                        value = row[col]
                        
                        # Format value
                        if pd.isna(value):
                            dpg.add_text("NULL", color=(231, 76, 60))
                        elif isinstance(value, float):
                            dpg.add_text(f"{value:.4g}")
                        else:
                            text = str(value)
                            if len(text) > 50:
                                text = text[:47] + "..."
                            dpg.add_text(text)
            
            # Show if truncated
            if len(df) > self.MAX_PREVIEW_ROWS:
                with dpg.table_row():
                    dpg.add_text(
                        f"... {len(df) - self.MAX_PREVIEW_ROWS:,} more rows",
                        color=(149, 165, 166)
                    )
                    # Add empty cells for remaining columns
                    for _ in range(len(columns) - 1):
                        dpg.add_text("")
    
    def _show_export_dialog(self) -> None:
        """Show the export file dialog."""
        if self.current_df is None or self.current_df.empty:
            self._show_error("No data to export")
            return
        
        # Ensure exports directory exists
        exports_dir = Path("exports")
        exports_dir.mkdir(exist_ok=True)
        
        # Create save dialog
        export_dialog_tag = "final_dataset_export_dialog"
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
            default_filename="final_dataset.csv",
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
        dpg.delete_item("final_dataset_export_dialog")
    
    def _export_to_file(self, file_path: str) -> None:
        """Export DataFrame to file."""
        if self.current_df is None:
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
                self.current_df.to_csv(path, index=False)
            elif suffix == ".tsv":
                self.current_df.to_csv(path, sep='\t', index=False)
            elif suffix == ".xlsx":
                self.current_df.to_excel(path, index=False, engine='openpyxl')
            elif suffix == ".parquet":
                self.current_df.to_parquet(path, index=False)
            elif suffix == ".json":
                self.current_df.to_json(path, orient='records', indent=2)
            else:
                # Default to CSV
                self.current_df.to_csv(path, index=False)
            
            self._show_success(f"Exported to {path.name}")
            
        except Exception as e:
            self._show_error(f"Export failed: {str(e)}")
    
    def _on_close(self) -> None:
        """Handle dialog close."""
        if self.dialog_tag and dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)
        self.dialog_tag = None
    
    def _show_error(self, message: str) -> None:
        """Show error popup."""
        popup_tag = "final_dataset_error_popup"
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
    
    def _show_success(self, message: str) -> None:
        """Show success popup."""
        popup_tag = "final_dataset_success_popup"
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
    
    def is_open(self) -> bool:
        """Check if dialog is currently open."""
        return self.dialog_tag is not None and dpg.does_item_exist(self.dialog_tag)
