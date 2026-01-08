"""
Results Panel for BIDS GUI

Displays data preview and export functionality.
"""

import dearpygui.dearpygui as dpg
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd


class ResultsPanel:
    """
    Data preview and export panel.
    
    Features:
    - DataFrame table display
    - Column statistics
    - Export to various formats
    - Compare original vs fixed
    """
    
    MAX_PREVIEW_ROWS = 100
    MAX_PREVIEW_COLS = 1000  # Allow all columns, horizontal scroll will handle it
    
    def __init__(self):
        """Initialize ResultsPanel."""
        self.current_df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        
        # UI tags
        self.table_container_tag: Optional[int] = None
        self.stats_text_tag: Optional[int] = None
        self.export_format_tag: Optional[int] = None
    
    def create(self) -> None:
        """Create the results panel UI."""
        # Stats bar
        with dpg.group(horizontal=True):
            dpg.add_text("Data Preview", color=(100, 149, 237))
            dpg.add_spacer(width=20)
            self.stats_text_tag = dpg.add_text(
                "No data loaded",
                color=(149, 165, 166)
            )
        
        dpg.add_spacer(height=5)
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        # Table container with horizontal scroll support
        with dpg.child_window(
            height=-60,
            border=True,
            horizontal_scrollbar=True
        ) as container:
            self.table_container_tag = container
            dpg.add_text(
                "Load a file to see data preview",
                color=(149, 165, 166)
            )
        
        dpg.add_spacer(height=10)
        
        # Export section
        with dpg.group(horizontal=True):
            dpg.add_text("Export as:", color=(149, 165, 166))
            
            self.export_format_tag = dpg.add_combo(
                items=["CSV", "TSV", "Excel (.xlsx)", "Parquet", "JSON"],
                default_value="CSV",
                width=120
            )
            
            dpg.add_button(
                label="Export",
                callback=self.show_export_dialog,
                width=80
            )
            
            dpg.add_spacer(width=20)
            
            dpg.add_button(
                label="Copy to Clipboard",
                callback=self._copy_to_clipboard,
                width=120
            )
    
    def show_dataframe(self, df: pd.DataFrame, is_original: bool = True) -> None:
        """
        Display a DataFrame in the preview.
        
        Args:
            df: DataFrame to display
            is_original: Whether this is the original (True) or fixed (False) data
        """
        if is_original:
            self.original_df = df
        self.current_df = df
        
        # Update stats
        stats = f"Rows: {len(df):,} | Columns: {len(df.columns)} | "
        null_count = df.isnull().sum().sum()
        stats += f"Nulls: {null_count:,}"
        
        if not is_original and self.original_df is not None:
            orig_nulls = self.original_df.isnull().sum().sum()
            if null_count < orig_nulls:
                stats += f" (reduced from {orig_nulls:,})"
        
        dpg.set_value(self.stats_text_tag, stats)
        
        # Rebuild table
        self._build_table(df)
    
    def _build_table(self, df: pd.DataFrame) -> None:
        """Build the data table display."""
        # Clear container
        dpg.delete_item(self.table_container_tag, children_only=True)
        
        if df is None or len(df) == 0:
            dpg.add_text(
                "No data to display",
                parent=self.table_container_tag,
                color=(149, 165, 166)
            )
            return
        
        # Limit rows for display, but show all columns with horizontal scroll
        display_df = df.head(self.MAX_PREVIEW_ROWS)
        columns = list(df.columns)  # Show all columns
        
        # Create table with horizontal scrolling enabled
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
            # Add all columns with minimum width
            for col in columns:
                dpg.add_table_column(
                    label=str(col),
                    width_fixed=True,
                    init_width_or_weight=120
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
            
            # Show if truncated (add message in first column only)
            if len(df) > self.MAX_PREVIEW_ROWS:
                with dpg.table_row():
                    dpg.add_text(
                        f"... {len(df) - self.MAX_PREVIEW_ROWS:,} more rows (scroll to see all columns)",
                        color=(149, 165, 166)
                    )
                    # Add empty cells for remaining columns
                    for _ in range(len(columns) - 1):
                        dpg.add_text("")
    
    def show_export_dialog(self) -> None:
        """Show the export file dialog."""
        if self.current_df is None:
            self._show_error("No data to export")
            return
        
        # Ensure exports directory exists
        exports_dir = Path("exports")
        exports_dir.mkdir(exist_ok=True)
        
        format_value = dpg.get_value(self.export_format_tag)
        extension = {
            "CSV": "csv",
            "TSV": "tsv",
            "Excel (.xlsx)": "xlsx",
            "Parquet": "parquet",
            "JSON": "json"
        }.get(format_value, "csv")
        
        # Create save dialog
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self._on_export_dialog_ok,
            cancel_callback=lambda: None,
            width=700,
            height=400,
            modal=True,
            tag="export_dialog",
            default_filename=f"bids_output.{extension}",
            default_path=str(exports_dir.absolute())
        ):
            dpg.add_file_extension(f".{extension}", color=(0, 255, 0))
    
    def _on_export_dialog_ok(self, sender, app_data) -> None:
        """Handle export dialog OK."""
        file_path = app_data.get("file_path_name", "")
        
        if not file_path:
            return
        
        self._export_to_file(file_path)
        dpg.delete_item("export_dialog")
    
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
    
    def _copy_to_clipboard(self) -> None:
        """Copy data to clipboard."""
        if self.current_df is None:
            self._show_error("No data to copy")
            return
        
        try:
            # Copy as TSV (works well for pasting into Excel/Sheets)
            self.current_df.to_clipboard(index=False)
            self._show_success("Copied to clipboard!")
        except Exception as e:
            self._show_error(f"Copy failed: {str(e)}")
    
    def compare_dataframes(self, original: pd.DataFrame, fixed: pd.DataFrame) -> None:
        """
        Show comparison between original and fixed DataFrames.
        
        Args:
            original: Original DataFrame
            fixed: Fixed DataFrame
        """
        self.original_df = original
        self.current_df = fixed
        
        # Clear container
        dpg.delete_item(self.table_container_tag, children_only=True)
        
        # Add comparison stats
        with dpg.group(parent=self.table_container_tag):
            dpg.add_text("Comparison Summary", color=(100, 149, 237))
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            # Row changes
            with dpg.group(horizontal=True):
                dpg.add_text("Rows:", color=(149, 165, 166))
                dpg.add_text(f"{len(original)} -> {len(fixed)}")
                if len(fixed) != len(original):
                    change = len(fixed) - len(original)
                    color = (46, 204, 113) if change < 0 else (241, 196, 15)
                    dpg.add_text(f" ({change:+d})", color=color)
            
            # Column changes
            with dpg.group(horizontal=True):
                dpg.add_text("Columns:", color=(149, 165, 166))
                dpg.add_text(f"{len(original.columns)} -> {len(fixed.columns)}")
            
            # Null changes
            orig_nulls = original.isnull().sum().sum()
            fixed_nulls = fixed.isnull().sum().sum()
            with dpg.group(horizontal=True):
                dpg.add_text("Null values:", color=(149, 165, 166))
                dpg.add_text(f"{orig_nulls:,} -> {fixed_nulls:,}")
                if fixed_nulls < orig_nulls:
                    dpg.add_text(
                        f" (-{orig_nulls - fixed_nulls:,})",
                        color=(46, 204, 113)
                    )
            
            dpg.add_spacer(height=10)
            dpg.add_text("Fixed Data Preview:", color=(100, 149, 237))
            dpg.add_spacer(height=5)
        
        # Show fixed data table
        self._build_table(fixed)
    
    def _show_error(self, message: str) -> None:
        """Show error popup."""
        with dpg.window(
            label="Error",
            modal=True,
            width=300,
            height=100,
            pos=[550, 400],
            on_close=lambda: dpg.delete_item("results_error_popup")
        ) as popup:
            dpg.set_item_alias(popup, "results_error_popup")
            dpg.add_text(message, wrap=-1, color=(231, 76, 60))
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="OK",
                callback=lambda: dpg.delete_item("results_error_popup"),
                width=-1
            )
    
    def _show_success(self, message: str) -> None:
        """Show success popup."""
        with dpg.window(
            label="Success",
            modal=True,
            width=300,
            height=100,
            pos=[550, 400],
            on_close=lambda: dpg.delete_item("results_success_popup")
        ) as popup:
            dpg.set_item_alias(popup, "results_success_popup")
            dpg.add_text(message, wrap=-1, color=(46, 204, 113))
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="OK",
                callback=lambda: dpg.delete_item("results_success_popup"),
                width=-1
            )
    
    def get_current_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the currently displayed DataFrame."""
        return self.current_df

