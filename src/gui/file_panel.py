"""
File Panel for BIDS GUI

Handles file selection, loading, and preview.
"""

import dearpygui.dearpygui as dpg
from pathlib import Path
from typing import Optional, Callable, List
import pandas as pd
import threading

# Import will be done lazily to avoid circular imports
SmartIngestor = None


def get_ingestor():
    """Lazy import of SmartIngestor."""
    global SmartIngestor
    if SmartIngestor is None:
        from ..ingestion import SmartIngestor as SI
        SmartIngestor = SI
    return SmartIngestor()


class FilePanel:
    """
    File selection and management panel.
    
    Features:
    - File browser dialog
    - Drag and drop support (planned)
    - File list management
    - Quick file info preview
    """
    
    SUPPORTED_EXTENSIONS = [
        ".csv", ".tsv", ".xlsx", ".xls", ".parquet",
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
        ".json", ".txt"
    ]
    
    def __init__(self, on_file_loaded: Optional[Callable] = None):
        """
        Initialize FilePanel.
        
        Args:
            on_file_loaded: Callback when file is loaded (df, file_path)
        """
        self.on_file_loaded = on_file_loaded
        self.loaded_files: List[str] = []
        self.current_df: Optional[pd.DataFrame] = None
        
        # UI element tags
        self.file_list_tag: Optional[int] = None
        self.info_text_tag: Optional[int] = None
        self.loading_indicator_tag: Optional[int] = None
    
    def create(self) -> None:
        """Create the file panel UI."""
        dpg.add_text("Input Files", color=(100, 149, 237))
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        # File list
        with dpg.child_window(height=250, border=False):
            self.file_list_tag = dpg.add_listbox(
                items=[],
                num_items=10,
                callback=self._on_file_selected,
                width=-1
            )
        
        dpg.add_spacer(height=10)
        
        # Action buttons
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="+ Add Files",
                callback=self.show_file_dialog,
                width=100
            )
            dpg.add_button(
                label="Clear All",
                callback=self._clear_files,
                width=80
            )
        
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        # File info section
        dpg.add_text("File Info", color=(100, 149, 237))
        dpg.add_spacer(height=5)
        
        with dpg.child_window(height=150, border=False):
            self.info_text_tag = dpg.add_text(
                "No file selected",
                wrap=280,
                color=(149, 165, 166)
            )
            
            self.loading_indicator_tag = dpg.add_loading_indicator(
                show=False,
                radius=2.0,
                color=(100, 149, 237)
            )
        
        # Create file dialog (hidden initially)
        self._create_file_dialog()
    
    def _create_file_dialog(self) -> None:
        """Create the file browser dialog."""
        # Ensure inputs directory exists
        inputs_dir = Path("inputs")
        inputs_dir.mkdir(exist_ok=True)
        
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_file_dialog_ok,
            cancel_callback=self._on_file_dialog_cancel,
            width=700,
            height=400,
            modal=True,
            tag="file_dialog"
        ):
            # Add file type filters
            dpg.add_file_extension(".*", color=(255, 255, 255))
            dpg.add_file_extension(".csv", color=(0, 255, 0))
            dpg.add_file_extension(".tsv", color=(0, 255, 0))
            dpg.add_file_extension(".xlsx", color=(0, 200, 255))
            dpg.add_file_extension(".xls", color=(0, 200, 255))
            dpg.add_file_extension(".parquet", color=(255, 200, 0))
            dpg.add_file_extension(".zip", color=(255, 100, 100))
            dpg.add_file_extension(".tar", color=(255, 100, 100))
            dpg.add_file_extension(".gz", color=(255, 100, 100))
            dpg.add_file_extension(".7z", color=(255, 100, 100))
            dpg.add_file_extension(".rar", color=(255, 100, 100))
    
    def show_file_dialog(self) -> None:
        """Show the file browser dialog."""
        # Ensure inputs directory exists
        inputs_dir = Path("inputs").absolute()
        inputs_dir.mkdir(exist_ok=True)
        
        # Try to set the file dialog path (if supported)
        try:
            # Dear PyGui may support setting the path via configure_item
            dpg.configure_item("file_dialog", default_path=str(inputs_dir))
        except:
            # If not supported, the dialog will open in the default location
            pass
        
        dpg.show_item("file_dialog")
    
    def _on_file_dialog_ok(self, sender, app_data) -> None:
        """Handle file dialog OK callback."""
        selections = app_data.get("selections", {})
        
        for file_name, file_path in selections.items():
            self._add_file(file_path)
    
    def _on_file_dialog_cancel(self, sender, app_data) -> None:
        """Handle file dialog cancel."""
        pass
    
    def _add_file(self, file_path: str) -> None:
        """Add a file to the list and load it."""
        if file_path in self.loaded_files:
            return
        
        self.loaded_files.append(file_path)
        self._update_file_list()
        
        # Auto-select and load the new file
        self._load_file(file_path)
    
    def _update_file_list(self) -> None:
        """Update the file listbox."""
        display_names = [Path(f).name for f in self.loaded_files]
        dpg.configure_item(self.file_list_tag, items=display_names)
    
    def _on_file_selected(self, sender, app_data) -> None:
        """Handle file selection from list."""
        selected_name = app_data
        
        # Find the full path
        for path in self.loaded_files:
            if Path(path).name == selected_name:
                self._load_file(path)
                break
    
    def _load_file(self, file_path: str) -> None:
        """Load a file in a background thread."""
        # Show loading indicator
        dpg.configure_item(self.loading_indicator_tag, show=True)
        dpg.set_value(self.info_text_tag, f"Loading {Path(file_path).name}...")
        
        # Load in background thread
        thread = threading.Thread(
            target=self._load_file_thread,
            args=(file_path,)
        )
        thread.daemon = True
        thread.start()
    
    def _load_file_thread(self, file_path: str) -> None:
        """Background thread for loading files."""
        try:
            ingestor = get_ingestor()
            result = ingestor.ingest(file_path)
            
            if result.dataframes:
                df = result.dataframes[0]
                
                # Pre-process: Remove columns that are all NULL
                original_cols = len(df.columns)
                df = df.dropna(axis=1, how='all')
                removed_cols = original_cols - len(df.columns)
                
                if removed_cols > 0:
                    # Log the removal (will be shown in info text below)
                    pass
                
                self.current_df = df
                
                # Update UI in main thread
                dpg.configure_item(self.loading_indicator_tag, show=False)
                
                # Build info text
                info = self._build_file_info(file_path, df, result)
                if removed_cols > 0:
                    info = f"Removed {removed_cols} empty column(s)\n\n" + info
                dpg.set_value(self.info_text_tag, info)
                
                # Trigger callback
                if self.on_file_loaded:
                    self.on_file_loaded(df, file_path)
            else:
                dpg.configure_item(self.loading_indicator_tag, show=False)
                dpg.set_value(
                    self.info_text_tag,
                    f"Failed to load file:\n{', '.join(result.errors)}"
                )
                
        except Exception as e:
            dpg.configure_item(self.loading_indicator_tag, show=False)
            dpg.set_value(
                self.info_text_tag,
                f"Error loading file:\n{str(e)}"
            )
    
    def _build_file_info(self, file_path: str, df: pd.DataFrame, result) -> str:
        """Build file info display string."""
        path = Path(file_path)
        size = path.stat().st_size
        
        # Format size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        
        info_lines = [
            f"File: {path.name}",
            f"Size: {size_str}",
            f"",
            f"Rows: {len(df):,}",
            f"Columns: {len(df.columns)}",
            f"",
            "Columns:",
        ]
        
        for col in df.columns[:8]:  # Show first 8 columns
            dtype = str(df[col].dtype)
            nulls = df[col].isnull().sum()
            null_str = f" ({nulls} nulls)" if nulls > 0 else ""
            info_lines.append(f"  - {col}: {dtype}{null_str}")
        
        if len(df.columns) > 8:
            info_lines.append(f"  ... and {len(df.columns) - 8} more")
        
        if result.metadata.get("used_llm_fallback"):
            info_lines.append("")
            info_lines.append("(Parsed with LLM assistance)")
        
        return "\n".join(info_lines)
    
    def _clear_files(self) -> None:
        """Clear all loaded files."""
        self.loaded_files = []
        self.current_df = None
        
        dpg.configure_item(self.file_list_tag, items=[])
        dpg.set_value(self.info_text_tag, "No file selected")
    
    def get_current_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the currently loaded DataFrame."""
        return self.current_df
    
    def get_loaded_files(self) -> List[str]:
        """Get list of loaded file paths."""
        return self.loaded_files.copy()
