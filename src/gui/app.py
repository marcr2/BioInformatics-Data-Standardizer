"""
BIDS GUI Application

Main Dear PyGui application for BIDS.
"""

import dearpygui.dearpygui as dpg
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading
import queue

from .file_panel import FilePanel
from .schema_editor import SchemaEditor
from .process_panel import ProcessPanel
from .results_panel import ResultsPanel
from .embedding_map_panel import EmbeddingMapPanel
from .agent_monitor_panel import AgentMonitorPanel, set_agent_monitor
from .cycle_visualization_panel import CycleVisualizationPanel
from .data_state_panel import DataStatePanel


class BIDSApp:
    """
    Main BIDS GUI Application using Dear PyGui.
    
    Features:
    - File selection and preview
    - Schema selection and custom editor
    - Processing status and logs
    - Results viewing and export
    """
    
    # Theme colors
    COLORS = {
        "background": (25, 25, 35),
        "panel": (35, 35, 50),
        "accent": (100, 149, 237),  # Cornflower blue
        "accent_hover": (130, 170, 255),
        "success": (46, 204, 113),
        "warning": (241, 196, 15),
        "error": (231, 76, 60),
        "text": (236, 240, 241),
        "text_dim": (149, 165, 166),
    }
    
    def __init__(self, width: int = None, height: int = None):
        """
        Initialize the BIDS application.
        
        Args:
            width: Window width (if None, loads from preferences)
            height: Window height (if None, loads from preferences)
        """
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        
        # Load resolution from preferences if not provided
        if width is None or height is None:
            pref_width, pref_height = prefs.get_window_size()
            self.width = width if width is not None else pref_width
            self.height = height if height is not None else pref_height
        else:
            self.width = width
            self.height = height
        
        # State
        self.current_df = None
        self.current_schema = None
        self.processing = False
        
        # Thread-safe UI update queue
        self.ui_update_queue = queue.Queue()
        
        # Panels (initialized in setup)
        self.file_panel: Optional[FilePanel] = None
        self.schema_editor: Optional[SchemaEditor] = None
        self.process_panel: Optional[ProcessPanel] = None
        self.results_panel: Optional[ResultsPanel] = None
        self.embedding_map_panel: Optional[EmbeddingMapPanel] = None
        self.agent_monitor_panel: Optional[AgentMonitorPanel] = None
        self.cycle_viz_panel: Optional[CycleVisualizationPanel] = None
        self.data_state_panel: Optional[DataStatePanel] = None
        
        # Checkpoint manager (shared)
        from ..utils.checkpoint_manager import CheckpointManager
        self.checkpoint_manager = CheckpointManager()
        
        # Base resolution for scaling calculations (reference resolution)
        self.base_width = 1400
        self.base_height = 900
        
        # Calculate scaling factors
        self.scale_x = self.width / self.base_width
        self.scale_y = self.height / self.base_height
        # Use average scale for consistent proportions
        self.scale = (self.scale_x + self.scale_y) / 2.0
    
    def scale_size(self, size: int) -> int:
        """Scale a size value based on window resolution."""
        return int(size * self.scale)
    
    def scale_width(self, width: int) -> int:
        """Scale a width value based on window resolution."""
        return int(width * self.scale_x)
    
    def scale_height(self, height: int) -> int:
        """Scale a height value based on window resolution."""
        return int(height * self.scale_y)
    
    def setup(self) -> None:
        """Set up Dear PyGui context and windows."""
        dpg.create_context()
        
        # Ensure inputs directory exists
        inputs_dir = Path("inputs")
        inputs_dir.mkdir(exist_ok=True)
        
        # Ensure cache directory exists
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Enable DPI awareness for crisp rendering on high-DPI displays
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            pass  # DPI awareness not available on this system
        
        # Configure viewport with resolution from preferences
        dpg.create_viewport(
            title="BIDS - Bioinformatics Data Standardizer",
            width=self.width,
            height=self.height,
            min_width=self.scale_width(1000),
            min_height=self.scale_height(700)
        )
        
        # Apply theme
        self._setup_theme()
        
        # Register fonts
        self._setup_fonts()
        
        # Create main window
        self._create_main_window()
        
        # Setup and show viewport
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def _setup_theme(self) -> None:
        """Set up the application theme."""
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Window
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, self.COLORS["background"])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, self.COLORS["panel"])
                
                # Text
                dpg.add_theme_color(dpg.mvThemeCol_Text, self.COLORS["text"])
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, self.COLORS["text_dim"])
                
                # Buttons
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.COLORS["accent"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self.COLORS["accent_hover"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.COLORS["accent"])
                
                # Headers
                dpg.add_theme_color(dpg.mvThemeCol_Header, self.COLORS["accent"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, self.COLORS["accent_hover"])
                
                # Frame
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (45, 45, 60))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (55, 55, 75))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (65, 65, 85))
                
                # Borders
                dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 60, 80))
                
                # Scrollbar
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, self.COLORS["panel"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, self.COLORS["accent"])
                
                # Tab
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (45, 45, 65))
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, self.COLORS["accent_hover"])
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, self.COLORS["accent"])
                
                # Table
                dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, (50, 50, 70))
                dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, (35, 35, 50))
                dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, (40, 40, 55))
                
                # Styling
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 12)
        
        dpg.bind_theme(self.global_theme)
        
        # Success button theme
        with dpg.theme() as self.success_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.COLORS["success"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (56, 224, 123))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.COLORS["success"])
        
        # Error button theme
        with dpg.theme() as self.error_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.COLORS["error"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (241, 86, 70))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.COLORS["error"])
    
    def _setup_fonts(self) -> None:
        """Set up fonts with cross-platform support."""
        import os
        import platform
        from pathlib import Path
        
        font_path = None
        system = platform.system()
        
        if system == "Windows":
            # Windows font paths
            windir = os.environ.get('WINDIR', 'C:\\Windows')
            windows_fonts = [
                Path(windir) / 'Fonts' / 'segoeui.ttf',
                Path(windir) / 'Fonts' / 'arial.ttf',
                Path(windir) / 'Fonts' / 'tahoma.ttf',
            ]
            
            for path in windows_fonts:
                if path.exists():
                    font_path = str(path)
                    break
        elif system == "Linux":
            # Linux font paths (common locations)
            linux_fonts = [
                Path.home() / '.fonts' / 'DejaVuSans.ttf',
                Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'),
                Path('/usr/share/fonts/TTF/DejaVuSans.ttf'),
                Path('/usr/local/share/fonts/DejaVuSans.ttf'),
                Path('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf'),
            ]
            
            for path in linux_fonts:
                if path.exists():
                    font_path = str(path)
                    break
        # macOS could be added here if needed
        # elif system == "Darwin":
        #     mac_fonts = [
        #         Path('/System/Library/Fonts/Helvetica.ttc'),
        #         Path('/Library/Fonts/Arial.ttf'),
        #     ]
        #     ...
        
        if font_path:
            with dpg.font_registry():
                # Scale font size based on resolution
                font_size = max(12, int(16 * self.scale))
                self.default_font = dpg.add_font(font_path, font_size)
            dpg.bind_font(self.default_font)
        # If no font found, Dear PyGui will use its built-in default
    
    def _create_main_window(self) -> None:
        """Create the main application window."""
        with dpg.window(
            label="BIDS", 
            tag="main_window",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True
        ):
            # Menu bar
            self._create_menu_bar()
            
            # Header
            self._create_header()
            
            dpg.add_spacer(height=self.scale_size(10))
            
            # Main content area with panels
            with dpg.group(horizontal=True):
                # Left panel - File selection (scaled width)
                left_panel_width = self.scale_width(320)
                with dpg.child_window(width=left_panel_width, height=-self.scale_height(60), border=True):
                    self.file_panel = FilePanel(
                        on_file_loaded=self._on_file_loaded,
                        scale=self.scale
                    )
                    self.file_panel.create()
                
                dpg.add_spacer(width=self.scale_width(10))
                
                # Center area
                with dpg.group():
                    # Preview/Process tabs (75% width)
                    with dpg.child_window(width=-1, height=-self.scale_height(60), border=True):
                        with dpg.tab_bar():
                            # Preview tab
                            with dpg.tab(label="Data Preview"):
                                self.results_panel = ResultsPanel()
                                self.results_panel.create()
                            
                            # Schema tab
                            with dpg.tab(label="Schema Editor"):
                                self.schema_editor = SchemaEditor(
                                    on_schema_changed=self._on_schema_changed
                                )
                                self.schema_editor.create()
                            
                            # Processing tab
                            with dpg.tab(label="Processing"):
                                self.process_panel = ProcessPanel(
                                    on_process_complete=self._on_process_complete,
                                    main_app=self,
                                    scale=self.scale
                                )
                                # Create process panel first
                                self.process_panel.create()
                                # Set checkpoint manager (will update orchestrator)
                                self.process_panel.set_checkpoint_manager(self.checkpoint_manager)
                            
                            # Embedding Map tab - Interactive visualization
                            with dpg.tab(label="Embedding Map"):
                                self.embedding_map_panel = EmbeddingMapPanel(
                                    get_current_data_callback=lambda: self.current_df,
                                    scale=self.scale
                                )
                                self.embedding_map_panel.create()
                            
                            # Agent Monitor tab - Real-time LLM activity
                            with dpg.tab(label="Agent Monitor"):
                                self.agent_monitor_panel = AgentMonitorPanel(scale=self.scale)
                                self.agent_monitor_panel.create()
                                # Register as global monitor for callbacks
                                set_agent_monitor(self.agent_monitor_panel)
                            
                            # Cycle Visualization tab - Processing cycle diagram
                            with dpg.tab(label="Cycle Visualization"):
                                self.cycle_viz_panel = CycleVisualizationPanel(
                                    checkpoint_manager=self.checkpoint_manager
                                )
                                self.cycle_viz_panel.create()
                            
                            # Data State tab - Last checkpoint data
                            with dpg.tab(label="Data State"):
                                self.data_state_panel = DataStatePanel(
                                    checkpoint_manager=self.checkpoint_manager
                                )
                                self.data_state_panel.create()
            
            dpg.add_spacer(height=self.scale_size(10))
            
            # Bottom status bar
            self._create_status_bar()
        
        # Set primary window
        dpg.set_primary_window("main_window", True)
    
    def _create_menu_bar(self) -> None:
        """Create the menu bar."""
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(
                    label="Open File...", 
                    callback=self._menu_open_file
                )
                dpg.add_menu_item(
                    label="Open Folder...", 
                    callback=self._menu_open_folder
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="Export Results...",
                    callback=self._menu_export
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="Exit",
                    callback=self._menu_exit
                )
            
            with dpg.menu(label="Schema"):
                dpg.add_menu_item(
                    label="IPA Standard",
                    callback=lambda: self._select_schema("IPA Standard")
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="New Custom Schema...",
                    callback=self._menu_new_schema
                )
                dpg.add_menu_item(
                    label="Load Schema...",
                    callback=self._menu_load_schema
                )
            
            with dpg.menu(label="Settings"):
                dpg.add_menu_item(
                    label="Preferences...",
                    callback=self._menu_preferences
                )
            
            with dpg.menu(label="Help"):
                dpg.add_menu_item(
                    label="Documentation",
                    callback=self._menu_documentation
                )
                dpg.add_menu_item(
                    label="About BIDS",
                    callback=self._menu_about
                )
    
    def _create_header(self) -> None:
        """Create the header section."""
        with dpg.group(horizontal=True):
            dpg.add_text("BIDS", color=self.COLORS["accent"])
            dpg.add_text(" - Bioinformatics Data Standardizer", color=self.COLORS["text_dim"])
            
            dpg.add_spacer(width=-1)
            
            # Quick action buttons
            self.btn_diagnose = dpg.add_button(
                label="Diagnose",
                callback=self._on_diagnose_click,
                enabled=False
            )
            dpg.add_spacer(width=5)
            self.btn_fix = dpg.add_button(
                label="Run Full Process",
                callback=self._on_fix_click,
                enabled=False
            )
            dpg.bind_item_theme(self.btn_fix, self.success_theme)
    
    def _create_status_bar(self) -> None:
        """Create the status bar."""
        with dpg.group(horizontal=True):
            self.status_text = dpg.add_text("Ready", color=self.COLORS["text_dim"])
            
            dpg.add_spacer(width=-1)
            
            self.progress_bar = dpg.add_progress_bar(
                default_value=0.0,
                width=self.scale_width(200),
                overlay="Idle"
            )
            
            dpg.add_spacer(width=self.scale_width(10))
            
            self.schema_label = dpg.add_text(
                "Schema: None",
                color=self.COLORS["text_dim"]
            )
    
    # Callback handlers
    def _on_file_loaded(self, df, file_path: str) -> None:
        """Handle file loaded callback (may be called from background thread)."""
        # Queue UI update to run in main thread
        self.ui_update_queue.put(('file_loaded', df, file_path))
        
        # Update process panel with data
        if self.process_panel:
            self.process_panel.set_data(df, self.current_schema)
    
    def _process_ui_updates(self) -> None:
        """Process queued UI updates from background threads."""
        try:
            while True:
                update_type, *args = self.ui_update_queue.get_nowait()
                
                if update_type == 'file_loaded':
                    df, file_path = args
                    self.current_df = df
                    
                    # Update status
                    dpg.set_value(self.status_text, f"Loaded: {Path(file_path).name}")
                    
                    # Enable buttons
                    dpg.configure_item(self.btn_diagnose, enabled=True)
                    if self.current_schema:
                        dpg.configure_item(self.btn_fix, enabled=True)
                    
                    # Update preview
                    if self.results_panel:
                        self.results_panel.show_dataframe(df)
                    
                    # Update process panel with data
                    if self.process_panel:
                        self.process_panel.set_data(df, self.current_schema)
        except queue.Empty:
            pass
    
    def _on_schema_changed(self, schema) -> None:
        """Handle schema changed callback."""
        self.current_schema = schema
        
        # Update label
        dpg.set_value(self.schema_label, f"Schema: {schema.name}")
        
        # Enable fix button if data loaded
        if self.current_df is not None:
            dpg.configure_item(self.btn_fix, enabled=True)
        
        # Update process panel with schema
        if self.process_panel and self.current_df is not None:
            self.process_panel.set_data(self.current_df, schema)
    
    def _on_process_complete(self, result: Dict[str, Any]) -> None:
        """Handle processing complete callback."""
        self.processing = False
        dpg.set_value(self.progress_bar, 1.0)
        dpg.configure_item(self.progress_bar, overlay="Complete")
        
        if result.get("success"):
            dpg.set_value(self.status_text, "Processing complete - Success!")
            if result.get("final_df") is not None:
                self.results_panel.show_dataframe(result["final_df"])
        else:
            dpg.set_value(self.status_text, "Processing complete - Issues remain")
    
    def _on_diagnose_click(self) -> None:
        """Handle diagnose button click - updates both main status and processing tab."""
        if self.current_df is None:
            return
        
        # Update main status bar
        dpg.set_value(self.status_text, "Diagnosing...")
        dpg.set_value(self.progress_bar, 0.3)
        dpg.configure_item(self.progress_bar, overlay="Diagnosing...")
        
        # Update processing tab status
        if self.process_panel:
            self.process_panel._update_status("Diagnosing...", 0.3)
            self.process_panel._log("Starting diagnosis...")
            self.process_panel.run_diagnosis(
                self.current_df,
                self.current_schema
            )
    
    def _on_fix_click(self) -> None:
        """Handle fix button click - updates both main status and processing tab."""
        if self.current_df is None or self.current_schema is None:
            return
        
        self.processing = True
        
        # Update main status bar
        dpg.set_value(self.status_text, "Processing...")
        dpg.set_value(self.progress_bar, 0.0)
        dpg.configure_item(self.progress_bar, overlay="Processing...")
        
        # Update processing tab status
        if self.process_panel:
            self.process_panel._update_status("Processing...", 0.1)
            self.process_panel._log("Starting full process...")
            self.process_panel.run_full_process(
                self.current_df,
                self.current_schema
            )
    
    def _select_schema(self, name: str) -> None:
        """Select a schema by name."""
        if self.schema_editor:
            self.schema_editor.load_schema(name)
    
    # Menu callbacks
    def _menu_open_file(self) -> None:
        """Open file menu callback."""
        if self.file_panel:
            self.file_panel.show_file_dialog()
    
    def _menu_open_folder(self) -> None:
        """Open folder menu callback."""
        pass  # TODO: Implement folder opening
    
    def _menu_export(self) -> None:
        """Export menu callback."""
        if self.results_panel:
            self.results_panel.show_export_dialog()
    
    def _menu_exit(self) -> None:
        """Exit menu callback."""
        dpg.stop_dearpygui()
    
    def _menu_new_schema(self) -> None:
        """New schema menu callback."""
        if self.schema_editor:
            self.schema_editor.new_schema()
    
    def _menu_load_schema(self) -> None:
        """Load schema menu callback."""
        if self.schema_editor:
            self.schema_editor.show_load_dialog()
    
    def _menu_preferences(self) -> None:
        """Preferences menu callback."""
        self._show_preferences_dialog()
    
    def _menu_documentation(self) -> None:
        """Documentation menu callback."""
        import webbrowser
        webbrowser.open("https://github.com/marcr2/BioInformatics-Data-Standardizer")
    
    def _menu_about(self) -> None:
        """About menu callback."""
        self._show_about_dialog()
    
    def _show_about_dialog(self) -> None:
        """Show the about dialog."""
        dialog_width = self.scale_width(400)
        dialog_height = self.scale_height(250)
        with dpg.window(
            label="About BIDS",
            modal=True,
            width=dialog_width,
            height=dialog_height,
            pos=[int(self.width * 0.3), int(self.height * 0.3)],
            on_close=lambda: dpg.delete_item("about_dialog")
        ) as dialog:
            dpg.set_item_alias(dialog, "about_dialog")
            
            dpg.add_text("BIDS", color=self.COLORS["accent"])
            dpg.add_text("Bioinformatics Data Standardizer")
            dpg.add_spacer(height=10)
            dpg.add_text("Version 1.0.0")
            dpg.add_spacer(height=10)
            dpg.add_text(
                "A system for cleaning and standardizing\n"
                "messy clinical/bioinformatics data.",
                color=self.COLORS["text_dim"]
            )
            dpg.add_spacer(height=20)
            dpg.add_text("Powered by:")
            dpg.add_text("  - Local LLM (100% Private)")
            dpg.add_text("  - No API calls, all processing local")
    
    def _show_preferences_dialog(self) -> None:
        """Show the preferences dialog."""
        from ..utils.preferences import get_preferences
        
        # Delete existing preferences dialog and its tagged items if they exist
        if dpg.does_item_exist("preferences_dialog"):
            dpg.delete_item("preferences_dialog")
        
        # Also delete any tagged items that might persist
        for tag in ["max_tokens_display", "max_tokens_slider", "max_tokens_percent_label"]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        
        prefs = get_preferences()
        gpu_info = prefs.get_gpu_info()
        
        dialog_width = self.scale_width(600)
        dialog_height = self.scale_height(650)
        with dpg.window(
            label="Preferences",
            modal=True,
            width=dialog_width,
            height=dialog_height,
            pos=[int(self.width * 0.2), int(self.height * 0.1)],
            on_close=lambda: dpg.delete_item("preferences_dialog")
        ) as dialog:
            dpg.set_item_alias(dialog, "preferences_dialog")
            
            dpg.add_text("Application Preferences", color=self.COLORS["accent"])
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # GPU Acceleration Settings
            dpg.add_text("GPU Acceleration", color=self.COLORS["text_dim"])
            dpg.add_spacer(height=5)
            
            # GPU status
            if gpu_info["available"]:
                gpu_status_text = f"GPU: {gpu_info['device_name']} ({gpu_info['memory_gb']:.1f} GB)"
                gpu_status_color = self.COLORS["success"]
            else:
                gpu_status_text = "GPU: Not available (will use CPU)"
                gpu_status_color = self.COLORS["text_dim"]
            
            dpg.add_text(gpu_status_text, color=gpu_status_color)
            dpg.add_spacer(height=5)
            
            # GPU toggle checkbox
            gpu_enabled = prefs.is_gpu_enabled() if gpu_info["available"] else False
            gpu_checkbox = dpg.add_checkbox(
                label="Enable GPU Acceleration",
                default_value=gpu_enabled,
                enabled=gpu_info["available"],
                callback=self._on_gpu_toggle_changed
            )
            
            if not gpu_info["available"]:
                dpg.add_text(
                    "Note: GPU not detected. Install CUDA-enabled PyTorch for GPU acceleration.",
                    color=self.COLORS["warning"],
                    wrap=-1
                )
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Processing Mode Settings
            from ..utils.llm_check import is_llm_available
            llm_available = is_llm_available()
            prefs.set("llm_installed", llm_available)
            
            dpg.add_text("Processing Mode", color=self.COLORS["text_dim"])
            dpg.add_spacer(height=5)
            
            # LLM Status
            if llm_available:
                llm_status_text = "LLM Support: Available"
                llm_status_color = self.COLORS["success"]
            else:
                llm_status_text = "LLM Support: Not installed"
                llm_status_color = self.COLORS["warning"]
            
            dpg.add_text(llm_status_text, color=llm_status_color)
            dpg.add_spacer(height=5)
            
            # Processing mode radio buttons
            current_mode = prefs.get_processing_mode()
            dpg.add_radio_button(
                items=["Auto (use LLM if available)", "Rules-based only (no LLM)", "LLM required"],
                default_value=0 if current_mode == "auto" else (1 if current_mode == "rules_only" else 2),
                callback=self._on_processing_mode_changed,
                tag="processing_mode_radio"
            )
            
            dpg.add_spacer(height=5)
            
            # Install LLM button (only show if not installed)
            if not llm_available:
                dpg.add_button(
                    label="Install LLM Support",
                    callback=self._on_install_llm_clicked,
                    tag="install_llm_button"
                )
                dpg.add_text(
                    "Click to install LLM dependencies (~8-16GB download).\n"
                    "This may take several minutes.",
                    color=(149, 165, 166),
                    wrap=-1
                )
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # LLM Token Settings (only show if LLM is available)
            if llm_available:
                dpg.add_text("LLM Token Settings", color=self.COLORS["text_dim"])
            dpg.add_spacer(height=5)
            
            token_info = prefs.get_max_tokens_info()
            
            # Build display text with model memory info
            if token_info.get('model_id') and token_info.get('model_memory_gb'):
                model_name = token_info['model_id'].split('/')[-1]  # Get just the model name
                base_pct = token_info.get('base_percentage', 0)
                display_text = (
                    f"Current max tokens: {token_info['max_tokens']} "
                    f"({token_info['percentage']}% of available memory after model)\n"
                    f"Model: {model_name} ({token_info['model_memory_gb']:.2f} GB, {base_pct:.1f}% of GPU)"
                )
            else:
                display_text = (
                    f"Current max tokens: {token_info['max_tokens']} "
                    f"({token_info['percentage']}% of GPU memory)\n"
                    f"Model not loaded yet - will auto-detect when loaded"
                )
            
            dpg.add_text(
                display_text,
                color=(149, 165, 166),
                tag="max_tokens_display",
                wrap=-1
            )
            dpg.add_spacer(height=5)
            
            with dpg.group(horizontal=True):
                dpg.add_text("Max Tokens (% of available):", color=self.COLORS["text_dim"])
                dpg.add_spacer(width=10)
                dpg.add_slider_int(
                    default_value=int(token_info['percentage']),
                    min_value=10,
                    max_value=100,
                    width=self.scale_width(200),
                    callback=self._on_max_tokens_changed,
                    tag="max_tokens_slider"
                )
                dpg.add_spacer(width=10)
                dpg.add_text(f"{token_info['percentage']}%", tag="max_tokens_percent_label")
            
            if token_info['gpu_available'] and token_info['gpu_memory_gb']:
                gpu_text = f"Total GPU Memory: {token_info['gpu_memory_gb']:.1f} GB"
                if token_info.get('available_memory_gb'):
                    gpu_text += f", Available: {token_info['available_memory_gb']:.2f} GB"
                gpu_text += f", ~{token_info['tokens_per_gb']} tokens/GB"
                dpg.add_text(
                    gpu_text,
                    color=(149, 165, 166)
                )
            else:
                dpg.add_text(
                    "GPU not available - using default token limit (4096)",
                    color=self.COLORS["warning"]
                )
            
                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_spacer(height=10)
            
            # LLM-Rewrite Settings (only show if LLM is available)
            if llm_available:
                dpg.add_text("LLM-Rewrite Step", color=self.COLORS["text_dim"])
                dpg.add_spacer(height=5)
                
                dpg.add_text(
                    "Optional step that uses LLM to reorganize data structure to match schema.",
                    color=(149, 165, 166),
                    wrap=-1
                )
                dpg.add_text(
                    "Only affects column names/organization - data values are NOT modified.",
                    color=(149, 165, 166),
                    wrap=-1
                )
                dpg.add_spacer(height=5)
                
                llm_rewrite_enabled = prefs.get("llm_rewrite_enabled", False)
                dpg.add_checkbox(
                    label="Enable LLM-Rewrite Step (structural reorganization only)",
                    default_value=llm_rewrite_enabled,
                    callback=self._on_llm_rewrite_toggle_changed
                )
            else:
                # Show disabled message if LLM not available
                dpg.add_text("LLM-Rewrite Step", color=self.COLORS["text_dim"])
                dpg.add_spacer(height=5)
                dpg.add_text(
                    "LLM support is not installed. Install LLM support to enable this feature.",
                    color=self.COLORS["warning"],
                    wrap=-1
                )
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # File settings
            dpg.add_text("File Settings", color=self.COLORS["text_dim"])
            dpg.add_spacer(height=5)
            
            dpg.add_text("Default Input Directory:", color=self.COLORS["text_dim"])
            inputs_path = Path("inputs").absolute()
            dpg.add_text(str(inputs_path), color=(149, 165, 166))
            dpg.add_spacer(height=10)
            
            # Display settings
            dpg.add_text("Display Settings", color=self.COLORS["text_dim"])
            dpg.add_spacer(height=5)
            
            dpg.add_text("Theme: Dark (default)", color=(149, 165, 166))
            dpg.add_spacer(height=10)
            
            # Window Resolution Settings
            dpg.add_text("Window Resolution", color=self.COLORS["text_dim"])
            dpg.add_spacer(height=5)
            
            # Get current resolution preset
            current_preset = prefs.get("resolution_preset", "auto")
            presets = prefs.get_resolution_presets()
            
            # Create list of preset labels for dropdown
            preset_items = [presets[key]["label"] for key in presets.keys()]
            current_preset_index = list(presets.keys()).index(current_preset) if current_preset in presets else 0
            
            # Resolution preset dropdown
            with dpg.group(horizontal=True):
                dpg.add_text("Resolution Preset:", color=self.COLORS["text_dim"])
                dpg.add_spacer(width=10)
                resolution_combo = dpg.add_combo(
                    items=preset_items,
                    default_value=presets[current_preset]["label"],
                    width=self.scale_width(250),
                    callback=self._on_resolution_preset_changed
                )
            
            dpg.add_spacer(height=5)
            
            # Custom resolution inputs (only shown if custom is selected)
            current_width = prefs.get("window_width")
            current_height = prefs.get("window_height")
            if current_preset == "custom" and current_width and current_height:
                custom_width = current_width
                custom_height = current_height
            else:
                # Use current window size or default
                custom_width = self.width if hasattr(self, 'width') else 1400
                custom_height = self.height if hasattr(self, 'height') else 900
            
            with dpg.group(horizontal=True, show=current_preset == "custom") as custom_res_group:
                dpg.add_text("Custom Width:", color=self.COLORS["text_dim"])
                dpg.add_spacer(width=10)
                custom_width_input = dpg.add_input_int(
                    default_value=custom_width,
                    min_value=800,
                    max_value=7680,
                    width=self.scale_width(120),
                    callback=self._on_custom_width_changed
                )
            
            with dpg.group(horizontal=True, show=current_preset == "custom") as custom_res_group2:
                dpg.add_text("Custom Height:", color=self.COLORS["text_dim"])
                dpg.add_spacer(width=self.scale_width(10))
                custom_height_input = dpg.add_input_int(
                    default_value=custom_height,
                    min_value=600,
                    max_value=4320,
                    width=self.scale_width(120),
                    callback=self._on_custom_height_changed
                )
            
            # Store references for showing/hiding
            self._custom_width_input = custom_width_input
            self._custom_height_input = custom_height_input
            self._custom_width_group = custom_res_group
            self._custom_height_group = custom_res_group2
            self._resolution_combo = resolution_combo
            
            dpg.add_text(
                "Note: Resolution changes take effect after restarting the application.",
                color=self.COLORS["text_dim"],
                wrap=-1
            )
            dpg.add_spacer(height=10)
            
            # Map Canvas Settings
            dpg.add_text("Embedding Map Canvas", color=self.COLORS["text_dim"])
            dpg.add_spacer(height=5)
            
            map_width = prefs.get("map_canvas_width", 800)
            map_height = prefs.get("map_canvas_height", 500)
            map_auto_fit = prefs.get("map_auto_fit", True)
            
            with dpg.group(horizontal=True):
                dpg.add_text("Canvas Width:", color=self.COLORS["text_dim"])
                dpg.add_spacer(width=10)
                map_width_input = dpg.add_input_int(
                    default_value=map_width,
                    min_value=400,
                    max_value=2000,
                    width=self.scale_width(100),
                    callback=lambda s, a: prefs.set("map_canvas_width", a)
                )
            
            with dpg.group(horizontal=True):
                dpg.add_text("Canvas Height:", color=self.COLORS["text_dim"])
                dpg.add_spacer(width=self.scale_width(10))
                map_height_input = dpg.add_input_int(
                    default_value=map_height,
                    min_value=300,
                    max_value=1500,
                    width=self.scale_width(100),
                    callback=lambda s, a: prefs.set("map_canvas_height", a)
                )
            
            dpg.add_spacer(height=5)
            
            auto_fit_checkbox = dpg.add_checkbox(
                label="Auto-fit to screen on startup",
                default_value=map_auto_fit,
                callback=lambda s, a: prefs.set("map_auto_fit", a)
            )
            
            dpg.add_text(
                "Note: Changes take effect after restarting the application.",
                color=self.COLORS["text_dim"],
                wrap=-1
            )
            
            dpg.add_spacer(height=10)
            
            # Processing settings
            dpg.add_text("Processing Settings", color=self.COLORS["text_dim"])
            dpg.add_spacer(height=5)
            
            # Max fix attempts input
            with dpg.group(horizontal=True):
                dpg.add_text("Max fix attempts:", color=self.COLORS["text_dim"])
                dpg.add_spacer(width=10)
                max_attempts = prefs.get("max_fix_attempts", 3)
                dpg.add_input_int(
                    default_value=max_attempts,
                    min_value=1,
                    max_value=10,
                    width=self.scale_width(80),
                    callback=self._on_max_attempts_changed
                )
            
            dpg.add_spacer(height=5)
            
            # Auto-fix checkbox
            auto_fix = prefs.get("auto_fix", True)
            dpg.add_checkbox(
                label="Auto-fix enabled",
                default_value=auto_fix,
                callback=self._on_auto_fix_changed
            )
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Cache settings
            dpg.add_text("Cache Settings", color=self.COLORS["text_dim"])
            dpg.add_spacer(height=5)
            
            # Cache info
            cache_dir = Path("cache")
            cache_size = 0
            if cache_dir.exists():
                cache_size = sum(
                    f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()
                )
            cache_size_mb = cache_size / (1024 * 1024)
            
            dpg.add_text(f"Cache directory: {cache_dir.absolute()}", color=(149, 165, 166))
            dpg.add_text(f"Cache size: {cache_size_mb:.2f} MB", color=(149, 165, 166))
            dpg.add_spacer(height=5)
            
            # Clear cache button
            dpg.add_button(
                label="Clear Cache",
                callback=self._on_clear_cache,
                width=self.scale_width(150)
            )
            dpg.bind_item_theme(dpg.last_item(), self.error_theme)
    
    def _on_gpu_toggle_changed(self, sender, app_data) -> None:
        """Handle GPU toggle checkbox change."""
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        prefs.set("gpu_acceleration", app_data)
        
        # Show message
        if app_data:
            dpg.set_value(self.status_text, "GPU acceleration enabled (restart recommended)")
        else:
            dpg.set_value(self.status_text, "GPU acceleration disabled (restart recommended)")
    
    def _on_llm_rewrite_toggle_changed(self, sender, app_data) -> None:
        """Handle LLM-Rewrite toggle checkbox change."""
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        prefs.set("llm_rewrite_enabled", app_data)
        
        # Show message
        if app_data:
            dpg.set_value(self.status_text, "LLM-Rewrite step enabled (structural reorganization only)")
        else:
            dpg.set_value(self.status_text, "LLM-Rewrite step disabled")
    
    def _on_processing_mode_changed(self, sender, app_data) -> None:
        """Handle processing mode radio button change."""
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        
        # Map radio button index to mode
        mode_map = {0: "auto", 1: "rules_only", 2: "llm_required"}
        mode = mode_map.get(app_data, "auto")
        prefs.set_processing_mode(mode)
        
        # Show message
        mode_names = {
            "auto": "Auto (use LLM if available)",
            "rules_only": "Rules-based only (no LLM)",
            "llm_required": "LLM required"
        }
        dpg.set_value(self.status_text, f"Processing mode set to: {mode_names.get(mode, mode)}")
    
    def _on_install_llm_clicked(self, sender, app_data) -> None:
        """Handle Install LLM button click."""
        import threading
        from ..utils.llm_installer import install_llm_dependencies
        from ..utils.preferences import get_preferences
        from ..utils.llm_check import is_llm_available
        
        # Disable button during installation
        if dpg.does_item_exist("install_llm_button"):
            dpg.configure_item("install_llm_button", enabled=False)
        
        dpg.set_value(self.status_text, "Installing LLM dependencies... This may take several minutes.")
        
        def install_thread():
            """Run installation in background thread."""
            try:
                # Find venv path
                venv_path = None
                import sys
                from pathlib import Path
                if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                    # We're in a virtual environment
                    venv_path = Path(sys.prefix)
                
                def progress_callback(msg: str):
                    """Update status with progress messages."""
                    dpg.set_value(self.status_text, f"Installing LLM: {msg}")
                
                result = install_llm_dependencies(
                    venv_path=venv_path,
                    progress_callback=progress_callback
                )
                
                # Update preferences
                prefs = get_preferences()
                prefs.set("llm_installed", result["success"])
                
                if result["success"]:
                    dpg.set_value(self.status_text, "LLM installation complete! Please restart the application.")
                    # Hide install button and refresh preferences dialog
                    if dpg.does_item_exist("install_llm_button"):
                        dpg.delete_item("install_llm_button")
                    # Close and reopen preferences to refresh
                    if dpg.does_item_exist("preferences_dialog"):
                        dpg.delete_item("preferences_dialog")
                    self._show_preferences_dialog()
                else:
                    error_msg = "; ".join(result.get("errors", []))
                    dpg.set_value(self.status_text, f"LLM installation failed: {error_msg}")
                    if dpg.does_item_exist("install_llm_button"):
                        dpg.configure_item("install_llm_button", enabled=True)
            except Exception as e:
                dpg.set_value(self.status_text, f"LLM installation error: {str(e)}")
                if dpg.does_item_exist("install_llm_button"):
                    dpg.configure_item("install_llm_button", enabled=True)
        
        # Start installation in background thread
        thread = threading.Thread(target=install_thread, daemon=True)
        thread.start()
    
    def _on_max_tokens_changed(self, sender, app_data) -> None:
        """Handle max tokens slider change."""
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        prefs.set_max_tokens_percentage(app_data)
        
        # Update display
        token_info = prefs.get_max_tokens_info()
        if dpg.does_item_exist("max_tokens_display"):
            # Build display text with model memory info
            if token_info.get('model_id') and token_info.get('model_memory_gb'):
                model_name = token_info['model_id'].split('/')[-1]  # Get just the model name
                base_pct = token_info.get('base_percentage', 0)
                display_text = (
                    f"Current max tokens: {token_info['max_tokens']} "
                    f"({token_info['percentage']}% of available memory after model)\n"
                    f"Model: {model_name} ({token_info['model_memory_gb']:.2f} GB, {base_pct:.1f}% of GPU)"
                )
            else:
                display_text = (
                    f"Current max tokens: {token_info['max_tokens']} "
                    f"({token_info['percentage']}% of GPU memory)\n"
                    f"Model not loaded yet - will auto-detect when loaded"
                )
            dpg.set_value("max_tokens_display", display_text)
        if dpg.does_item_exist("max_tokens_percent_label"):
            dpg.set_value("max_tokens_percent_label", f"{token_info['percentage']}%")
        
        dpg.set_value(self.status_text, f"Max tokens set to {token_info['max_tokens']} ({app_data}%)")
    
    def _on_max_attempts_changed(self, sender, app_data) -> None:
        """Handle max fix attempts input change."""
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        # Ensure value is within valid range
        value = max(1, min(10, app_data))
        prefs.set("max_fix_attempts", value)
        dpg.set_value(self.status_text, f"Max fix attempts set to {value}")
    
    def _on_auto_fix_changed(self, sender, app_data) -> None:
        """Handle auto-fix checkbox change."""
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        prefs.set("auto_fix", app_data)
        
        # Show message
        status = "enabled" if app_data else "disabled"
        dpg.set_value(self.status_text, f"Auto-fix {status}")
    
    def _on_clear_cache(self) -> None:
        """Handle clear cache button click."""
        import shutil
        from pathlib import Path
        
        cache_dir = Path("cache")
        
        if not cache_dir.exists():
            dpg.set_value(self.status_text, "Cache directory does not exist")
            return
        
        # Clear checkpoint manager cache
        if self.checkpoint_manager:
            self.checkpoint_manager.clear_cache()
        
        # Clear entire cache directory
        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(exist_ok=True)
            
            # Recreate checkpoint directory
            checkpoint_dir = cache_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Update cycle visualization if available
            if self.cycle_viz_panel:
                self.cycle_viz_panel.update_stage("idle")
            
            # Update data state panel if available
            if self.data_state_panel:
                self.data_state_panel._load_latest()
            
            dpg.set_value(self.status_text, "Cache cleared successfully")
            
            # Refresh preferences dialog to show updated cache size
            # (user would need to reopen preferences to see the change)
            
        except Exception as e:
            dpg.set_value(self.status_text, f"Error clearing cache: {str(e)[:50]}")
    
    def _on_resolution_preset_changed(self, sender, app_data) -> None:
        """Handle resolution preset dropdown change."""
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        presets = prefs.get_resolution_presets()
        
        # Find the preset key from the label
        selected_preset = None
        for key, preset_data in presets.items():
            if preset_data["label"] == app_data:
                selected_preset = key
                break
        
        if selected_preset:
            prefs.set("resolution_preset", selected_preset)
            
            # If not custom, set the width/height from preset
            if selected_preset != "custom":
                preset_data = presets[selected_preset]
                prefs.set("window_width", preset_data["width"])
                prefs.set("window_height", preset_data["height"])
            
            # Show/hide custom inputs
            show_custom = selected_preset == "custom"
            if hasattr(self, '_custom_width_group'):
                dpg.configure_item(self._custom_width_group, show=show_custom)
            if hasattr(self, '_custom_height_group'):
                dpg.configure_item(self._custom_height_group, show=show_custom)
            
            dpg.set_value(self.status_text, f"Resolution preset changed to {app_data} (restart required)")
    
    def _on_custom_width_changed(self, sender, app_data) -> None:
        """Handle custom width input change."""
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        
        # Ensure value is within valid range
        value = max(800, min(7680, app_data))
        prefs.set("window_width", value)
        prefs.set("resolution_preset", "custom")
        dpg.set_value(self.status_text, f"Custom width set to {value} (restart required)")
    
    def _on_custom_height_changed(self, sender, app_data) -> None:
        """Handle custom height input change."""
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        
        # Ensure value is within valid range
        value = max(600, min(4320, app_data))
        prefs.set("window_height", value)
        prefs.set("resolution_preset", "custom")
        dpg.set_value(self.status_text, f"Custom height set to {value} (restart required)")
    
    def run(self) -> None:
        """Run the application main loop."""
        self.setup()
        
        # Load default schema
        self._select_schema("IPA Standard")
        
        # Main loop
        while dpg.is_dearpygui_running():
            # Process UI updates from background threads
            self._process_ui_updates()
            
            # Process agent monitor updates
            if self.agent_monitor_panel:
                self.agent_monitor_panel.process_updates()
            
            
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()


def run_app() -> None:
    """Entry point for running the BIDS application."""
    app = BIDSApp()
    app.run()
