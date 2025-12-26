"""
BIDS GUI Application

Main Dear PyGui application for BIDS.
"""

import dearpygui.dearpygui as dpg
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading

from .file_panel import FilePanel
from .schema_editor import SchemaEditor
from .process_panel import ProcessPanel
from .results_panel import ResultsPanel


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
    
    def __init__(self, width: int = 1400, height: int = 900):
        """
        Initialize the BIDS application.
        
        Args:
            width: Window width
            height: Window height
        """
        self.width = width
        self.height = height
        
        # State
        self.current_df = None
        self.current_schema = None
        self.processing = False
        
        # Panels (initialized in setup)
        self.file_panel: Optional[FilePanel] = None
        self.schema_editor: Optional[SchemaEditor] = None
        self.process_panel: Optional[ProcessPanel] = None
        self.results_panel: Optional[ResultsPanel] = None
        
    def setup(self) -> None:
        """Set up Dear PyGui context and windows."""
        dpg.create_context()
        
        # Configure viewport
        dpg.create_viewport(
            title="BIDS - Bioinformatics Data Standardizer",
            width=self.width,
            height=self.height,
            min_width=1000,
            min_height=700
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
        """Set up fonts."""
        # Try to use a Windows system font for better appearance
        import os
        
        # Common Windows font paths
        windows_fonts = [
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'segoeui.ttf'),
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'arial.ttf'),
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'tahoma.ttf'),
        ]
        
        font_path = None
        for path in windows_fonts:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path:
            with dpg.font_registry():
                self.default_font = dpg.add_font(font_path, 16)
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
            
            dpg.add_spacer(height=10)
            
            # Main content area with panels
            with dpg.group(horizontal=True):
                # Left panel - File selection (25% width)
                with dpg.child_window(width=320, height=-60, border=True):
                    self.file_panel = FilePanel(
                        on_file_loaded=self._on_file_loaded
                    )
                    self.file_panel.create()
                
                dpg.add_spacer(width=10)
                
                # Center area
                with dpg.group():
                    # Preview/Process tabs (75% width)
                    with dpg.child_window(width=-1, height=-60, border=True):
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
                                    on_process_complete=self._on_process_complete
                                )
                                self.process_panel.create()
            
            dpg.add_spacer(height=10)
            
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
                    label="API Keys...",
                    callback=self._menu_api_keys
                )
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
            
            dpg.add_spacer(width=-200)
            
            # Quick action buttons
            self.btn_diagnose = dpg.add_button(
                label="Diagnose",
                callback=self._on_diagnose_click,
                enabled=False
            )
            dpg.add_spacer(width=5)
            self.btn_fix = dpg.add_button(
                label="Fix & Export",
                callback=self._on_fix_click,
                enabled=False
            )
            dpg.bind_item_theme(self.btn_fix, self.success_theme)
    
    def _create_status_bar(self) -> None:
        """Create the status bar."""
        with dpg.group(horizontal=True):
            self.status_text = dpg.add_text("Ready", color=self.COLORS["text_dim"])
            
            dpg.add_spacer(width=-300)
            
            self.progress_bar = dpg.add_progress_bar(
                default_value=0.0,
                width=200,
                overlay="Idle"
            )
            
            dpg.add_spacer(width=10)
            
            self.schema_label = dpg.add_text(
                "Schema: None",
                color=self.COLORS["text_dim"]
            )
    
    # Callback handlers
    def _on_file_loaded(self, df, file_path: str) -> None:
        """Handle file loaded callback."""
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
    
    def _on_schema_changed(self, schema) -> None:
        """Handle schema changed callback."""
        self.current_schema = schema
        
        # Update label
        dpg.set_value(self.schema_label, f"Schema: {schema.name}")
        
        # Enable fix button if data loaded
        if self.current_df is not None:
            dpg.configure_item(self.btn_fix, enabled=True)
    
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
        """Handle diagnose button click."""
        if self.current_df is None:
            return
        
        dpg.set_value(self.status_text, "Diagnosing...")
        dpg.set_value(self.progress_bar, 0.3)
        dpg.configure_item(self.progress_bar, overlay="Diagnosing...")
        
        if self.process_panel:
            self.process_panel.run_diagnosis(
                self.current_df,
                self.current_schema
            )
    
    def _on_fix_click(self) -> None:
        """Handle fix button click."""
        if self.current_df is None or self.current_schema is None:
            return
        
        self.processing = True
        dpg.set_value(self.status_text, "Processing...")
        dpg.set_value(self.progress_bar, 0.0)
        dpg.configure_item(self.progress_bar, overlay="Processing...")
        
        if self.process_panel:
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
    
    def _menu_api_keys(self) -> None:
        """API keys menu callback."""
        self._show_api_keys_dialog()
    
    def _menu_preferences(self) -> None:
        """Preferences menu callback."""
        pass  # TODO: Implement preferences
    
    def _menu_documentation(self) -> None:
        """Documentation menu callback."""
        pass  # TODO: Open documentation
    
    def _menu_about(self) -> None:
        """About menu callback."""
        self._show_about_dialog()
    
    def _show_about_dialog(self) -> None:
        """Show the about dialog."""
        with dpg.window(
            label="About BIDS",
            modal=True,
            width=400,
            height=250,
            pos=[500, 300],
            no_resize=True,
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
            dpg.add_text("  - Google Gemini (Diagnostics)")
            dpg.add_text("  - Claude Opus (Fix Generation)")
            dpg.add_spacer(height=20)
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.delete_item("about_dialog"),
                width=-1
            )
    
    def _show_api_keys_dialog(self) -> None:
        """Show API keys configuration dialog."""
        import os
        from dotenv import load_dotenv, set_key
        
        load_dotenv()
        
        with dpg.window(
            label="API Keys Configuration",
            modal=True,
            width=500,
            height=280,
            pos=[450, 250],
            no_resize=True,
            on_close=lambda: dpg.delete_item("api_keys_dialog")
        ) as dialog:
            dpg.set_item_alias(dialog, "api_keys_dialog")
            
            dpg.add_text("Configure your LLM API keys:")
            dpg.add_spacer(height=10)
            
            dpg.add_text("Google Gemini API Key:", color=self.COLORS["text_dim"])
            gemini_input = dpg.add_input_text(
                default_value=os.getenv("GOOGLE_API_KEY", ""),
                width=-1,
                password=True
            )
            
            dpg.add_spacer(height=10)
            
            dpg.add_text("Anthropic API Key:", color=self.COLORS["text_dim"])
            anthropic_input = dpg.add_input_text(
                default_value=os.getenv("ANTHROPIC_API_KEY", ""),
                width=-1,
                password=True
            )
            
            dpg.add_spacer(height=20)
            
            def save_keys():
                env_path = Path(".env")
                if not env_path.exists():
                    env_path.touch()
                
                gemini_key = dpg.get_value(gemini_input)
                anthropic_key = dpg.get_value(anthropic_input)
                
                if gemini_key:
                    set_key(str(env_path), "GOOGLE_API_KEY", gemini_key)
                if anthropic_key:
                    set_key(str(env_path), "ANTHROPIC_API_KEY", anthropic_key)
                
                load_dotenv(override=True)
                dpg.delete_item("api_keys_dialog")
            
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Save",
                    callback=save_keys,
                    width=100
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item("api_keys_dialog"),
                    width=100
                )
    
    def run(self) -> None:
        """Run the application main loop."""
        self.setup()
        
        # Load default schema
        self._select_schema("IPA Standard")
        
        # Main loop
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()


def run_app() -> None:
    """Entry point for running the BIDS application."""
    app = BIDSApp()
    app.run()
