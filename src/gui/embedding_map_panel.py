"""
Embedding Map Panel for BIDS GUI

Interactive 2D visualization of data fingerprints and their connections.
Features:
- t-SNE projection of high-dimensional embeddings
- Interactive nodes showing fingerprint data
- Click-to-view associated scripts from the library
- Visual connections between similar fingerprints
"""

import dearpygui.dearpygui as dpg
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import math
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import threading
import time
import zipfile
import json
import tempfile


def _get_key_code(key_attr_name: str, fallback_code: int) -> int:
    """
    Get a key code from dearpygui, with fallback to GLFW key code.
    
    Args:
        key_attr_name: Name of the key constant (e.g., 'mvKey_Equal')
        fallback_code: GLFW key code to use if constant doesn't exist
        
    Returns:
        Key code to use for the key handler
    """
    try:
        return getattr(dpg, key_attr_name)
    except AttributeError:
        # Fallback to GLFW key codes
        return fallback_code

# VisPy imports for 3D visualization
try:
    from vispy import app, scene
    from vispy.scene import visuals
    VISPY_AVAILABLE = True
except ImportError:
    VISPY_AVAILABLE = False


@dataclass
class MapNode:
    """Represents a node in the embedding map."""
    id: str
    x: float
    y: float
    label: str
    status: str  # SUCCESS, FAILURE, PENDING
    script_content: str
    metadata: Dict[str, Any]
    fingerprint_tokens: List[str]
    connections: List[str]  # IDs of connected nodes
    z: float = 0.0  # Z coordinate for 3D support


class EmbeddingMapPanel:
    """
    Interactive 2D visualization of embeddings and fingerprints.
    
    Features:
    - Canvas with zoomable/pannable view
    - Nodes representing fingerprints colored by status
    - Edges showing similarity connections
    - Click nodes to view associated scripts
    """
    
    # Theme colors
    COLORS = {
        "background": (20, 22, 30, 255),
        "grid": (40, 42, 55, 100),
        "node_success": (46, 204, 113, 255),
        "node_failure": (231, 76, 60, 255),
        "node_pending": (241, 196, 15, 255),
        "node_hover": (100, 149, 237, 255),
        "node_selected": (255, 255, 255, 255),
        "edge": (100, 100, 120, 80),
        "edge_strong": (130, 149, 237, 150),
        "text": (200, 200, 210, 255),
        "text_dim": (120, 130, 140, 255),
        "accent": (100, 149, 237, 255),
        "panel_bg": (30, 32, 42, 255),
    }
    
    NODE_RADIUS = 14  # Base radius for nodes (reduced for better spacing)
    NODE_BOUNDARY_PADDING = 3  # Padding around nodes to prevent overlap
    CONNECTION_THRESHOLD = 0.7  # Similarity threshold for drawing connections
    
    def __init__(self, get_current_data_callback=None, scale: float = 1.0):
        """
        Initialize the embedding map panel.
        
        Args:
            get_current_data_callback: Callback function to get currently loaded DataFrame
            scale: Scaling factor for UI elements
        """
        self.nodes: Dict[str, MapNode] = {}
        self.embeddings: Dict[str, np.ndarray] = {}  # Store embeddings by node ID
        self.selected_node: Optional[str] = None
        self.hovered_node: Optional[str] = None
        self.scale = scale
        
        # View state
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.zoom = 1.0
        
        # Load preferences
        from ..utils.preferences import get_preferences
        prefs = get_preferences()
        
        # Drag sensitivity (load from preferences or use default)
        self.drag_sensitivity = prefs.get("map_drag_sensitivity", 1.0)
        self.drag_sensitivity = max(0.1, min(3.0, self.drag_sensitivity))  # Clamp between 0.1 and 3.0
        
        # Load canvas size from preferences or auto-detect
        
        pref_width = prefs.get("map_canvas_width")
        pref_height = prefs.get("map_canvas_height")
        
        if pref_width and pref_height:
            self.canvas_width = pref_width
            self.canvas_height = pref_height
        else:
            # Auto-detect screen resolution for initial sizing
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
                
                # Calculate canvas size to fit screen with proper aspect ratio
                # Account for UI elements (toolbar, panels, etc.) - use ~60% of available space
                aspect_ratio = screen_width / screen_height
                
                # Calculate based on available space in the embedding map tab
                # Left panel takes ~350px, right panel takes ~350px, so canvas gets the rest
                available_width = screen_width - 700  # Account for side panels
                available_height = screen_height - 200  # Account for toolbar and margins
                
                if aspect_ratio > 1.5:  # Wide screen
                    self.canvas_width = int(available_width * 0.9)
                    self.canvas_height = int(self.canvas_width / aspect_ratio * 0.9)
                else:  # Standard or tall screen
                    self.canvas_height = int(available_height * 0.9)
                    self.canvas_width = int(self.canvas_height * aspect_ratio * 0.9)
                
                # Clamp to reasonable bounds
                self.canvas_width = max(400, min(2000, self.canvas_width))
                self.canvas_height = max(300, min(1500, self.canvas_height))
                
                # Save detected size to preferences
                prefs.set("map_canvas_width", self.canvas_width)
                prefs.set("map_canvas_height", self.canvas_height)
            except:
                # Fallback if detection fails
                self.canvas_width = 800
                self.canvas_height = 500
        
        # UI elements
        self.drawlist_tag: Optional[int] = None
        self.canvas_window_tag: Optional[int] = None
        self.canvas_container_tag: Optional[int] = None
        self.script_text_tag: Optional[int] = None
        self.info_container_tag: Optional[int] = None
        self.stats_text_tag: Optional[int] = None
        self.pse_section_tag: Optional[int] = None
        self.ideal_file_text: Optional[int] = None
        self.import_ideal_button: Optional[int] = None
        self.drag_sensitivity_slider: Optional[int] = None
        
        # Mouse state
        self.is_dragging = False
        self.is_resizing = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.resize_start_width = 0
        self.resize_start_height = 0
        self.resize_start_mouse_x = 0
        self.resize_start_mouse_y = 0
        self.drag_redraw_timer = 0  # Track when to redraw during drag
        self.shift_held = False  # Track Shift key state
        self.last_drag_time = 0  # For throttling redraws
        
        # PSE generation state
        self.get_current_data = get_current_data_callback
        self.pse_generating = False
        self.ideal_df: Optional[pd.DataFrame] = None
        self.ideal_file_path: Optional[str] = None
        
        # Search state
        self.search_input_tag: Optional[int] = None
        self.search_results_tag: Optional[int] = None
        self.search_results: List[str] = []  # List of node IDs matching search
        
        # DND options group (initialized in create)
        self.dnd_options_group: Optional[int] = None
        
        # Canvas dimension mode
        self.dimension_mode: str = "2D"  # "2D" or "3D"
        
        # Vispy view and camera (for 3D mode, optional)
        self.vispy_view: Optional[Any] = None
        self.vispy_camera: Optional[Any] = None
        self.vispy_canvas: Optional[Any] = None
        self.vispy_window: Optional[Any] = None  # Store reference to 3D popup window
    
    def _world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Convert world coordinates to screen coordinates (local drawlist coordinates).
        
        Args:
            world_x: World X coordinate
            world_y: World Y coordinate
            
        Returns:
            Tuple of (screen_x, screen_y) in local drawlist coordinates (top-left origin)
        """
        # Apply zoom and offset transformation
        # World origin (0,0) maps to canvas center, so add center offset
        screen_x = (world_x * self.zoom) + self.view_offset_x + (self.canvas_width / 2)
        screen_y = (world_y * self.zoom) + self.view_offset_y + (self.canvas_height / 2)
        return (screen_x, screen_y)
    
    def create(self) -> None:
        """Create the embedding map panel UI."""
        # Create error theme for red buttons
        with dpg.theme() as self.error_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.COLORS["node_failure"][:3])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (241, 86, 70))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.COLORS["node_failure"][:3])
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))  # White text
        
        # Create blue button theme for export/import buttons
        with dpg.theme() as self.blue_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.COLORS["accent"][:3])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (130, 170, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.COLORS["accent"][:3])
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))  # White text
        
        with dpg.group(horizontal=True):
            # Left side: Canvas
            right_panel_width = int(350 * self.scale)
            with dpg.child_window(width=-right_panel_width, height=-1, border=True) as canvas_window:
                self.canvas_window_tag = canvas_window
                self._create_canvas_area()
            
            # Right side: Info panel
            with dpg.child_window(width=-1, height=-1, border=True):
                self._create_info_panel()
        
        # Create ideal file dialog (hidden initially)
        self._create_ideal_file_dialog()
        
        # Automatically load embeddings from vector store
        self._load_from_vector_store()
        
        # Canvas size will be updated automatically on first draw
    
    def _create_canvas_area(self) -> None:
        """Create the interactive canvas area."""
        # Toolbar - split into two rows for better spacing
        with dpg.group():
            # First row: Main buttons
            with dpg.group(horizontal=True):
                dpg.add_text("Embedding Map", color=self.COLORS["accent"][:3])
                dpg.add_spacer(width=15)
                dpg.add_button(
                    label="Refresh",
                    callback=self._on_refresh_click,
                    width=100
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Reset View",
                    callback=self._reset_view,
                    width=120
                )
                dpg.add_spacer(width=15)
                manage_btn = dpg.add_button(
                    label="Manage Embeddings",
                    callback=self._show_manage_embeddings_dialog,
                    width=170
                )
                dpg.add_spacer(width=15)
                dpg.add_button(
                    label="Open 3D Map",
                    callback=self._open_3d_map,
                    width=130
                )
                dpg.add_spacer(width=-1)  # Push remaining items to the right
                self.stats_text_tag = dpg.add_text(
                    "Nodes: 0 | Connections: 0",
                    color=self.COLORS["text_dim"][:3]
                )
            
            # Second row: Zoom and sensitivity controls
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=0)  # Align with first row
                dpg.add_button(
                    label="+",
                    callback=self._on_zoom_in_click,
                    width=40
                )
                dpg.add_spacer(width=5)
                dpg.add_button(
                    label="-",
                    callback=self._on_zoom_out_click,
                    width=40
                )
                dpg.add_spacer(width=15)
                dpg.add_text("Drag Sensitivity:", color=self.COLORS["text_dim"][:3])
                dpg.add_spacer(width=5)
                self.drag_sensitivity_slider = dpg.add_slider_float(
                    default_value=self.drag_sensitivity,
                    min_value=0.1,
                    max_value=3.0,
                    width=150,
                    format="%.2f",
                    callback=self._on_drag_sensitivity_changed
                )
        
        dpg.add_spacer(height=5)
        
        # Search bar
        with dpg.group(horizontal=True):
            dpg.add_text("Search:", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(width=10)
            self.search_input_tag = dpg.add_input_text(
                hint="Search fingerprints by name...",
                width=-1,
                callback=self._on_search_input,
                default_value=""
            )
        
        # Search results container (initially hidden)
        self.search_results_tag = dpg.add_group()
        dpg.hide_item(self.search_results_tag)
        
        dpg.add_spacer(height=5)
        
        # Legend (using ASCII-compatible characters for better font support)
        with dpg.group(horizontal=True):
            dpg.add_text("Legend:", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(width=10)
            dpg.add_text("[o]", color=self.COLORS["node_success"][:3])
            dpg.add_text("Success", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(width=10)
            dpg.add_text("[o]", color=self.COLORS["node_failure"][:3])
            dpg.add_text("Failure", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(width=10)
            dpg.add_text("[o]", color=self.COLORS["node_pending"][:3])
            dpg.add_text("Pending", color=self.COLORS["text_dim"][:3])
        
        dpg.add_spacer(height=5)
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        # Canvas area - Wrap drawlist in a child window that fills available space
        # Then we'll update the drawlist size to match the container
        with dpg.child_window(width=-1, height=-1, border=False) as canvas_container:
            self.canvas_container_tag = canvas_container
            # Create drawlist with initial size (will be updated to match container)
            with dpg.drawlist(width=self.canvas_width, height=self.canvas_height) as drawlist:
                self.drawlist_tag = drawlist
                dpg.configure_item(self.drawlist_tag, callback=self._on_canvas_click)
        
        # Connect mouse events for pan/zoom
        with dpg.item_handler_registry() as handler_registry:
            # Only bind click handler to left mouse button to avoid conflicts with right-click drag
            # The callback on drawlist will handle left clicks
            dpg.add_item_active_handler(callback=self._on_canvas_drag)
            dpg.add_item_hover_handler(callback=self._on_canvas_hover)
            # Note: Mouse wheel handler not available in DearPyGui, using keyboard shortcuts instead
            # Zoom: +/= to zoom in, -/_ to zoom out
        dpg.bind_item_handler_registry(self.drawlist_tag, handler_registry)
        
        # Add keyboard handlers for zoom (workaround for missing wheel handler)
        # Use helper function to get key codes with fallback to GLFW key codes
        # GLFW key codes: '=' = 61, '-' = 45, Keypad '+' = 334, Keypad '-' = 333
        with dpg.handler_registry():
            dpg.add_key_press_handler(key=_get_key_code('mvKey_Equal', 61), callback=self._on_zoom_in_key)
            dpg.add_key_press_handler(key=_get_key_code('mvKey_KeypadAdd', 334), callback=self._on_zoom_in_key)
            dpg.add_key_press_handler(key=_get_key_code('mvKey_Minus', 45), callback=self._on_zoom_out_key)
            dpg.add_key_press_handler(key=_get_key_code('mvKey_KeypadSubtract', 333), callback=self._on_zoom_out_key)
        
        # Track mouse wheel state for zooming
        self.last_wheel_delta = 0.0
        self.last_wheel_check_time = time.time()
        
        # Initial draw
        self._draw_map()
    
    def _on_canvas_drag(self, sender, app_data) -> None:
        """Handle canvas drag events for panning."""
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
            if not self.is_dragging:
                self.is_dragging = True
                self.last_mouse_x, self.last_mouse_y = dpg.get_mouse_pos()
                self.last_drag_time = time.time()
            else:
                current_x, current_y = dpg.get_mouse_pos()
                dx = current_x - self.last_mouse_x
                dy = current_y - self.last_mouse_y
                
                # Pan the view with sensitivity multiplier
                self.view_offset_x += dx * self.drag_sensitivity
                self.view_offset_y += dy * self.drag_sensitivity
                
                self.last_mouse_x, self.last_mouse_y = current_x, current_y
                
                # Update visuals in real-time with higher frequency for smoother feel
                # Reduced throttling for better responsiveness (120 FPS target)
                current_time = time.time()
                time_since_last = current_time - self.last_drag_time
                
                if time_since_last >= 0.008:  # ~120 FPS for very smooth real-time updates
                    self._update_visuals(skip_size_check=True)
                    self.last_drag_time = current_time
    
    
    def _on_canvas_hover(self, sender, app_data) -> None:
        """Handle canvas hover events and check for mouse wheel zoom."""
        # Check for mouse wheel zoom (polling approach since DearPyGui doesn't have wheel handler)
        try:
            # Try to get mouse wheel delta if available
            # This is a workaround - DearPyGui may not expose this directly
            # We'll use keyboard shortcuts as primary zoom method
            pass
        except Exception:
            pass
        
        if not self.nodes or self.is_dragging:
            return
        
        try:
            mouse_pos = dpg.get_mouse_pos(local=False)
            if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
                return
            
            drawlist_pos = dpg.get_item_pos(self.drawlist_tag)
            if not drawlist_pos:
                return
            
            local_x = mouse_pos[0] - drawlist_pos[0]
            local_y = mouse_pos[1] - drawlist_pos[1]
            
            hovered = self._get_node_at_position_local(local_x, local_y)
            if hovered != self.hovered_node:
                self.hovered_node = hovered
                self._draw_map()
        except Exception:
            pass
    
    def _create_info_panel(self) -> None:
        """Create the information panel on the right."""
        # Generate PSEs section
        dpg.add_text("Generate PSEs", color=self.COLORS["accent"][:3])
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        with dpg.group() as pse_section:
            self.pse_section_tag = pse_section
            
            # Import Ideal File section
            dpg.add_text("Ideal File (0% noise):", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                self.ideal_file_text = dpg.add_text(
                    "No ideal file imported",
                    color=self.COLORS["text_dim"][:3],
                    wrap=200
                )
            dpg.add_spacer(height=3)
            self.import_ideal_button = dpg.add_button(
                label="Import Ideal",
                callback=self._on_import_ideal_click,
                width=-1
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # DND Settings (always visible, default mode)
            dpg.add_text("Dynamic Noise Density (DND) Settings", color=self.COLORS["accent"][:3])
            dpg.add_text(
                "Creates PSEs with varying noise ratios across the specified range",
                color=self.COLORS["text_dim"][:3],
                wrap=250
            )
            dpg.add_spacer(height=10)
            
            # Noise types checkboxes
            dpg.add_text("Noise Types:", color=self.COLORS["text_dim"][:3])
            self.typo_checkbox = dpg.add_checkbox(label="Typo", default_value=True)
            self.semantic_checkbox = dpg.add_checkbox(label="Semantic", default_value=True)
            self.structural_checkbox = dpg.add_checkbox(label="Structural", default_value=True)
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            # Min noise ratio
            dpg.add_text("Min Noise Ratio (%):", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=20)
                self.dnd_min_noise_input = dpg.add_input_float(
                    default_value=0.01,
                    min_value=0.01,
                    max_value=1.0,
                    format="%.2f",
                    width=120
                )
                dpg.add_spacer(width=20)
            
            dpg.add_spacer(height=5)
            
            # Max noise ratio
            dpg.add_text("Max Noise Ratio (%):", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=20)
                self.dnd_max_noise_input = dpg.add_input_float(
                    default_value=0.1,
                    min_value=0.01,
                    max_value=1.0,
                    format="%.2f",
                    width=120
                )
                dpg.add_spacer(width=20)
            
            dpg.add_spacer(height=5)
            
            # Step size
            dpg.add_text("Step Size (%):", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=20)
                self.dnd_step_size_input = dpg.add_input_float(
                    default_value=0.01,
                    min_value=0.01,
                    max_value=0.1,
                    format="%.2f",
                    width=120
                )
                dpg.add_spacer(width=20)
            
            dpg.add_spacer(height=5)
            
            # PSEs per step
            dpg.add_text("PSEs per Step:", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=20)
                self.dnd_pses_per_step_input = dpg.add_slider_int(
                    default_value=10,
                    min_value=1,
                    max_value=50,
                    width=120,
                    callback=self._on_dnd_pses_per_step_changed
                )
                dpg.add_spacer(width=20)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=20)
                dpg.add_text("Value: ", color=self.COLORS["text_dim"][:3])
                self.dnd_pses_per_step_display = dpg.add_text("10", color=self.COLORS["text"][:3])
                dpg.add_spacer(width=20)
            
            dpg.add_spacer(height=10)
            
            # Generate button
            self.generate_pse_button = dpg.add_button(
                label="Generate PSEs",
                callback=self._on_generate_pse_click,
                width=-1
            )
            
            # Status text
            self.pse_status_text = dpg.add_text(
                "",
                color=self.COLORS["text_dim"][:3],
                wrap=300
            )
        
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)
        
        # Node Details section
        dpg.add_text("Node Details", color=self.COLORS["accent"][:3])
        dpg.add_separator()
        dpg.add_spacer(height=10)
        
        # Node info container
        with dpg.child_window(height=200, border=True) as info_container:
            self.info_container_tag = info_container
            dpg.add_text(
                "Click a node to view details",
                color=self.COLORS["text_dim"][:3],
                wrap=300
            )
        
        dpg.add_spacer(height=10)
        dpg.add_text("Associated Script", color=self.COLORS["accent"][:3])
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        # Script content
        with dpg.child_window(height=-1, border=True):
            self.script_text_tag = dpg.add_input_text(
                default_value="# Select a node to view its associated script\n\n"
                              "# Scripts are stored in the vector store\n"
                              "# and associated with data fingerprints.",
                multiline=True,
                readonly=True,
                width=-1,
                height=-1,
                tab_input=True
            )
    
    def _draw_empty_state(self) -> None:
        """Draw the empty state when no data is loaded."""
        dpg.delete_item(self.drawlist_tag, children_only=True)
        
        # Background
        dpg.draw_rectangle(
            pmin=[0, 0],
            pmax=[self.canvas_width, self.canvas_height],
            fill=self.COLORS["background"],
            parent=self.drawlist_tag
        )
        
        # Grid
        self._draw_grid()
        
        # Empty state message
        center_x = self.canvas_width // 2
        center_y = self.canvas_height // 2
        
        dpg.draw_text(
            pos=[center_x - 120, center_y - 30],
            text="No embeddings loaded",
            color=self.COLORS["text_dim"],
            size=18,
            parent=self.drawlist_tag
        )
        dpg.draw_text(
            pos=[center_x - 150, center_y + 10],
            text="Click 'Refresh' to load from vector store",
            color=self.COLORS["text_dim"],
            size=14,
            parent=self.drawlist_tag
        )
    
    def _draw_grid(self) -> None:
        """Draw the background grid."""
        grid_size = int(50 * self.zoom)
        if grid_size < 10:
            grid_size = 10
        
        # Vertical lines
        for x in range(0, self.canvas_width + grid_size, grid_size):
            adjusted_x = (x + int(self.view_offset_x % grid_size))
            if 0 <= adjusted_x <= self.canvas_width:
                dpg.draw_line(
                    p1=[adjusted_x, 0],
                    p2=[adjusted_x, self.canvas_height],
                    color=self.COLORS["grid"],
                    parent=self.drawlist_tag
                )
        
        # Horizontal lines
        for y in range(0, self.canvas_height + grid_size, grid_size):
            adjusted_y = (y + int(self.view_offset_y % grid_size))
            if 0 <= adjusted_y <= self.canvas_height:
                dpg.draw_line(
                    p1=[0, adjusted_y],
                    p2=[self.canvas_width, adjusted_y],
                    color=self.COLORS["grid"],
                    parent=self.drawlist_tag
                )
    
    def _update_canvas_size(self, redraw: bool = False) -> None:
        """Update canvas size to match the container size (auto-resize)."""
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        try:
            # Get the size of the container (child_window) instead of the drawlist
            container_size = None
            if self.canvas_container_tag and dpg.does_item_exist(self.canvas_container_tag):
                container_size = dpg.get_item_rect_size(self.canvas_container_tag)
            elif self.canvas_window_tag and dpg.does_item_exist(self.canvas_window_tag):
                # Fallback to parent window if container doesn't exist
                container_size = dpg.get_item_rect_size(self.canvas_window_tag)
            
            if container_size and len(container_size) >= 2:
                new_width = int(container_size[0])
                new_height = int(container_size[1])
                
                # Validate size - must be positive and reasonable
                if new_width > 0 and new_height > 0 and new_width < 10000 and new_height < 10000:
                    # Only update if size actually changed to avoid unnecessary redraws
                    if new_width != self.canvas_width or new_height != self.canvas_height:
                        old_width = self.canvas_width
                        old_height = self.canvas_height
                        self.canvas_width = new_width
                        self.canvas_height = new_height
                        
                        # Update the drawlist size to match
                        dpg.configure_item(self.drawlist_tag, width=self.canvas_width, height=self.canvas_height)
                        
                        # Save to preferences
                        from ..utils.preferences import get_preferences
                        prefs = get_preferences()
                        prefs.set("map_canvas_width", self.canvas_width)
                        prefs.set("map_canvas_height", self.canvas_height)
                        
                        # Only redraw if explicitly requested (to avoid recursion)
                        if redraw:
                            self._draw_map()
                # If size is invalid, keep current dimensions and log for debugging
                elif new_width == 0 or new_height == 0:
                    print(f"Debug: Container size is invalid: {container_size}, keeping current size: {self.canvas_width}x{self.canvas_height}")
        except Exception as e:
            # If we can't get the size, just continue with current dimensions
            print(f"Debug: Error getting container size: {e}, using current size: {self.canvas_width}x{self.canvas_height}")
    
    def _update_visuals(self, skip_size_check: bool = False) -> None:
        """Update the drawlist visualization."""
        # Update canvas size first to ensure it matches available space
        # Skip during dragging for better performance
        if not skip_size_check:
            self._update_canvas_size(redraw=False)
        self._draw_map(skip_size_check=skip_size_check)
    
    
    def _draw_map(self, skip_size_check: bool = False) -> None:
        """Draw the full embedding map."""
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        # Update canvas size to match available space (auto-resize)
        # Skip during dragging for better performance
        if not skip_size_check:
            self._update_canvas_size()
        
        # Ensure we have valid dimensions before drawing
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            print(f"Debug: Invalid canvas size: {self.canvas_width}x{self.canvas_height}, skipping draw")
            return
        
        # Always delete and redraw for consistency (DearPyGUI handles this efficiently)
        dpg.delete_item(self.drawlist_tag, children_only=True)
        
        # Background
        dpg.draw_rectangle(
            pmin=[0, 0],
            pmax=[self.canvas_width, self.canvas_height],
            fill=self.COLORS["background"],
            parent=self.drawlist_tag
        )
        
        # Grid
        self._draw_grid()
        
        if not self.nodes:
            # Draw empty state
            center_x = self.canvas_width // 2
            center_y = self.canvas_height // 2
            dpg.draw_text(
                pos=[center_x - 120, center_y - 30],
                text="No embeddings loaded",
                color=self.COLORS["text_dim"],
                size=18,
                parent=self.drawlist_tag
            )
            dpg.draw_text(
                pos=[center_x - 150, center_y + 10],
                text="Click 'Refresh' to load from vector store",
                color=self.COLORS["text_dim"],
                size=14,
                parent=self.drawlist_tag
            )
            return
        
        # Draw edges first (so nodes appear on top)
        self._draw_edges()
        
        # Draw nodes
        self._draw_nodes()
    
    def _draw_edges(self) -> None:
        """Draw connections between nodes."""
        drawn_edges = set()
        
        for node_id, node in self.nodes.items():
            for conn_id in node.connections:
                if conn_id in self.nodes:
                    # Avoid drawing duplicate edges
                    edge_key = tuple(sorted([node_id, conn_id]))
                    if edge_key in drawn_edges:
                        continue
                    drawn_edges.add(edge_key)
                    
                    conn_node = self.nodes[conn_id]
                    
                    # Get screen coordinates
                    sx1, sy1 = self._world_to_screen(node.x, node.y)
                    sx2, sy2 = self._world_to_screen(conn_node.x, conn_node.y)
                    
                    # Skip if off-screen
                    if not self._is_visible(sx1, sy1) and not self._is_visible(sx2, sy2):
                        continue
                    
                    # Draw edge with gradient effect
                    edge_color = self.COLORS["edge_strong"] if (
                        node_id == self.selected_node or conn_id == self.selected_node
                    ) else self.COLORS["edge"]
                    
                    dpg.draw_line(
                        p1=[sx1, sy1],
                        p2=[sx2, sy2],
                        color=edge_color,
                        thickness=2 if self.selected_node in [node_id, conn_id] else 1,
                        parent=self.drawlist_tag
                    )
    
    def _draw_nodes(self) -> None:
        """Draw all nodes."""
        # Calculate base radius that scales with zoom for better readability
        base_radius = self.NODE_RADIUS * max(0.5, min(2.0, self.zoom))  # Scale with zoom, clamp between 0.5x and 2x
        
        for node_id, node in self.nodes.items():
            sx, sy = self._world_to_screen(node.x, node.y)
            
            # Skip if off-screen
            if not self._is_visible(sx, sy):
                continue
            
            # Determine color based on status and state
            if node_id == self.selected_node:
                color = self.COLORS["node_selected"]
                radius = base_radius + 4
            elif node_id == self.hovered_node:
                color = self.COLORS["node_hover"]
                radius = base_radius + 2
            else:
                if node.status == "SUCCESS":
                    color = self.COLORS["node_success"]
                elif node.status == "FAILURE":
                    color = self.COLORS["node_failure"]
                else:
                    color = self.COLORS["node_pending"]
                radius = base_radius
            
            # Draw boundary padding (subtle outer circle to prevent visual overlap)
            boundary_radius = radius + self.NODE_BOUNDARY_PADDING
            dpg.draw_circle(
                center=[sx, sy],
                radius=boundary_radius,
                color=(color[0], color[1], color[2], 30),  # Very subtle boundary
                fill=(color[0], color[1], color[2], 15),  # Very light fill
                thickness=1,
                parent=self.drawlist_tag
            )
            
            # Outer glow for selected/hovered
            if node_id in [self.selected_node, self.hovered_node]:
                dpg.draw_circle(
                    center=[sx, sy],
                    radius=radius + 4,
                    color=(color[0], color[1], color[2], 50),
                    fill=(color[0], color[1], color[2], 30),
                    parent=self.drawlist_tag
                )
            
            # Main node circle
            dpg.draw_circle(
                center=[sx, sy],
                radius=radius,
                color=color,
                fill=(color[0], color[1], color[2], 200),
                thickness=2,
                parent=self.drawlist_tag
            )
            
            # Node label (truncated, centered inside circle)
            # Truncate label to fit inside circle (roughly 2*radius characters)
            max_chars = int((radius * 2) / 6)  # ~6 pixels per character
            label = node.label[:max_chars] + "..." if len(node.label) > max_chars else node.label
            
            # Scale text size with zoom (minimum 9, maximum 16)
            text_size = int(max(9, min(16, 11 * max(0.5, min(2.0, self.zoom)))))
            
            # Calculate text width for centering (rough approximation: ~6 pixels per character)
            text_width = len(label) * 6
            text_x = sx - (text_width // 2)
            # Center vertically (text baseline is typically ~70% of font size from top)
            text_y = sy - (text_size * 0.35)
            
            # Use high-contrast white text for readability against colored backgrounds
            text_color = (255, 255, 255, 255)  # White for maximum contrast
            
            dpg.draw_text(
                pos=[text_x, text_y],
                text=label,
                size=text_size,
                color=text_color,
                parent=self.drawlist_tag
            )
    
    def _is_visible(self, sx: float, sy: float) -> bool:
        """Check if a screen position is visible in the canvas."""
        # Include boundary padding in margin calculation
        margin = (self.NODE_RADIUS + self.NODE_BOUNDARY_PADDING) * 2
        return (
            -margin <= sx <= self.canvas_width + margin and
            -margin <= sy <= self.canvas_height + margin
        )
    
    def _on_canvas_click(self, sender, app_data) -> None:
        """Handle canvas click events."""
        # Don't process clicks if right mouse button is down (we're dragging)
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
            return
        
        # Don't process clicks if we're currently dragging (right button still down)
        if self.is_dragging and dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
            return
        
        # Clear dragging state
        if self.is_dragging:
            self.is_dragging = False
            # Final redraw when drag ends (with size check to ensure canvas is correct)
            self._update_visuals(skip_size_check=False)
            # Small delay to avoid processing click immediately after drag
            return
        
        # Stop resizing if clicking
        if self.is_resizing:
            self.is_resizing = False
        
        # Only process clicks if we have a drawlist
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        # Check if mouse is actually over the drawlist
        if not dpg.is_item_hovered(self.drawlist_tag):
            return
        
        # Get mouse position (local to drawlist)
        try:
            # Get mouse position - use exact same method as mouse move handler
            # The mouse move handler works for hover, so we should match it exactly
            mouse_pos = dpg.get_mouse_pos(local=False)
            drawlist_pos = dpg.get_item_pos(self.drawlist_tag)
            
            if not drawlist_pos:
                return
            
            # Convert to local coordinates - EXACTLY match the mouse move handler
            local_x = mouse_pos[0] - drawlist_pos[0]
            local_y = mouse_pos[1] - drawlist_pos[1]
            
            # #region agent log
            import json
            log_path = r"c:\Users\marce\Desktop\BIDS\.cursor\debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"A","location":"embedding_map_panel.py:977","message":"Click coordinates","data":{"mouse_pos":[float(mouse_pos[0]),float(mouse_pos[1])],"drawlist_pos":[float(drawlist_pos[0]),float(drawlist_pos[1])],"local_x":float(local_x),"local_y":float(local_y),"canvas_width":int(self.canvas_width),"canvas_height":int(self.canvas_height),"view_offset_x":float(self.view_offset_x),"view_offset_y":float(self.view_offset_y),"zoom":float(self.zoom)},"timestamp":int(time.time()*1000)}) + "\n")
            # #endregion
            
            # Ensure coordinates are within drawlist bounds (or allow slightly outside for edge cases)
            if local_x < -50 or local_x > self.canvas_width + 50 or \
               local_y < -50 or local_y > self.canvas_height + 50:
                # #region agent log
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"B","location":"embedding_map_panel.py:981","message":"Click out of bounds","data":{"local_x":local_x,"local_y":local_y},"timestamp":int(time.time()*1000)}) + "\n")
                # #endregion
                return
            
            # Find clicked node using local coordinates (same system as _world_to_screen returns)
            clicked_node = self._get_node_at_position_local(local_x, local_y)
            
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"embedding_map_panel.py:986","message":"Click result","data":{"clicked_node":str(clicked_node) if clicked_node else None},"timestamp":int(time.time()*1000)}) + "\n")
            # #endregion
            
            if clicked_node:
                self.selected_node = clicked_node
                self._show_node_details(self.nodes[clicked_node])
                self._update_visuals()  # Update to show selection
            else:
                # Only clear if we had a selection before
                if self.selected_node:
                    self.selected_node = None
                    self._clear_node_details()
                    self._update_visuals()
        except Exception as e:
            print(f"Debug: Error in click handler: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_node_at_position(self, mx: float, my: float) -> Optional[str]:
        """Find a node at the given global mouse position (legacy method)."""
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return None
        
        try:
            canvas_pos = dpg.get_item_pos(self.drawlist_tag)
            if not canvas_pos:
                return None
            
            local_x = mx - canvas_pos[0]
            local_y = my - canvas_pos[1]
            return self._get_node_at_position_local(local_x, local_y)
        except Exception:
            return None
    
    def _get_node_at_position_local(self, local_x: float, local_y: float) -> Optional[str]:
        """Find a node at the given local drawlist coordinates (top-left origin)."""
        if not self.nodes:
            return None
        
        # #region agent log
        import json
        log_path = r"c:\Users\marce\Desktop\BIDS\.cursor\debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"D","location":"embedding_map_panel.py:1019","message":"Node search start","data":{"local_x":float(local_x),"local_y":float(local_y),"num_nodes":len(self.nodes),"base_radius":float(self.NODE_RADIUS),"zoom":float(self.zoom),"scale":float(self.scale)},"timestamp":int(time.time()*1000)}) + "\n")
        # #endregion
        
        # Calculate base radius that scales with zoom (same as drawing)
        # Increase click radius significantly for easier selection (account for scaling)
        base_radius = self.NODE_RADIUS * max(0.5, min(2.0, self.zoom)) * self.scale
        click_radius = base_radius + (15 * self.scale * max(1.0, self.zoom))
        
        # #region agent log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"E","location":"embedding_map_panel.py:1026","message":"Click radius calculation","data":{"base_radius":float(base_radius),"click_radius":float(click_radius)},"timestamp":int(time.time()*1000)}) + "\n")
        # #endregion
        
        # Don't check bounds strictly - nodes might be outside visible area but still clickable
        # Find closest node
        closest_node = None
        closest_distance = float('inf')
        candidates = []
        
        for node_id, node in self.nodes.items():
            # _world_to_screen returns coordinates in local drawlist coordinate system
            sx, sy = self._world_to_screen(node.x, node.y)
            
            # Compare with local click coordinates (same coordinate system)
            # Both are in local drawlist coordinates (top-left origin)
            distance = math.sqrt((local_x - sx) ** 2 + (local_y - sy) ** 2)
            
            # #region agent log
            if len(candidates) < 10:  # Log first 10 nodes
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"F","location":"embedding_map_panel.py:1035","message":"Node distance","data":{"node_id":str(node_id),"world_x":float(node.x),"world_y":float(node.y),"screen_x":float(sx),"screen_y":float(sy),"click_x":float(local_x),"click_y":float(local_y),"distance":float(distance),"click_radius":float(click_radius),"within_radius":bool(distance <= click_radius)},"timestamp":int(time.time()*1000)}) + "\n")
            # #endregion
            
            if distance <= click_radius:
                candidates.append({"node_id": node_id, "distance": distance})
                if distance < closest_distance:
                    closest_distance = distance
                    closest_node = node_id
        
        # #region agent log
        with open(log_path, "a", encoding="utf-8") as f:
            serializable_candidates = [{"node_id": str(c["node_id"]), "distance": float(c["distance"])} for c in candidates[:5]]
            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"G","location":"embedding_map_panel.py:1046","message":"Node search result","data":{"closest_node":str(closest_node) if closest_node else None,"closest_distance":float(closest_distance) if closest_distance != float('inf') else None,"num_candidates":len(candidates),"candidates":serializable_candidates},"timestamp":int(time.time()*1000)}) + "\n")
        # #endregion
        
        return closest_node
    
    def _on_mouse_move(self, sender, app_data) -> None:
        """Handle mouse move for hover effects and resize handle."""
        # Re-enable scrollbars if mouse moves away from map (after zoom)
        if hasattr(self, 'canvas_window_tag') and self.canvas_window_tag:
            if not dpg.is_item_hovered(self.drawlist_tag):
                try:
                    dpg.configure_item(self.canvas_window_tag, no_scrollbar=False)
                except:
                    pass
        
        # Update resize handle position if drawlist size changed
        if hasattr(self, 'resize_handle_tag') and self.resize_handle_tag:
            self._draw_resize_handle()
        
        # Check if dragging ended (right button released)
        if self.is_dragging:
            if not dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
                self.is_dragging = False
                # Final redraw when drag ends
                self._update_visuals()
                return
        
        # Handle node hover (only if mouse is over drawlist)
        if not self.nodes or not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        if not dpg.is_item_hovered(self.drawlist_tag):
            if self.hovered_node:
                self.hovered_node = None
                if not self.is_dragging:
                    self._update_visuals()
            return
        
        try:
            mouse_pos = dpg.get_mouse_pos(local=False)
            drawlist_pos = dpg.get_item_pos(self.drawlist_tag)
            if drawlist_pos:
                local_x = mouse_pos[0] - drawlist_pos[0]
                local_y = mouse_pos[1] - drawlist_pos[1]
                new_hovered = self._get_node_at_position_local(local_x, local_y)
                
                if new_hovered != self.hovered_node:
                    self.hovered_node = new_hovered
                    # Only redraw if not currently dragging (to avoid lag)
                    if not self.is_dragging:
                        self._update_visuals()
        except Exception:
            pass
        
        # Check if we should stop resizing (mouse button released)
        if self.is_resizing:
            # Check if middle button is still pressed
            if not dpg.is_mouse_button_down(dpg.mvMouseButton_Middle):
                self.is_resizing = False
    
    def _on_key_press(self, sender, app_data) -> None:
        """Track key press events."""
        # Check for Shift key (key codes vary, try common ones)
        # 340 = Left Shift, 344 = Right Shift in GLFW
        if app_data in [340, 344, 16]:  # 16 is Windows virtual key code for Shift
            self.shift_held = True
            # Disable scrollbars when Shift is held
            if hasattr(self, 'canvas_window_tag') and self.canvas_window_tag:
                try:
                    dpg.configure_item(self.canvas_window_tag, no_scrollbar=True)
                except:
                    pass
    
    def _on_key_release(self, sender, app_data) -> None:
        """Track key release events."""
        if app_data in [340, 344, 16]:
            self.shift_held = False
            # Re-enable scrollbars when Shift is released
            if hasattr(self, 'canvas_window_tag') and self.canvas_window_tag:
                try:
                    dpg.configure_item(self.canvas_window_tag, no_scrollbar=False)
                except:
                    pass
    
    def _on_zoom_in_click(self, sender, app_data) -> None:
        """Handle zoom in button click."""
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        zoom_factor = 1.1
        new_zoom = self.zoom * zoom_factor
        new_zoom = max(0.1, min(10.0, new_zoom))  # Increased max zoom from 3.0 to 10.0
        self.zoom = new_zoom
        self._update_visuals()
    
    def _on_zoom_out_click(self, sender, app_data) -> None:
        """Handle zoom out button click."""
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        zoom_factor = 0.9
        new_zoom = self.zoom * zoom_factor
        new_zoom = max(0.1, min(10.0, new_zoom))  # Increased max zoom from 3.0 to 10.0
        self.zoom = new_zoom
        self._update_visuals()
    
    def _on_zoom_in_key(self, sender, app_data) -> None:
        """Handle zoom in keyboard shortcut (+/= key)."""
        # Only zoom if mouse is over the drawlist
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        if not dpg.is_item_hovered(self.drawlist_tag):
            return
        
        zoom_factor = 1.1
        new_zoom = self.zoom * zoom_factor
        new_zoom = max(0.1, min(10.0, new_zoom))  # Increased max zoom from 3.0 to 10.0
        self.zoom = new_zoom
        self._update_visuals()
    
    def _on_zoom_out_key(self, sender, app_data) -> None:
        """Handle zoom out keyboard shortcut (-/_ key)."""
        # Only zoom if mouse is over the drawlist
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        if not dpg.is_item_hovered(self.drawlist_tag):
            return
        
        zoom_factor = 0.9
        new_zoom = self.zoom * zoom_factor
        new_zoom = max(0.1, min(10.0, new_zoom))  # Increased max zoom from 3.0 to 10.0
        self.zoom = new_zoom
        self._update_visuals()
    
    def _on_drag_sensitivity_changed(self, sender, app_data) -> None:
        """Handle drag sensitivity slider change."""
        if self.drag_sensitivity_slider and dpg.does_item_exist(self.drag_sensitivity_slider):
            new_sensitivity = dpg.get_value(self.drag_sensitivity_slider)
            self.drag_sensitivity = max(0.1, min(3.0, new_sensitivity))
            
            # Save to preferences
            from ..utils.preferences import get_preferences
            prefs = get_preferences()
            prefs.set("map_drag_sensitivity", self.drag_sensitivity)
    
    def _on_mouse_wheel(self, sender, app_data) -> None:
        """Handle mouse wheel for zooming (when cursor is over map, ignores scrollbars)."""
        # Only process if we have a drawlist
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        # Check if mouse is within drawlist bounds
        try:
            mouse_pos = dpg.get_mouse_pos(local=False)
            drawlist_pos = dpg.get_item_pos(self.drawlist_tag)
            drawlist_size = dpg.get_item_rect_size(self.drawlist_tag)
            
            if drawlist_pos and drawlist_size:
                # Check if mouse is within drawlist rectangle
                is_inside = (drawlist_pos[0] <= mouse_pos[0] <= drawlist_pos[0] + drawlist_size[0] and
                        drawlist_pos[1] <= mouse_pos[1] <= drawlist_pos[1] + drawlist_size[1])
                
                if not is_inside:
                    return  # Mouse not over map, allow normal scrollbar behavior
                
                # Mouse is over map - disable scrollbars temporarily to prevent scrolling
                if hasattr(self, 'canvas_window_tag') and self.canvas_window_tag:
                    try:
                        dpg.configure_item(self.canvas_window_tag, no_scrollbar=True)
                    except:
                        pass
        except Exception:
            # If we can't check bounds, still try to process (fallback)
            pass
        
        # Mouse is over map - disable scrollbars and zoom instead of scrolling
        if hasattr(self, 'canvas_window_tag') and self.canvas_window_tag:
            try:
                dpg.configure_item(self.canvas_window_tag, no_scrollbar=True)
            except:
                pass
        
        # Zoom in/out
        delta = app_data
        zoom_factor = 1.1 if delta > 0 else 0.9
        new_zoom = self.zoom * zoom_factor
        
        # Clamp zoom
        new_zoom = max(0.1, min(10.0, new_zoom))  # Increased max zoom from 3.0 to 10.0
        
        self.zoom = new_zoom
        self._update_visuals()
        
        # Re-enable scrollbars after zoom completes (in mouse move handler)
    
    def _on_mouse_drag(self, sender, app_data) -> None:
        """Handle right-click drag for panning."""
        # Skip if we're resizing
        if self.is_resizing:
            return
        
        # Mark that we're dragging
        if not self.is_dragging:
            self.is_dragging = True
            self.drag_redraw_timer = 0
            self.last_drag_time = time.time()
        
        # app_data is [dx, dy] - use directly without sensitivity multiplier to remove inertia
        dx, dy = app_data[1], app_data[2]
        
        # Update offset directly (1:1 mapping, no momentum/inertia)
        self.view_offset_x += dx
        self.view_offset_y += dy
        
        # Throttle redraws to reduce flashing and lag (redraw at max 30 FPS)
        current_time = time.time()
        time_since_last = current_time - self.last_drag_time
        
        if time_since_last >= 0.033:  # ~30 FPS
            self._update_visuals()
            self.last_drag_time = current_time
    
    def _on_resize_drag(self, sender, app_data) -> None:
        """Handle middle-click drag for resizing the map."""
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        try:
            # Start resize on first middle-click drag
            if not self.is_resizing:
                self.is_resizing = True
                self.resize_start_width = self.canvas_width
                self.resize_start_height = self.canvas_height
                mouse_pos = dpg.get_mouse_pos(local=False)
                self.resize_start_mouse_x = mouse_pos[0]
                self.resize_start_mouse_y = mouse_pos[1]
            
            # Continue resize if already resizing
            if self.is_resizing:
                # app_data is [dx, dy]
                dx, dy = app_data[1], app_data[2]
                
                # Calculate new size based on drag distance
                new_width = max(200, min(2000, self.resize_start_width + dx))
                new_height = max(200, min(2000, self.resize_start_height + dy))
                
                # Update canvas size
                self.canvas_width = int(new_width)
                self.canvas_height = int(new_height)
                
                # Update drawlist size
                dpg.configure_item(self.drawlist_tag, width=self.canvas_width, height=self.canvas_height)
                
                # Redraw
                self._update_visuals()
                self._draw_resize_handle()
        except Exception as e:
            self.is_resizing = False
    
    def _draw_resize_handle(self) -> None:
        """Draw the resize handle in the bottom-right corner."""
        if not hasattr(self, 'resize_handle_tag') or not self.resize_handle_tag:
            return
        
        try:
            dpg.delete_item(self.resize_handle_tag, children_only=True)
            
            # Get drawlist position and size
            drawlist_pos = dpg.get_item_pos(self.drawlist_tag)
            drawlist_size = dpg.get_item_rect_size(self.drawlist_tag)
            
            if not drawlist_pos or not drawlist_size:
                return
            
            # Calculate handle position (bottom-right corner)
            handle_x = drawlist_pos[0] + drawlist_size[0] - 20
            handle_y = drawlist_pos[1] + drawlist_size[1] - 20
            
            # Set handle position
            dpg.set_item_pos(self.resize_handle_tag, [handle_x, handle_y])
            
            # Draw grip lines
            for i in range(3):
                for j in range(3):
                    x = 4 + i * 4
                    y = 4 + j * 4
                    dpg.draw_line(
                        p1=[x, y],
                        p2=[x + 2, y + 2],
                        color=(150, 150, 150, 200),
                        thickness=1,
                        parent=self.resize_handle_tag
                    )
        except Exception:
            pass
    
    def _reset_view(self) -> None:
        """Reset the view to default."""
        if self.vispy_camera:
            if hasattr(self.vispy_camera, 'reset'):
                self.vispy_camera.reset()
            else:
                # Manual reset
                self.vispy_camera.center = (0, 0, 0) if self.dimension_mode == "3D" else (0, 0)
                if hasattr(self.vispy_camera, 'zoom'):
                    self.vispy_camera.zoom = 1.0
            if self.vispy_canvas:
                self.vispy_canvas.update()
        else:
            # Fallback for old code
            self.view_offset_x = 0
            self.view_offset_y = 0
            self.zoom = 1.0
            self._update_visuals()
    
    def _recenter_view(self) -> None:
        """Re-center the view on the map content."""
        if not self.nodes:
            self._reset_view()
            return
        
        if not self.vispy_camera:
            # 2D mode without vispy - use manual centering
            xs = [node.x for node in self.nodes.values()]
            ys = [node.y for node in self.nodes.values()]
            
            if not xs or not ys:
                self._update_visuals()
                return
            
            # Calculate bounding box
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Calculate center
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            # Calculate zoom to fit with 20% padding
            width_span = max_x - min_x if max_x > min_x else 1.0
            height_span = max_y - min_y if max_y > min_y else 1.0
            
            zoom_x = (self.canvas_width * 0.8) / width_span if width_span > 0 else 1.0
            zoom_y = (self.canvas_height * 0.8) / height_span if height_span > 0 else 1.0
            self.zoom = min(zoom_x, zoom_y, 10.0)  # Cap at 10x zoom (increased from 3.0)
            self.zoom = max(self.zoom, 0.1)  # Minimum zoom (decreased from 0.2)
            
            # Center the view
            self.view_offset_x = -center_x * self.zoom + (self.canvas_width / 2)
            self.view_offset_y = -center_y * self.zoom + (self.canvas_height / 2)
            
            self._update_visuals()
            return
        
        # Calculate bounding box of all nodes
        if self.dimension_mode == "3D":
            xs = [node.x for node in self.nodes.values()]
            ys = [node.y for node in self.nodes.values()]
            zs = [node.z for node in self.nodes.values()]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            center_z = sum(zs) / len(zs)
            
            # Calculate bounding rect
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            min_z, max_z = min(zs), max(zs)
            
            # Set camera center
            self.vispy_camera.center = (center_x, center_y, center_z)
            
            # Fit view to bounding box
            width = max(max_x - min_x, 1)
            height = max(max_y - min_y, 1)
            depth = max(max_z - min_z, 1)
            # Use rect for 2D projection, or adjust zoom for 3D
            if hasattr(self.vispy_camera, 'rect'):
                # For 2D cameras, use rect
                margin = max(width, height) * 0.1
                self.vispy_camera.rect = (min_x - margin, min_y - margin, 
                                         width + 2 * margin, height + 2 * margin)
        else:
            xs = [node.x for node in self.nodes.values()]
            ys = [node.y for node in self.nodes.values()]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            
            # Calculate bounding rect
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Set camera center
            self.vispy_camera.center = (center_x, center_y)
            
            # Fit view to bounding box
            width = max(max_x - min_x, 1)
            height = max(max_y - min_y, 1)
            if hasattr(self.vispy_camera, 'rect'):
                margin = max(width, height) * 0.1
                self.vispy_camera.rect = (min_x - margin, min_y - margin, 
                                         width + 2 * margin, height + 2 * margin)
        
        if self.vispy_canvas:
            self.vispy_canvas.update()
    
    def _open_3d_map(self) -> None:
        """Open a VisPy popup window with 3D map visualization (placeholder for now)."""
        if not VISPY_AVAILABLE:
            # Show error message if VisPy is not available
            print("Error: VisPy is not installed. Please install it with: pip install vispy")
            return
        
        # Close existing 3D window if open
        if self.vispy_window is not None:
            try:
                self.vispy_window.close()
            except:
                pass
            self.vispy_window = None
        
        # Create a new VisPy application window
        try:
            # Create canvas
            canvas = scene.SceneCanvas(
                keys='interactive',
                size=(800, 600),
                show=True,
                title='3D Embedding Map - Placeholder'
            )
            
            # Store reference
            self.vispy_window = canvas
            
            # Create 3D view
            view = canvas.central_widget.add_view()
            view.camera = 'turntable'
            view.camera.fov = 45
            view.camera.distance = 10
            
            # Add placeholder text/visualization
            # Create a simple 3D scatter plot placeholder
            n_points = 100
            pos = np.random.rand(n_points, 3) * 10 - 5  # Random positions in 3D space
            
            # Create scatter plot
            scatter = visuals.Markers()
            scatter.set_data(pos, edge_color='white', face_color='cornflowerblue', size=10)
            view.add(scatter)
            
            # Add axes for reference
            axis = visuals.XYZAxis(parent=view.scene)
            
            # Add title text
            title = visuals.Text(
                '3D Embedding Map\n(Placeholder - 3D embedding modeling coming soon)',
                parent=view.scene,
                font_size=16,
                color='white',
                pos=(0, 0, 5)
            )
            
            # VisPy will handle its own event loop
            # The window will run independently
            # Note: VisPy windows run in their own event loop, so this won't block DearPyGui
            
        except Exception as e:
            print(f"Error opening 3D map window: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_search_input(self, sender, app_data) -> None:
        """Handle search input changes."""
        search_text = app_data.strip().lower()
        
        if not search_text:
            # Hide results if search is empty
            if self.search_results_tag:
                dpg.hide_item(self.search_results_tag)
            self.search_results = []
            return
        
        # Search through node labels (error_type)
        matches = []
        for node_id, node in self.nodes.items():
            label = node.label.lower()
            if search_text in label:
                matches.append((node_id, node))
        
        # Sort by relevance (exact match first, then by position)
        matches.sort(key=lambda x: (
            0 if x[1].label.lower().startswith(search_text) else 1,
            x[1].label.lower()
        ))
        
        # Take top 5 results
        self.search_results = [node_id for node_id, _ in matches[:5]]
        
        # Update search results UI
        if self.search_results_tag:
            dpg.delete_item(self.search_results_tag, children_only=True)
            
            if self.search_results:
                dpg.add_text("Results:", color=self.COLORS["text_dim"][:3], parent=self.search_results_tag)
                dpg.add_spacer(height=3, parent=self.search_results_tag)
                
                for i, node_id in enumerate(self.search_results):
                    node = self.nodes[node_id]
                    # Create callback that captures node_id properly
                    def make_callback(nid):
                        return lambda s, a: self._on_search_result_click(nid)
                    
                    with dpg.group(horizontal=True, parent=self.search_results_tag):
                        # Clickable result item
                        result_text = f"{i+1}. {node.label}"
                        dpg.add_button(
                            label=result_text,
                            callback=make_callback(node_id),
                            width=-1
                        )
                
                dpg.show_item(self.search_results_tag)
            else:
                dpg.add_text("No matches found", color=self.COLORS["text_dim"][:3], parent=self.search_results_tag)
                dpg.show_item(self.search_results_tag)
    
    def _on_search_result_click(self, node_id: str) -> None:
        """Handle clicking on a search result."""
        if node_id not in self.nodes:
            return
        
        # Select the node
        self.selected_node = node_id
        self._show_node_details(self.nodes[node_id])
        
        # Center view on the selected node
        node = self.nodes[node_id]
        self.view_offset_x = -node.x * self.zoom
        self.view_offset_y = -node.y * self.zoom
        
        # Hide search results
        if self.search_results_tag:
            dpg.hide_item(self.search_results_tag)
        
        # Clear search input
        if self.search_input_tag:
            dpg.set_value(self.search_input_tag, "")
        
        self._update_visuals()
    
    def _show_node_details(self, node: MapNode) -> None:
        """Display details for a selected node."""
        # Clear info container
        if self.info_container_tag and dpg.does_item_exist(self.info_container_tag):
            dpg.delete_item(self.info_container_tag, children_only=True)
        else:
            return
        
        # Node summary
        with dpg.group(parent=self.info_container_tag):
            # Node label/type
            dpg.add_text("Node Summary", color=self.COLORS["accent"][:3])
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            # Error type / Label
            error_type = node.metadata.get("error_type", node.label)
            dpg.add_text(f"Type: {error_type}", color=self.COLORS["text"][:3], wrap=250)
            
            # Status
            with dpg.group(horizontal=True):
                dpg.add_text("Status:", color=self.COLORS["text_dim"][:3])
                status_color = {
                    "SUCCESS": self.COLORS["node_success"][:3],
                    "FAILURE": self.COLORS["node_failure"][:3],
                    "PENDING": self.COLORS["node_pending"][:3]
                }.get(node.status, self.COLORS["text"][:3])
                dpg.add_text(node.status, color=status_color)
            
            dpg.add_spacer(height=5)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            # Source file
            source_file = node.metadata.get("source_file", "unknown")
            dpg.add_text(f"Source: {source_file}", color=self.COLORS["text_dim"][:3], wrap=250)
            
            # Created date
            created_at = node.metadata.get("created_at", "")
            if created_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                    dpg.add_text(f"Created: {date_str}", color=self.COLORS["text_dim"][:3])
                except:
                    pass
            
            # Execution time if available
            exec_time = node.metadata.get("execution_time", 0.0)
            if exec_time and exec_time > 0:
                dpg.add_text(f"Execution: {exec_time:.2f}s", color=self.COLORS["text_dim"][:3])
            
            # Error message if failed
            error_msg = node.metadata.get("error_message", "")
            if error_msg:
                dpg.add_spacer(height=5)
                dpg.add_text("Error:", color=self.COLORS["node_failure"][:3])
                dpg.add_text(error_msg[:100] + ("..." if len(error_msg) > 100 else ""), 
                            color=self.COLORS["text_dim"][:3], wrap=250)
            
            dpg.add_spacer(height=5)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            # Connections
            dpg.add_text(
                f"Connections: {len(node.connections)}",
                color=self.COLORS["accent"][:3]
            )
            
            # Token count
            token_count = node.metadata.get("token_count", len(node.fingerprint_tokens))
            dpg.add_text(
                f"Fingerprint Tokens: {token_count}",
                color=self.COLORS["text_dim"][:3]
            )
            
            dpg.add_spacer(height=5)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            # Embedding info (show in details panel)
            if self.selected_node and self.selected_node in self.embeddings:
                embedding = self.embeddings[self.selected_node]
                dpg.add_text("Embedding Info:", color=self.COLORS["accent"][:3])
                if isinstance(embedding, np.ndarray):
                    dpg.add_text(f"Shape: {embedding.shape}", color=self.COLORS["text_dim"][:3])
                    dpg.add_text(f"Dimensions: {len(embedding)}", color=self.COLORS["text_dim"][:3])
                    dpg.add_text(f"Data Type: {embedding.dtype}", color=self.COLORS["text_dim"][:3])
                    dpg.add_text(
                        f"Min: {embedding.min():.6f} | Max: {embedding.max():.6f} | Mean: {embedding.mean():.6f}",
                        color=self.COLORS["text_dim"][:3],
                        wrap=250
                    )
                else:
                    dpg.add_text(f"Type: {type(embedding).__name__}", color=self.COLORS["text_dim"][:3])
                    dpg.add_text(f"Length: {len(embedding)}", color=self.COLORS["text_dim"][:3])
                
                dpg.add_spacer(height=5)
                dpg.add_separator()
                dpg.add_spacer(height=5)
            
            # Preview Embedding button (opens detailed dialog)
            dpg.add_button(
                label="Preview Embedding (Detailed)",
                callback=self._on_preview_embedding_click,
                width=-1
            )
            
            dpg.add_spacer(height=5)
            
            # Delete Embedding button
            dpg.add_button(
                label="Delete Embedding",
                callback=self._on_delete_embedding_click,
                width=-1
            )
        
        # Update script display
        dpg.set_value(
            self.script_text_tag,
            node.script_content or "# No script associated with this fingerprint"
        )
    
    def _clear_node_details(self) -> None:
        """Clear the node details panel."""
        dpg.delete_item(self.info_container_tag, children_only=True)
        dpg.add_text(
            "Click a node to view details",
            color=self.COLORS["text_dim"][:3],
            wrap=300,
            parent=self.info_container_tag
        )
        
        dpg.set_value(
            self.script_text_tag,
            "# Select a node to view its associated script"
        )
    
    def _on_preview_embedding_click(self) -> None:
        """Show embedding preview dialog for selected node."""
        if not self.selected_node or self.selected_node not in self.embeddings:
            dpg.set_value(
                self.pse_status_text if hasattr(self, 'pse_status_text') else None,
                "No embedding available for selected node"
            )
            return
        
        embedding = self.embeddings[self.selected_node]
        node = self.nodes[self.selected_node]
        
        # Create or show dialog
        dialog_tag = "embedding_preview_dialog"
        if dpg.does_item_exist(dialog_tag):
            dpg.delete_item(dialog_tag)
        
        with dpg.window(
            label=f"Embedding Preview: {node.label}",
            modal=True,
            tag=dialog_tag,
            width=600,
            height=500,
            pos=[100, 100]
        ):
            dpg.add_text(f"Node ID: {self.selected_node}", color=self.COLORS["accent"][:3])
            dpg.add_text(f"Label: {node.label}", color=self.COLORS["text"][:3])
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            # Embedding info
            if isinstance(embedding, np.ndarray):
                dpg.add_text(f"Shape: {embedding.shape}", color=self.COLORS["text_dim"][:3])
                dpg.add_text(f"Dimensions: {len(embedding)}", color=self.COLORS["text_dim"][:3])
                dpg.add_text(f"Data Type: {embedding.dtype}", color=self.COLORS["text_dim"][:3])
                dpg.add_text(
                    f"Min: {embedding.min():.6f} | Max: {embedding.max():.6f} | Mean: {embedding.mean():.6f}",
                    color=self.COLORS["text_dim"][:3]
                )
            else:
                dpg.add_text(f"Type: {type(embedding).__name__}", color=self.COLORS["text_dim"][:3])
                dpg.add_text(f"Length: {len(embedding)}", color=self.COLORS["text_dim"][:3])
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            # Embedding values (show first 100 values, or all if less)
            dpg.add_text("Embedding Values:", color=self.COLORS["accent"][:3])
            dpg.add_spacer(height=3)
            
            # Convert to list for display
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            
            # Format values for display
            max_display = 100
            if len(embedding_list) > max_display:
                display_values = embedding_list[:max_display]
                truncated = True
            else:
                display_values = embedding_list
                truncated = False
            
            # Create formatted string
            formatted_values = ", ".join([f"{v:.6f}" if isinstance(v, (int, float)) else str(v) for v in display_values])
            if truncated:
                formatted_values += f"\n... ({len(embedding_list) - max_display} more values)"
            
            # Display in scrollable text area
            with dpg.child_window(height=250, border=True):
                dpg.add_input_text(
                    default_value=formatted_values,
                    multiline=True,
                    readonly=True,
                    width=-1,
                    height=-1,
                    tag=f"{dialog_tag}_text"
                )
            
            dpg.add_spacer(height=10)
            
            # Close button
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="Close",
                    callback=lambda: dpg.delete_item(dialog_tag),
                    width=100
                )
    
    def _on_refresh_click(self) -> None:
        """Refresh embeddings from vector store."""
        self._load_from_vector_store()
    
    def _load_from_vector_store(self) -> None:
        """Load embeddings from the vector store and project to 2D."""
        try:
            from ..vector_store import get_vector_store
            
            # Get vector store
            vector_store = get_vector_store("data/vector_store")
            
            # Get all entries
            all_results = vector_store.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not all_results.get("ids"):
                self._update_visuals()
                dpg.set_value(self.stats_text_tag, "Nodes: 0 | Connections: 0")
                return
            
            # Get embeddings and project to 2D
            embeddings = all_results.get("embeddings", [])
            # Handle numpy array or list - check length first to avoid truth value ambiguity
            if len(embeddings) == 0:
                self._update_visuals()
                return
            
            # Convert to list if it's a numpy array
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Project to 2D or 3D using t-SNE or simple PCA fallback
            n_components = 3 if self.dimension_mode == "3D" else 2
            positions = self._project_embeddings(np.array(embeddings), n_components=n_components)
            
            # Build similarity connections
            connections = self._compute_connections(np.array(embeddings))
            
            # Create nodes
            self.nodes = {}
            for i, entry_id in enumerate(all_results["ids"]):
                metadata = all_results["metadatas"][i] if all_results.get("metadatas") else {}
                document = all_results["documents"][i] if all_results.get("documents") else ""
                
                # Scale positions to fit canvas
                x = positions[i, 0] * 200
                y = positions[i, 1] * 200
                z = positions[i, 2] * 200 if n_components == 3 else 0.0
                
                node = MapNode(
                    id=entry_id,
                    x=x,
                    y=y,
                    label=metadata.get("error_type", "unknown")[:15],
                    status=metadata.get("status", "PENDING"),
                    script_content=document,
                    metadata=metadata,
                    fingerprint_tokens=metadata.get("fingerprint_tokens", "").split()[:20],
                    connections=connections.get(i, []),
                    z=z
                )
                
                # Convert connection indices to IDs
                node.connections = [all_results["ids"][j] for j in connections.get(i, []) 
                                   if j < len(all_results["ids"])]
                
                self.nodes[entry_id] = node
                
                # Store embedding for this node
                if i < len(embeddings):
                    embedding_array = np.array(embeddings[i])
                    self.embeddings[entry_id] = embedding_array
            
            # Update stats
            total_connections = sum(len(n.connections) for n in self.nodes.values()) // 2
            dpg.set_value(
                self.stats_text_tag,
                f"Nodes: {len(self.nodes)} | Connections: {total_connections}"
            )
            
            # Auto-fit and center view after loading to ensure map is visible
            # BUT: Only recenter if we're not currently dragging (preserve user's view during import)
            if self.nodes and len(self.nodes) > 0:
                # Only recenter if not dragging and this is initial load (not after import)
                # Check if this is an initial load by checking if we have existing nodes
                should_recenter = not getattr(self, 'is_dragging', False) and len(self.nodes) > 0
                
                if should_recenter:
                    # Calculate optimal zoom to fit all nodes
                    min_x = min(node.x for node in self.nodes.values())
                    max_x = max(node.x for node in self.nodes.values())
                    min_y = min(node.y for node in self.nodes.values())
                    max_y = max(node.y for node in self.nodes.values())
                    
                    # Calculate required zoom to fit nodes with some padding
                    width_span = max_x - min_x
                    height_span = max_y - min_y
                    
                    if width_span > 0 and height_span > 0:
                        # Calculate zoom to fit with 20% padding
                        zoom_x = (self.canvas_width * 0.8) / width_span if width_span > 0 else 1.0
                        zoom_y = (self.canvas_height * 0.8) / height_span if height_span > 0 else 1.0
                        self.zoom = min(zoom_x, zoom_y, 3.0)  # Cap at 3x zoom
                        self.zoom = max(self.zoom, 0.2)  # Minimum zoom
                    
                    # Center the view
                    self._recenter_view()
                else:
                    # Preserve current view during import/reload
                    print(f"[DEBUG] Preserving view during load (is_dragging={getattr(self, 'is_dragging', False)})")
            
            # Draw the map
            self._update_visuals()
            
        except Exception as e:
            print(f"Error loading from vector store: {e}")
            self._update_visuals()
            dpg.set_value(self.stats_text_tag, f"Error: {str(e)[:30]}")
    
    def _project_embeddings(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Project high-dimensional embeddings to 2D or 3D.
        
        Args:
            embeddings: High-dimensional embedding vectors
            n_components: Number of dimensions (2 for 2D, 3 for 3D)
        
        Returns:
            Projected positions with shape (N, n_components)
        """
        if len(embeddings) == 0:
            return np.array([]).reshape(0, n_components)
        
        if len(embeddings) == 1:
            return np.zeros((1, n_components))
        
        try:
            # Use t-SNE for small datasets, PCA for larger
            if len(embeddings) <= 50:
                from sklearn.manifold import TSNE
                perplexity = min(30, max(5, len(embeddings) - 1))
                tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
                positions = tsne.fit_transform(embeddings)
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_components, random_state=42)
                positions = pca.fit_transform(embeddings)
            
            # Normalize to [-1, 1]
            if positions.max() != positions.min():
                positions = (positions - positions.min()) / (positions.max() - positions.min()) * 2 - 1
            
            return positions
            
        except Exception as e:
            print(f"Projection error: {e}")
            # Fallback: random positions
            return np.random.randn(len(embeddings), n_components)
    
    def _compute_connections(self, embeddings: np.ndarray) -> Dict[int, List[int]]:
        """Compute similarity connections between embeddings."""
        connections = {}
        
        if len(embeddings) < 2:
            return connections
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Compute pairwise similarities
            similarities = cosine_similarity(embeddings)
            
            # Find connections above threshold
            for i in range(len(embeddings)):
                connections[i] = []
                for j in range(i + 1, len(embeddings)):
                    if similarities[i, j] >= self.CONNECTION_THRESHOLD:
                        connections[i].append(j)
                        if j not in connections:
                            connections[j] = []
                        connections[j].append(i)
            
        except Exception as e:
            print(f"Connection computation error: {e}")
        
        return connections
    
    def add_fingerprint(
        self,
        fingerprint_id: str,
        embedding: np.ndarray,
        status: str,
        script_content: str,
        metadata: Dict[str, Any],
        tokens: List[str]
    ) -> None:
        """
        Add a new fingerprint to the map dynamically.
        
        Args:
            fingerprint_id: Unique ID
            embedding: High-dimensional embedding vector
            status: SUCCESS, FAILURE, or PENDING
            script_content: Associated fix script
            metadata: Additional metadata
            tokens: Fingerprint tokens
        """
        # For dynamic adding, use a simple random position relative to existing nodes
        if self.nodes:
            avg_x = sum(n.x for n in self.nodes.values()) / len(self.nodes)
            avg_y = sum(n.y for n in self.nodes.values()) / len(self.nodes)
            x = avg_x + np.random.randn() * 50
            y = avg_y + np.random.randn() * 50
        else:
            x, y = 0, 0
        
        node = MapNode(
            id=fingerprint_id,
            x=x,
            y=y,
            label=metadata.get("error_type", "unknown")[:15],
            status=status,
            script_content=script_content,
            metadata=metadata,
            fingerprint_tokens=tokens[:20],
            connections=[]
        )
        
        self.nodes[fingerprint_id] = node
        
        # Store embedding
        self.embeddings[fingerprint_id] = embedding
        
        # Update stats and redraw
        total_connections = sum(len(n.connections) for n in self.nodes.values()) // 2
        dpg.set_value(
            self.stats_text_tag,
            f"Nodes: {len(self.nodes)} | Connections: {total_connections}"
        )
        
        self._update_visuals()
    
    def update_node_position(self, node_id: str, x: float, y: float, z: Optional[float] = None) -> None:
        """Update the position of a node."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.x = x
        node.y = y
        if z is not None:
            node.z = z
        
        # Update visuals
        self._update_visuals()
    
    def add_node(self, node: MapNode) -> None:
        """Add a new node to the map."""
        self.nodes[node.id] = node
        
        # Update stats
        total_connections = sum(len(n.connections) for n in self.nodes.values()) // 2
        dpg.set_value(
            self.stats_text_tag,
            f"Nodes: {len(self.nodes)} | Connections: {total_connections}"
        )
        
        # Update visuals
        self._update_visuals()
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the map."""
        if node_id not in self.nodes:
            return
        
        # Remove node
        del self.nodes[node_id]
        
        # Remove from embeddings
        if node_id in self.embeddings:
            del self.embeddings[node_id]
        
        # Remove connections to this node from other nodes
        for node in self.nodes.values():
            if node_id in node.connections:
                node.connections.remove(node_id)
        
        # Clear selection if this node was selected
        if self.selected_node == node_id:
            self.selected_node = None
            self._clear_node_details()
        
        # Clear hover if this node was hovered
        if self.hovered_node == node_id:
            self.hovered_node = None
        
        # Update stats
        total_connections = sum(len(n.connections) for n in self.nodes.values()) // 2
        dpg.set_value(
            self.stats_text_tag,
            f"Nodes: {len(self.nodes)} | Connections: {total_connections}"
        )
        
        # Update visuals
        self._update_visuals()
    
    def update_connections(self, node_id: str, new_connections: List[str]) -> None:
        """Update the connections for a node."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.connections = new_connections
        
        # Update stats
        total_connections = sum(len(n.connections) for n in self.nodes.values()) // 2
        dpg.set_value(
            self.stats_text_tag,
            f"Nodes: {len(self.nodes)} | Connections: {total_connections}"
        )
        
        # Update visuals
        self._update_visuals()
    
    
    def _on_dnd_pses_per_step_changed(self, sender, app_data) -> None:
        """Update the display text when the DND PSEs per step slider changes."""
        if hasattr(self, 'dnd_pses_per_step_display'):
            dpg.set_value(self.dnd_pses_per_step_display, str(int(app_data)))
    
    def _on_generate_pse_click(self) -> None:
        """Handle Generate PSEs button click."""
        if self.pse_generating:
            return
        
        # Use ideal file if available, otherwise fall back to current data
        if self.ideal_df is not None:
            current_df = self.ideal_df
        else:
            # Get current data as fallback
            if not self.get_current_data:
                dpg.set_value(self.pse_status_text, "Error: No data access callback configured")
                return
            
            current_df = self.get_current_data()
            if current_df is None:
                dpg.set_value(self.pse_status_text, "Error: No data loaded. Please import an ideal file or load a file first.")
                return
        
        # Always use DND mode - get DND parameters from UI
        min_noise = dpg.get_value(self.dnd_min_noise_input) if hasattr(self, 'dnd_min_noise_input') else 0.01
        max_noise = dpg.get_value(self.dnd_max_noise_input) if hasattr(self, 'dnd_max_noise_input') else 0.1
        step_size = dpg.get_value(self.dnd_step_size_input) if hasattr(self, 'dnd_step_size_input') else 0.01
        pses_per_step = int(dpg.get_value(self.dnd_pses_per_step_input)) if hasattr(self, 'dnd_pses_per_step_input') else 10
        
        # Calculate total PSEs
        noise_ratios = []
        current_noise = min_noise
        while current_noise <= max_noise:
            noise_ratios.append(current_noise)
            current_noise += step_size
        num_pses = len(noise_ratios) * pses_per_step
        noise_ratio = max_noise  # Used as fallback parameter
        
        typo_enabled = dpg.get_value(self.typo_checkbox)
        semantic_enabled = dpg.get_value(self.semantic_checkbox)
        structural_enabled = dpg.get_value(self.structural_checkbox)
        
        # Validate noise types
        noise_types = []
        if typo_enabled:
            noise_types.append("typo")
        if semantic_enabled:
            noise_types.append("semantic")
        if structural_enabled:
            noise_types.append("structural")
        
        if not noise_types:
            dpg.set_value(self.pse_status_text, "Error: At least one noise type must be selected")
            return
        
        # DND always uses uniform distribution, no target columns
        target_columns = None
        
        # Disable button and show status
        self.pse_generating = True
        dpg.configure_item(self.generate_pse_button, enabled=False)
        dpg.set_value(self.pse_status_text, f"Generating {num_pses} PSEs with DND...")
        
        # Run generation in background thread (always DND mode)
        import threading
        thread = threading.Thread(
            target=self._generate_pse_thread,
            args=(current_df, num_pses, noise_ratio, noise_types, target_columns, True, min_noise, max_noise, step_size, pses_per_step),
            daemon=True
        )
        thread.start()
    
    def _generate_pse_thread(
        self,
        df,
        num_pses: int,
        noise_ratio: float,
        noise_types: List[str],
        target_columns: Optional[List[str]],
        dynamic_noise: bool,
        dnd_min_noise: Optional[float] = None,
        dnd_max_noise: Optional[float] = None,
        dnd_step_size: Optional[float] = None,
        dnd_pses_per_step: Optional[int] = None
    ) -> None:
        """Background thread for generating PSEs."""
        try:
            from ..noise_generator import NoiseGenerator, NoiseConfig, DistributionMode, NoiseType
            from ..vectorizer import NoiseVectorizer
            from ..vector_store import get_vector_store, ScriptStatus
            
            # Initialize components
            vector_store = get_vector_store("data/vector_store")
            vectorizer = NoiseVectorizer()
            
            # Always use DND mode
            if dnd_min_noise is not None and dnd_max_noise is not None and dnd_step_size is not None and dnd_pses_per_step is not None:
                # Generate noise ratios from min to max in steps
                noise_ratios = []
                current_noise = dnd_min_noise
                while current_noise <= dnd_max_noise:
                    noise_ratios.append(current_noise)
                    current_noise += dnd_step_size
                
                # Distribute PSEs across noise ratios
                pses_per_ratio = dnd_pses_per_step
                
                # Create list of (noise_ratio, count) pairs
                pse_configs = []
                for nr in noise_ratios:
                    pse_configs.append((nr, pses_per_ratio))
            else:
                # Fallback: use single noise ratio (should not happen in normal operation)
                pse_configs = [(noise_ratio, num_pses)]
            
            # Generate PSEs
            generated_count = 0
            fingerprints_to_fit = []
            reports = []
            pse_configs_flat = []
            
            # Flatten configs for processing
            for noise_ratio_val, count in pse_configs:
                for _ in range(count):
                    pse_configs_flat.append(noise_ratio_val)
            
            for i, current_noise_ratio in enumerate(pse_configs_flat):
                # Create noise config with current noise ratio (always uniform for DND)
                noise_type_enums = [NoiseType(nt) for nt in noise_types]
                
                config = NoiseConfig(
                    noise_ratio=current_noise_ratio,
                    distribution_mode=DistributionMode.UNIFORM,
                    target_columns=[],
                    noise_types=noise_type_enums
                )
                
                generator = NoiseGenerator(config)
                
                # Generate noisy DataFrame
                noisy_df, report = generator.inject_noise(df)
                reports.append((report, current_noise_ratio))
                
                # Create fingerprint
                fingerprint = vectorizer.create_fingerprint(noisy_df)
                fingerprints_to_fit.append(fingerprint)
            
            # Fit vectorizer on all fingerprints at once (more efficient)
            if len(fingerprints_to_fit) > 0:
                vectorizer.fit(fingerprints_to_fit)
                # Vectorize all fingerprints
                for fp in fingerprints_to_fit:
                    vectorizer.transform(fp)
            
            # Now store all fingerprints
            for i, (fingerprint, report_data) in enumerate(zip(fingerprints_to_fit, reports)):
                report, current_noise_ratio = report_data if isinstance(report_data, tuple) else (report_data, noise_ratio)
                
                # Create a placeholder script (since these are synthetic test data)
                script_content = f"""# PSE Generated Script
# This is a synthetic Pre-Made Standard Error (PSE) generated for testing
# Original data shape: {df.shape}
# Noise ratio: {current_noise_ratio:.2%}
# Noise types: {', '.join(noise_types)}
# Cells modified: {report.cells_modified}

import pandas as pd
import numpy as np

def fix_data(df):
    '''
    Placeholder fix script for PSE.
    This script would be replaced by actual fix logic during processing.
    '''
    # TODO: Implement actual fix logic based on error patterns
    return df.copy()

# Apply fix
fixed_df = fix_data(df)
"""
                
                # Determine error type label (always DND)
                error_type_label = f"PSE_DND_{noise_types[0] if noise_types else 'mixed'}_{current_noise_ratio:.0%}"
                
                # Add to vector store
                entry_id = vector_store.add_entry(
                    fingerprint=fingerprint,
                    script_content=script_content,
                    status=ScriptStatus.PENDING,
                    error_type=error_type_label,
                    source_file="pse_generated",
                    execution_time=None,
                    error_message=None
                )
                
                generated_count += 1
                
                # Update status periodically
                total_pses = len(pse_configs_flat)
                if i % max(1, total_pses // 10) == 0 or i == total_pses - 1:
                    dpg.set_value(
                        self.pse_status_text,
                        f"Generated {generated_count}/{total_pses} PSEs..."
                    )
            
            # Update UI
            dpg.set_value(
                self.pse_status_text,
                f"Successfully generated {generated_count} PSEs with dynamic noise density! Click 'Refresh' to see them on the map."
            )
            dpg.configure_item(self.generate_pse_button, enabled=True)
            self.pse_generating = False
            
        except Exception as e:
            import traceback
            error_msg = f"Error generating PSEs: {str(e)}"
            print(f"PSE Generation Error: {traceback.format_exc()}")
            dpg.set_value(self.pse_status_text, error_msg)
            dpg.configure_item(self.generate_pse_button, enabled=True)
            self.pse_generating = False
    
    def _create_ideal_file_dialog(self) -> None:
        """Create the file dialog for importing ideal file."""
        inputs_dir = Path("inputs")
        inputs_dir.mkdir(exist_ok=True)
        
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_ideal_file_dialog_ok,
            cancel_callback=self._on_ideal_file_dialog_cancel,
            width=700,
            height=400,
            modal=True,
            tag="ideal_file_dialog"
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
    
    def _on_import_ideal_click(self) -> None:
        """Handle Import Ideal button click."""
        # Create dialog if it doesn't exist
        if not dpg.does_item_exist("ideal_file_dialog"):
            self._create_ideal_file_dialog()
        
        # Show the dialog
        inputs_dir = Path("inputs").absolute()
        inputs_dir.mkdir(exist_ok=True)
        
        try:
            dpg.configure_item("ideal_file_dialog", default_path=str(inputs_dir))
        except:
            pass
        
        dpg.show_item("ideal_file_dialog")
    
    def _on_ideal_file_dialog_ok(self, sender, app_data) -> None:
        """Handle ideal file dialog OK callback."""
        selections = app_data.get("selections", {})
        
        if not selections:
            return
        
        # Get the first selected file
        file_path = list(selections.values())[0]
        self._load_ideal_file(file_path)
    
    def _on_ideal_file_dialog_cancel(self, sender, app_data) -> None:
        """Handle ideal file dialog cancel."""
        pass
    
    def _load_ideal_file(self, file_path: str) -> None:
        """Load the ideal file in a background thread."""
        # Update UI to show loading
        dpg.set_value(self.ideal_file_text, f"Loading {Path(file_path).name}...")
        dpg.configure_item(self.import_ideal_button, enabled=False)
        
        # Load in background thread
        thread = threading.Thread(
            target=self._load_ideal_file_thread,
            args=(file_path,),
            daemon=True
        )
        thread.start()
    
    def _load_ideal_file_thread(self, file_path: str) -> None:
        """Background thread for loading ideal file."""
        try:
            # Use the same ingestor as FilePanel
            from ..ingestion import SmartIngestor
            ingestor = SmartIngestor()
            result = ingestor.ingest(file_path)
            
            if result.dataframes:
                df = result.dataframes[0]
                
                # Pre-process: Remove columns that are all NULL (same as FilePanel)
                original_cols = len(df.columns)
                df = df.dropna(axis=1, how='all')
                removed_cols = original_cols - len(df.columns)
                
                self.ideal_df = df
                self.ideal_file_path = file_path
                
                # Update UI in main thread
                file_name = Path(file_path).name
                if removed_cols > 0:
                    dpg.set_value(
                        self.ideal_file_text,
                        f"{file_name}\n(Removed {removed_cols} empty column(s))"
                    )
                else:
                    dpg.set_value(self.ideal_file_text, file_name)
                
                dpg.configure_item(self.import_ideal_button, enabled=True)
                dpg.set_value(
                    self.pse_status_text,
                    f"Ideal file loaded: {file_name} ({len(df)} rows, {len(df.columns)} columns)"
                )
            else:
                dpg.set_value(
                    self.ideal_file_text,
                    f"Failed to load: {', '.join(result.errors)}"
                )
                dpg.configure_item(self.import_ideal_button, enabled=True)
                dpg.set_value(
                    self.pse_status_text,
                    f"Error loading ideal file: {', '.join(result.errors)}"
                )
                
        except Exception as e:
            dpg.set_value(
                self.ideal_file_text,
                f"Error: {str(e)[:50]}"
            )
            dpg.configure_item(self.import_ideal_button, enabled=True)
            dpg.set_value(
                self.pse_status_text,
                f"Error loading ideal file: {str(e)}"
            )
    
    def _on_manage_embeddings_click(self) -> None:
        """Open the embedding management dialog."""
        dialog_tag = "embedding_management_dialog"
        if dpg.does_item_exist(dialog_tag):
            dpg.delete_item(dialog_tag)
        
        with dpg.window(
            label="Manage Embeddings",
            modal=True,
            tag=dialog_tag,
            width=500,
            height=400,
            pos=[200, 200]
        ):
            dpg.add_text("Embedding Management", color=self.COLORS["accent"][:3])
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Export current map
            dpg.add_text("Export & Import:", color=self.COLORS["text"][:3])
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Export Current Map",
                    callback=self._on_export_map_click,
                    width=200
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Import Map",
                    callback=self._on_import_map_click,
                    width=200
                )
            
            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Delete operations
            dpg.add_text("Delete Operations:", color=self.COLORS["text"][:3])
            dpg.add_spacer(height=5)
            
            # Delete selected embedding
            delete_selected_enabled = self.selected_node is not None
            selected_label = ""
            if self.selected_node and self.selected_node in self.nodes:
                node_label = self.nodes[self.selected_node].label[:15]
                selected_label = f" ({node_label}...)"
            else:
                selected_label = " (None selected)"
            
            self.delete_selected_button = dpg.add_button(
                label=f"Delete Selected Embedding{selected_label}",
                callback=self._on_delete_selected_embedding_click,
                width=-1,
                enabled=delete_selected_enabled
            )
            
            dpg.add_spacer(height=10)
            
            # Delete all embeddings
            dpg.add_text("[!] Warning: This will delete ALL embeddings!", color=self.COLORS["node_failure"][:3])
            dpg.add_spacer(height=5)
            self.delete_all_button = dpg.add_button(
                label="Delete All Embeddings",
                callback=self._on_delete_all_embeddings_click,
                width=-1
            )
            
            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Status text
            self.management_status_text = dpg.add_text(
                "",
                color=self.COLORS["text_dim"][:3],
                wrap=450
            )
            
            dpg.add_spacer(height=10)
            
            # Close button
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="Close",
                    callback=lambda: dpg.delete_item(dialog_tag),
                    width=100
                )
    
    def _on_export_map_click(self) -> None:
        """Export current map to JSON file."""
        try:
            from ..vector_store import get_vector_store
            
            # Create file dialog for saving
            export_dialog_tag = "export_map_dialog"
            if dpg.does_item_exist(export_dialog_tag):
                dpg.delete_item(export_dialog_tag)
            
            with dpg.file_dialog(
                directory_selector=False,
                show=True,
                callback=self._on_export_map_dialog_ok,
                cancel_callback=lambda s, a: dpg.delete_item(export_dialog_tag),
                width=700,
                height=400,
                modal=True,
                tag=export_dialog_tag,
                default_filename="embedding_map_export.json"
            ):
                dpg.add_file_extension(".*", color=(255, 255, 255))
                dpg.add_file_extension(".json", color=(0, 255, 0))
                
        except Exception as e:
            if hasattr(self, 'management_status_text'):
                dpg.set_value(
                    self.management_status_text,
                    f"Error opening export dialog: {str(e)}"
                )
    
    def _on_export_map_dialog_ok(self, sender, app_data) -> None:
        """Handle export map dialog OK."""
        try:
            selections = app_data.get("selections", {})
            if not selections:
                return
            
            file_path = list(selections.values())[0]
            
            from ..vector_store import get_vector_store
            vector_store = get_vector_store("data/vector_store")
            
            if vector_store.export_to_json(file_path):
                if hasattr(self, 'management_status_text'):
                    dpg.set_value(
                        self.management_status_text,
                        f"Successfully exported {len(self.nodes)} embeddings to {Path(file_path).name}"
                    )
            else:
                if hasattr(self, 'management_status_text'):
                    dpg.set_value(
                        self.management_status_text,
                        "Error: Failed to export embeddings"
                    )
            
            # Close export dialog
            if dpg.does_item_exist("export_map_dialog"):
                dpg.delete_item("export_map_dialog")
                
        except Exception as e:
            if hasattr(self, 'management_status_text'):
                dpg.set_value(
                    self.management_status_text,
                    f"Error exporting: {str(e)}"
                )
    
    def _on_import_map_click(self) -> None:
        """Import map from JSON file."""
        try:
            # Create file dialog for loading
            import_dialog_tag = "import_map_dialog"
            if dpg.does_item_exist(import_dialog_tag):
                dpg.delete_item(import_dialog_tag)
            
            inputs_dir = Path("inputs")
            inputs_dir.mkdir(exist_ok=True)
            
            with dpg.file_dialog(
                directory_selector=False,
                show=True,
                callback=self._on_import_map_dialog_ok,
                cancel_callback=lambda s, a: dpg.delete_item(import_dialog_tag),
                width=700,
                height=400,
                modal=True,
                tag=import_dialog_tag
            ):
                dpg.add_file_extension(".*", color=(255, 255, 255))
                dpg.add_file_extension(".json", color=(0, 255, 0))
            
            try:
                dpg.configure_item(import_dialog_tag, default_path=str(inputs_dir.absolute()))
            except:
                pass
                
        except Exception as e:
            if hasattr(self, 'management_status_text'):
                dpg.set_value(
                    self.management_status_text,
                    f"Error opening import dialog: {str(e)}"
                )
    
    def _on_import_map_dialog_ok(self, sender, app_data) -> None:
        """Handle import map dialog OK."""
        try:
            selections = app_data.get("selections", {})
            if not selections:
                return
            
            file_path = list(selections.values())[0]
            
            from ..vector_store import get_vector_store
            vector_store = get_vector_store("data/vector_store")
            
            count = vector_store.import_from_json(file_path)
            
            if count > 0:
                if hasattr(self, 'management_status_text'):
                    dpg.set_value(
                        self.management_status_text,
                        f"Successfully imported {count} embeddings from {Path(file_path).name}. Click 'Refresh' to see them."
                    )
                # Refresh the map
                self._load_from_vector_store()
            else:
                if hasattr(self, 'management_status_text'):
                    dpg.set_value(
                        self.management_status_text,
                        "Error: Failed to import embeddings or file was empty"
                    )
            
            # Close import dialog
            if dpg.does_item_exist("import_map_dialog"):
                dpg.delete_item("import_map_dialog")
                
        except Exception as e:
            if hasattr(self, 'management_status_text'):
                dpg.set_value(
                    self.management_status_text,
                    f"Error importing: {str(e)}"
                )
    
    def _on_delete_selected_embedding_click(self) -> None:
        """Delete the currently selected embedding."""
        if not self.selected_node:
            if hasattr(self, 'management_status_text'):
                dpg.set_value(
                    self.management_status_text,
                    "No embedding selected. Please select a node first."
                )
            return
        
        # Confirm deletion
        confirm_tag = "delete_selected_confirm"
        if dpg.does_item_exist(confirm_tag):
            dpg.delete_item(confirm_tag)
        
        node_label = self.nodes[self.selected_node].label if self.selected_node in self.nodes else "selected"
        
        with dpg.window(
            label="Confirm Deletion",
            modal=True,
            tag=confirm_tag,
            width=400,
            height=200,
            pos=[300, 300]
        ):
            dpg.add_text(f"Are you sure you want to delete this embedding?", color=self.COLORS["text"][:3])
            dpg.add_text(f"Label: {node_label}", color=self.COLORS["text_dim"][:3])
            dpg.add_text(f"ID: {self.selected_node[:20]}...", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(height=15)
            
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item(confirm_tag),
                    width=80
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Delete",
                    callback=lambda: self._confirm_delete_selected(confirm_tag),
                    width=80
                )
    
    def _confirm_delete_selected(self, confirm_dialog_tag: str) -> None:
        """Confirm and execute deletion of selected embedding."""
        try:
            if not self.selected_node:
                return
            
            from ..vector_store import get_vector_store
            vector_store = get_vector_store("data/vector_store")
            
            node_id = self.selected_node
            
            delete_result = vector_store.delete_entry(node_id)
            
            if delete_result:
                # Remove from local state
                if node_id in self.nodes:
                    del self.nodes[node_id]
                if node_id in self.embeddings:
                    del self.embeddings[node_id]
                
                # Clear selection
                self.selected_node = None
                self._clear_node_details()
                
                # Refresh map
                self._load_from_vector_store()
                
                if hasattr(self, 'management_status_text'):
                    dpg.set_value(
                        self.management_status_text,
                        f"Successfully deleted embedding: {node_id[:20]}..."
                    )
            else:
                if hasattr(self, 'management_status_text'):
                    dpg.set_value(
                        self.management_status_text,
                        "Error: Failed to delete embedding"
                    )
            
            # Close confirm dialog
            if dpg.does_item_exist(confirm_dialog_tag):
                dpg.delete_item(confirm_dialog_tag)
                
        except Exception as e:
            if hasattr(self, 'management_status_text'):
                dpg.set_value(
                    self.management_status_text,
                    f"Error deleting embedding: {str(e)}"
                )
            if dpg.does_item_exist(confirm_dialog_tag):
                dpg.delete_item(confirm_dialog_tag)
    
    def _on_delete_embedding_click(self) -> None:
        """Delete the currently selected embedding (from node details panel)."""
        if not self.selected_node:
            return
        
        # Use the same confirmation logic
        self._on_delete_selected_embedding_click()
    
    def _on_delete_all_embeddings_click(self) -> None:
        """Delete all embeddings after confirmation."""
        # Confirm deletion
        confirm_tag = "delete_all_confirm"
        if dpg.does_item_exist(confirm_tag):
            dpg.delete_item(confirm_tag)
        
        node_count = len(self.nodes)
        
        with dpg.window(
            label="Confirm Deletion",
            modal=True,
            tag=confirm_tag,
            width=450,
            height=250,
            pos=[300, 300]
        ):
            dpg.add_text("[!] WARNING: This will delete ALL embeddings!", color=self.COLORS["node_failure"][:3])
            dpg.add_spacer(height=5)
            dpg.add_text(f"This action cannot be undone.", color=self.COLORS["text"][:3])
            dpg.add_text(f"Total embeddings to delete: {node_count}", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(height=15)
            
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item(confirm_tag),
                    width=80
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Delete All",
                    callback=lambda: self._confirm_delete_all(confirm_tag),
                    width=100
                )
    
    def _confirm_delete_all(self, confirm_dialog_tag: str) -> None:
        """Confirm and execute deletion of all embeddings."""
        try:
            from ..vector_store import get_vector_store
            vector_store = get_vector_store("data/vector_store")
            
            node_count = len(self.nodes)
            
            if vector_store.clear():
                # Clear local state
                self.nodes = {}
                self.embeddings = {}
                self.selected_node = None
                self._clear_node_details()
                
                # Refresh map
                self._update_visuals()
                dpg.set_value(self.stats_text_tag, "Nodes: 0 | Connections: 0")
                
                if hasattr(self, 'management_status_text'):
                    dpg.set_value(
                        self.management_status_text,
                        f"Successfully deleted all {node_count} embeddings."
                    )
            else:
                if hasattr(self, 'management_status_text'):
                    dpg.set_value(
                        self.management_status_text,
                        "Error: Failed to delete all embeddings"
                    )
            
            # Close confirm dialog
            if dpg.does_item_exist(confirm_dialog_tag):
                dpg.delete_item(confirm_dialog_tag)
            
            # Close management dialog
            if dpg.does_item_exist("embedding_management_dialog"):
                dpg.delete_item("embedding_management_dialog")
                
        except Exception as e:
            if hasattr(self, 'management_status_text'):
                dpg.set_value(
                    self.management_status_text,
                    f"Error deleting all embeddings: {str(e)}"
                )
            if dpg.does_item_exist(confirm_dialog_tag):
                dpg.delete_item(confirm_dialog_tag)
    
    def _show_manage_embeddings_dialog(self) -> None:
        """Show the manage embeddings dialog."""
        dialog_tag = "manage_embeddings_dialog"
        if dpg.does_item_exist(dialog_tag):
            dpg.delete_item(dialog_tag)
        
        try:
            with dpg.window(
                label="Manage Embeddings",
                modal=True,
                tag=dialog_tag,
                width=500,
                height=300,
                pos=[400, 300],
                on_close=lambda: dpg.delete_item(dialog_tag)
            ):
                dpg.add_text("Embedding Management", color=self.COLORS["accent"][:3])
                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_spacer(height=20)
                
                # Clear Embeddings button (red, white text)
                clear_btn = dpg.add_button(
                    label="Clear Embeddings",
                    callback=self._on_clear_embeddings_click,
                    width=-1,
                    height=40
                )
                if hasattr(self, 'error_theme'):
                    dpg.bind_item_theme(clear_btn, self.error_theme)
                dpg.add_spacer(height=10)
                
                # Export Embeddings button (blue, white text)
                export_btn = dpg.add_button(
                    label="Export Embeddings",
                    callback=self._on_export_embeddings_click,
                    width=-1,
                    height=40
                )
                if hasattr(self, 'blue_button_theme'):
                    dpg.bind_item_theme(export_btn, self.blue_button_theme)
                dpg.add_spacer(height=10)
                
                # Import and Load Embeddings button (blue, white text)
                import_btn = dpg.add_button(
                    label="Import and Load Embeddings",
                    callback=self._on_import_embeddings_click,
                    width=-1,
                    height=40
                )
                if hasattr(self, 'blue_button_theme'):
                    dpg.bind_item_theme(import_btn, self.blue_button_theme)
                dpg.add_spacer(height=20)
                
                # Status text
                self.manage_embeddings_status_text = dpg.add_text(
                    "",
                    color=self.COLORS["text_dim"][:3],
                    wrap=450
                )
        except Exception as e:
            raise
    
    def _on_clear_embeddings_click(self) -> None:
        """Handle clear embeddings button click - shows confirmation dialog."""
        print("[DEBUG] Clear Embeddings button clicked!")
        try:
            print("[DEBUG] Creating confirmation dialog...")
            # Close the manage embeddings dialog first to avoid modal-on-modal issues
            # Must fully close parent modal before creating new one
            if dpg.does_item_exist("manage_embeddings_dialog"):
                dpg.delete_item("manage_embeddings_dialog")
                # Force a render to ensure the deletion takes effect
                dpg.render_dearpygui_frame()
            
            confirm_tag = "clear_embeddings_confirm"
            if dpg.does_item_exist(confirm_tag):
                dpg.delete_item(confirm_tag)
            
            node_count = len(self.nodes)
            print(f"[DEBUG] Node count: {node_count}, Creating confirmation window with tag: {confirm_tag}")
            
            # Calculate window position based on viewport size (centered)
            # Try to get viewport dimensions, fallback to fixed position
            dialog_width = int(450 * self.scale)
            dialog_height = int(250 * self.scale)
            
            try:
                # Try multiple methods to get viewport size
                viewport_width = None
                viewport_height = None
                
                # Method 1: Try get_viewport_client_width/height (if available)
                try:
                    viewport_width = dpg.get_viewport_client_width()
                    viewport_height = dpg.get_viewport_client_height()
                except:
                    pass
                
                # Method 2: Try get_viewport_width/height (alternative API)
                if viewport_width is None:
                    try:
                        viewport_width = dpg.get_viewport_width()
                        viewport_height = dpg.get_viewport_height()
                    except:
                        pass
                
                # Method 3: Try getting primary window size
                if viewport_width is None:
                    try:
                        if dpg.does_item_exist("main_window"):
                            window_size = dpg.get_item_rect_size("main_window")
                            if window_size and len(window_size) >= 2:
                                viewport_width = int(window_size[0])
                                viewport_height = int(window_size[1])
                    except:
                        pass
                
                if viewport_width and viewport_height:
                    dialog_x = int((viewport_width - dialog_width) / 2)
                    dialog_y = int((viewport_height - dialog_height) / 2)
                else:
                    # Fallback to scaled fixed position
                    dialog_x = int(400 * self.scale)
                    dialog_y = int(400 * self.scale)
            except Exception as e:
                # Fallback to scaled fixed position
                dialog_x = int(400 * self.scale)
                dialog_y = int(400 * self.scale)
            
            # Create modal window - must be at viewport level (not child of other windows)
            # Modal windows in Dear PyGui should be created with show=True by default
            with dpg.window(
                label="Confirm Clear Embeddings",
                modal=True,
                tag=confirm_tag,
                width=dialog_width,
                height=dialog_height,
                pos=[dialog_x, dialog_y],
                on_close=lambda: dpg.delete_item(confirm_tag),
                show=True  # Explicitly show the window
            ) as dialog_window:
                # Set item alias for proper reference (required for modal windows in Dear PyGui)
                dpg.set_item_alias(dialog_window, confirm_tag)
                dpg.add_text("[!] WARNING: This will delete ALL embeddings!", color=self.COLORS["node_failure"][:3])
                dpg.add_spacer(height=5)
                dpg.add_text("This action cannot be undone.", color=self.COLORS["text"][:3])
                dpg.add_text(f"Total embeddings to delete: {node_count}", color=self.COLORS["text_dim"][:3])
                dpg.add_spacer(height=15)
                
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=-1)
                    dpg.add_button(
                        label="Cancel",
                        callback=lambda: dpg.delete_item(confirm_tag),
                        width=80
                    )
                    dpg.add_spacer(width=10)
                    # Store callback as instance method to avoid scope issues
                    # Create a wrapper that will be properly bound
                    clear_all_btn = dpg.add_button(
                        label="Clear All",
                        callback=lambda s, a, tag=confirm_tag: self._on_clear_all_confirm(tag),
                        width=100
                    )
            
            # Modal windows should be visible by default, but ensure it's shown
            if dpg.does_item_exist(confirm_tag):
                # Force show the window and bring to front
                dpg.show_item(confirm_tag)
                # Try to bring window to front (if supported)
                try:
                    dpg.focus_item(confirm_tag)
                except:
                    pass
        except Exception as e:
            raise
    
    def _on_clear_all_confirm(self, confirm_dialog_tag: str) -> None:
        """Handle Clear All button click in confirmation dialog."""
        print(f"[DEBUG] Clear All button clicked! Tag: {confirm_dialog_tag}")
        self._confirm_clear_embeddings(confirm_dialog_tag)
    
    def _confirm_clear_embeddings(self, confirm_dialog_tag: str) -> None:
        """Confirm and execute clearing all embeddings."""
        print(f"[DEBUG] _confirm_clear_embeddings called with tag: {confirm_dialog_tag}")
        try:
            from ..vector_store import get_vector_store
            vector_store = get_vector_store("data/vector_store")
            
            node_count = len(self.nodes)
            print(f"[DEBUG] About to clear {node_count} embeddings from vector store")
            
            # Clear the vector store
            clear_result = vector_store.clear()
            print(f"[DEBUG] Vector store clear() returned: {clear_result}")
            
            if clear_result:
                print(f"[DEBUG] Clear successful! Clearing local state and reloading...")
                # Re-get the collection reference after clearing (it gets recreated)
                vector_store.collection = vector_store.client.get_or_create_collection(
                    name=vector_store.COLLECTION_NAME,
                    metadata={"description": "BIDS error fingerprints and fix scripts"}
                )
                
                # Clear local state
                self.nodes = {}
                self.embeddings = {}
                self.selected_node = None
                self._clear_node_details()
                
                # Force a reload from vector store to ensure everything is cleared
                print(f"[DEBUG] Reloading from vector store after clear...")
                self._load_from_vector_store()
                success_msg = f"Nodes: 0 | Connections: 0 | Deleted {node_count} embeddings"
                dpg.set_value(self.stats_text_tag, success_msg)
                print(f"[DEBUG] {success_msg}")
                
                # Show success dialog
                success_dialog_tag = "clear_success_dialog"
                if dpg.does_item_exist(success_dialog_tag):
                    dpg.delete_item(success_dialog_tag)
                
                with dpg.window(
                    label="Clear Successful",
                    modal=True,
                    tag=success_dialog_tag,
                    width=400,
                    height=150,
                    pos=[450, 400]
                ):
                    dpg.add_text(f"[OK] Successfully deleted {node_count} embeddings!", color=self.COLORS["node_success"][:3])
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=-1)
                        dpg.add_button(
                            label="OK",
                            callback=lambda: dpg.delete_item(success_dialog_tag),
                            width=80
                        )
            else:
                error_msg = "Error: Failed to delete all embeddings"
                dpg.set_value(self.stats_text_tag, error_msg)
                print(f"[DEBUG] {error_msg}")
                
                # Show error dialog
                error_dialog_tag = "clear_error_dialog"
                if dpg.does_item_exist(error_dialog_tag):
                    dpg.delete_item(error_dialog_tag)
                
                with dpg.window(
                    label="Clear Failed",
                    modal=True,
                    tag=error_dialog_tag,
                    width=400,
                    height=150,
                    pos=[450, 400]
                ):
                    dpg.add_text("[ERROR] Failed to delete embeddings from vector store!", color=self.COLORS["node_failure"][:3])
                    dpg.add_spacer(height=20)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=-1)
                        dpg.add_button(
                            label="OK",
                            callback=lambda: dpg.delete_item(error_dialog_tag),
                            width=80
                        )
            
            # Close confirm dialog
            if dpg.does_item_exist(confirm_dialog_tag):
                print(f"[DEBUG] Closing confirmation dialog: {confirm_dialog_tag}")
                dpg.delete_item(confirm_dialog_tag)
                
        except Exception as e:
            dpg.set_value(self.stats_text_tag, f"Error deleting embeddings: {str(e)}")
            if dpg.does_item_exist(confirm_dialog_tag):
                dpg.delete_item(confirm_dialog_tag)
    
    def _on_export_embeddings_click(self) -> None:
        """Handle export embeddings button click - creates archive file."""
        try:
            if not self.nodes:
                if hasattr(self, 'manage_embeddings_status_text'):
                    dpg.set_value(
                        self.manage_embeddings_status_text,
                        "No embeddings to export."
                    )
                return
            
            # Close the manage embeddings dialog first to avoid modal-on-modal issues
            if dpg.does_item_exist("manage_embeddings_dialog"):
                dpg.delete_item("manage_embeddings_dialog")
            
            # Create exports directory if it doesn't exist
            exports_dir = Path("exports")
            exports_dir.mkdir(exist_ok=True)
            
            # Create file dialog for saving
            export_dialog_tag = "export_embeddings_dialog"
            if dpg.does_item_exist(export_dialog_tag):
                dpg.delete_item(export_dialog_tag)
            
            with dpg.file_dialog(
                directory_selector=False,
                show=True,
                callback=self._on_export_embeddings_dialog_ok,
                cancel_callback=lambda s, a: dpg.delete_item(export_dialog_tag),
                width=700,
                height=400,
                modal=True,
                tag=export_dialog_tag,
                default_filename="embeddings_export.zip",
                default_path=str(exports_dir.absolute())
            ):
                dpg.add_file_extension(".*", color=(255, 255, 255))
                dpg.add_file_extension(".zip", color=(0, 255, 0))
                
        except Exception as e:
            if hasattr(self, 'manage_embeddings_status_text'):
                dpg.set_value(
                    self.manage_embeddings_status_text,
                    f"Error opening export dialog: {str(e)}"
                )
    
    def _on_export_embeddings_dialog_ok(self, sender, app_data) -> None:
        """Handle export embeddings dialog OK - creates archive."""
        try:
            # For save dialogs, use file_path_name instead of selections
            file_path = app_data.get("file_path_name", "")
            
            if not file_path:
                return
            
            # Ensure .zip extension (handle cases where user might type name without extension)
            file_path = str(file_path).strip()
            if not file_path.lower().endswith('.zip'):
                # If it ends with 'zip' but no dot, add the dot
                if file_path.lower().endswith('zip'):
                    file_path = file_path[:-3] + '.zip'
                else:
                    file_path += '.zip'
            
            # Ensure file is saved in exports directory
            exports_dir = Path("exports")
            exports_dir.mkdir(exist_ok=True)
            file_path_obj = Path(file_path)
            # If file_path is not already in exports directory, move it there
            if "exports" not in str(file_path_obj.parent).replace("\\", "/"):
                file_path = str(exports_dir / file_path_obj.name)
            
            from ..vector_store import get_vector_store
            vector_store = get_vector_store("data/vector_store")
            
            # Get all entries from vector store
            all_results = vector_store.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not all_results.get("ids"):
                # Show error popup
                with dpg.window(
                    label="Export Error",
                    modal=True,
                    width=400,
                    height=120,
                    pos=[500, 400]
                ) as error_popup:
                    dpg.add_text("No embeddings found to export.", color=self.COLORS["node_failure"][:3])
                    dpg.add_spacer(height=10)
                    dpg.add_button(
                        label="OK",
                        callback=lambda: dpg.delete_item(error_popup),
                        width=-1
                    )
                if dpg.does_item_exist("export_embeddings_dialog"):
                    dpg.delete_item("export_embeddings_dialog")
                return
            
            # Create temporary directory for organizing files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create a folder for each embedding
                for i, entry_id in enumerate(all_results["ids"]):
                    embedding_dir = temp_path / entry_id
                    embedding_dir.mkdir(exist_ok=True)
                    
                    # Save embedding vector
                    embeddings_list = all_results.get("embeddings", [])
                    # Check if embeddings_list exists and has items (avoid numpy array truth value error)
                    if embeddings_list is not None and len(embeddings_list) > 0 and i < len(embeddings_list):
                        embedding = embeddings_list[i]
                        # Convert numpy array to list if needed
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        with open(embedding_dir / "embedding.json", 'w') as f:
                            json.dump(embedding, f, indent=2)
                    
                    # Save script content
                    script_content = all_results["documents"][i] if all_results.get("documents") else ""
                    with open(embedding_dir / "script.py", 'w', encoding='utf-8') as f:
                        f.write(script_content)
                    
                    # Save metadata
                    metadata = all_results["metadatas"][i] if all_results.get("metadatas") else {}
                    with open(embedding_dir / "metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Save fingerprint tokens if available
                    if metadata.get("fingerprint_tokens"):
                        with open(embedding_dir / "fingerprint_tokens.txt", 'w', encoding='utf-8') as f:
                            f.write(metadata["fingerprint_tokens"])
                
                # Create zip archive
                with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for embedding_dir in temp_path.iterdir():
                        if embedding_dir.is_dir():
                            for file_path_in_dir in embedding_dir.rglob('*'):
                                if file_path_in_dir.is_file():
                                    arcname = str(Path(embedding_dir.name) / file_path_in_dir.relative_to(embedding_dir))
                                    zipf.write(str(file_path_in_dir), arcname)
                
                count = len(all_results["ids"])
                # Update stats to show success
                dpg.set_value(self.stats_text_tag, f"Nodes: {len(self.nodes)} | Connections: {sum(len(n.connections) for n in self.nodes.values()) // 2} | Exported {count} embeddings")
                
                # Show success message in a popup
                with dpg.window(
                    label="Export Success",
                    modal=True,
                    width=400,
                    height=150,
                    pos=[500, 400]
                ) as success_popup:
                    dpg.add_text(f"Successfully exported {count} embeddings", color=self.COLORS["node_success"][:3])
                    dpg.add_text(f"File: {Path(file_path).name}", color=self.COLORS["text_dim"][:3])
                    dpg.add_spacer(height=10)
                    dpg.add_button(
                        label="OK",
                        callback=lambda: dpg.delete_item(success_popup),
                        width=-1
                    )
            
            # Close export dialog
            if dpg.does_item_exist("export_embeddings_dialog"):
                dpg.delete_item("export_embeddings_dialog")
                
        except Exception as e:
            # Show error in a popup
            with dpg.window(
                label="Export Error",
                modal=True,
                width=400,
                height=150,
                pos=[500, 400]
            ) as error_popup:
                dpg.add_text(f"Error exporting embeddings: {str(e)}", color=self.COLORS["node_failure"][:3])
                dpg.add_spacer(height=10)
                dpg.add_button(
                    label="OK",
                    callback=lambda: dpg.delete_item(error_popup),
                    width=-1
                )
            if dpg.does_item_exist("export_embeddings_dialog"):
                dpg.delete_item("export_embeddings_dialog")
    
    def _on_import_embeddings_click(self) -> None:
        """Handle import embeddings button click - opens file dialog."""
        try:
            # Close the manage embeddings dialog first to avoid modal-on-modal issues
            if dpg.does_item_exist("manage_embeddings_dialog"):
                dpg.delete_item("manage_embeddings_dialog")
            
            # Create file dialog for loading
            import_dialog_tag = "import_embeddings_dialog"
            if dpg.does_item_exist(import_dialog_tag):
                dpg.delete_item(import_dialog_tag)
            
            inputs_dir = Path("inputs")
            inputs_dir.mkdir(exist_ok=True)
            
            with dpg.file_dialog(
                directory_selector=False,
                show=True,
                callback=self._on_import_embeddings_dialog_ok,
                cancel_callback=lambda s, a: dpg.delete_item(import_dialog_tag),
                width=700,
                height=400,
                modal=True,
                tag=import_dialog_tag
            ):
                dpg.add_file_extension(".*", color=(255, 255, 255))
                dpg.add_file_extension(".zip", color=(0, 255, 0))
            
            try:
                dpg.configure_item(import_dialog_tag, default_path=str(inputs_dir.absolute()))
            except:
                pass
                
        except Exception as e:
            if hasattr(self, 'manage_embeddings_status_text'):
                dpg.set_value(
                    self.manage_embeddings_status_text,
                    f"Error opening import dialog: {str(e)}"
                )
    
    def _on_import_embeddings_dialog_ok(self, sender, app_data) -> None:
        """Handle import embeddings dialog OK - loads from archive."""
        try:
            selections = app_data.get("selections", {})
            if not selections:
                return
            
            file_path = list(selections.values())[0]
            
            from ..vector_store import get_vector_store
            vector_store = get_vector_store("data/vector_store")
            
            # Extract archive to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract zip file
                with zipfile.ZipFile(file_path, 'r') as zipf:
                    zipf.extractall(temp_path)
                
                # Import each embedding folder
                count = 0
                for embedding_dir in temp_path.iterdir():
                    if not embedding_dir.is_dir():
                        continue
                    
                    entry_id = embedding_dir.name
                    
                    # Load embedding
                    embedding_file = embedding_dir / "embedding.json"
                    if embedding_file.exists():
                        with open(embedding_file, 'r') as f:
                            embedding = json.load(f)
                    else:
                        embedding = [0.0] * vector_store.embedding_dimension
                    
                    # Load script content
                    script_file = embedding_dir / "script.py"
                    script_content = ""
                    if script_file.exists():
                        with open(script_file, 'r', encoding='utf-8') as f:
                            script_content = f.read()
                    
                    # Load metadata
                    metadata_file = embedding_dir / "metadata.json"
                    metadata = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    
                    # Add to vector store
                    try:
                        vector_store.collection.add(
                            ids=[entry_id],
                            embeddings=[embedding],
                            documents=[script_content],
                            metadatas=[metadata]
                        )
                        count += 1
                    except Exception as e:
                        # If entry already exists, skip or update
                        # For now, just skip duplicates
                        continue
                
                if count > 0:
                    # Show success message in a popup dialog
                    success_msg = f"[OK] {count} embedding{'s' if count != 1 else ''} successfully loaded!"
                    
                    # Show success popup
                    success_dialog_tag = "import_success_dialog"
                    if dpg.does_item_exist(success_dialog_tag):
                        dpg.delete_item(success_dialog_tag)
                    
                    with dpg.window(
                        label="Import Successful",
                        modal=True,
                        tag=success_dialog_tag,
                        width=400,
                        height=150,
                        pos=[450, 400]
                    ):
                        dpg.add_text(success_msg, color=self.COLORS["node_success"][:3])
                        dpg.add_spacer(height=20)
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=-1)
                            dpg.add_button(
                                label="OK",
                                callback=lambda: dpg.delete_item(success_dialog_tag),
                                width=80
                            )
                    
                    # Also update stats text
                    if hasattr(self, 'stats_text_tag'):
                        dpg.set_value(self.stats_text_tag, success_msg)
                    # Refresh the map - preserve current view position
                    print(f"[DEBUG] Loading embeddings after import, preserving view...")
                    # Store current view state to preserve it
                    old_view_x = self.view_offset_x
                    old_view_y = self.view_offset_y
                    old_zoom = self.zoom
                    self._load_from_vector_store()
                    # Restore view state after load
                    self.view_offset_x = old_view_x
                    self.view_offset_y = old_view_y
                    self.zoom = old_zoom
                    print(f"[DEBUG] View restored: offset=({old_view_x}, {old_view_y}), zoom={old_zoom}")
                    self._update_visuals()
                    
                    # Close management dialog
                    if dpg.does_item_exist("manage_embeddings_dialog"):
                        dpg.delete_item("manage_embeddings_dialog")
                else:
                    if hasattr(self, 'manage_embeddings_status_text'):
                        dpg.set_value(
                            self.manage_embeddings_status_text,
                            "Error: Failed to import embeddings or archive was empty/invalid"
                        )
            
            # Close import dialog
            if dpg.does_item_exist("import_embeddings_dialog"):
                dpg.delete_item("import_embeddings_dialog")
                
        except Exception as e:
            if hasattr(self, 'manage_embeddings_status_text'):
                dpg.set_value(
                    self.manage_embeddings_status_text,
                    f"Error importing embeddings: {str(e)}"
                )
            if dpg.does_item_exist("import_embeddings_dialog"):
                dpg.delete_item("import_embeddings_dialog")

