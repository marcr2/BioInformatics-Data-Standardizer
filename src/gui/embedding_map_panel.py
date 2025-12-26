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
    
    NODE_RADIUS = 12
    CONNECTION_THRESHOLD = 0.7  # Similarity threshold for drawing connections
    
    def __init__(self, get_current_data_callback=None):
        """
        Initialize the embedding map panel.
        
        Args:
            get_current_data_callback: Callback function to get currently loaded DataFrame
        """
        self.nodes: Dict[str, MapNode] = {}
        self.embeddings: Dict[str, np.ndarray] = {}  # Store embeddings by node ID
        self.selected_node: Optional[str] = None
        self.hovered_node: Optional[str] = None
        
        # View state
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.zoom = 1.0
        self.canvas_width = 800
        self.canvas_height = 500
        
        # UI elements
        self.drawlist_tag: Optional[int] = None
        self.script_text_tag: Optional[int] = None
        self.info_container_tag: Optional[int] = None
        self.stats_text_tag: Optional[int] = None
        self.pse_section_tag: Optional[int] = None
        self.ideal_file_text: Optional[int] = None
        self.import_ideal_button: Optional[int] = None
        
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
    
    def create(self) -> None:
        """Create the embedding map panel UI."""
        with dpg.group(horizontal=True):
            # Left side: Canvas
            with dpg.child_window(width=-350, height=-1, border=True):
                self._create_canvas_area()
            
            # Right side: Info panel
            with dpg.child_window(width=-1, height=-1, border=True):
                self._create_info_panel()
        
        # Create ideal file dialog (hidden initially)
        self._create_ideal_file_dialog()
        
        # Automatically load embeddings from vector store
        self._load_from_vector_store()
    
    def _create_canvas_area(self) -> None:
        """Create the interactive canvas area."""
        # Toolbar
        with dpg.group(horizontal=True):
            dpg.add_text("Embedding Map", color=self.COLORS["accent"][:3])
            dpg.add_spacer(width=20)
            dpg.add_button(
                label="Refresh",
                callback=self._on_refresh_click,
                width=80
            )
            dpg.add_button(
                label="Reset View",
                callback=self._reset_view,
                width=80
            )
            dpg.add_button(
                label="Re-center",
                callback=self._recenter_view,
                width=80
            )
            dpg.add_spacer(width=20)
            dpg.add_button(
                label="Manage Embeddings",
                callback=self._on_manage_embeddings_click,
                width=130
            )
            dpg.add_spacer(width=20)
            self.stats_text_tag = dpg.add_text(
                "Nodes: 0 | Connections: 0",
                color=self.COLORS["text_dim"][:3]
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
        
        # Legend
        with dpg.group(horizontal=True):
            dpg.add_text("Legend:", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(width=10)
            dpg.add_text("●", color=self.COLORS["node_success"][:3])
            dpg.add_text("Success", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(width=10)
            dpg.add_text("●", color=self.COLORS["node_failure"][:3])
            dpg.add_text("Failure", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(width=10)
            dpg.add_text("●", color=self.COLORS["node_pending"][:3])
            dpg.add_text("Pending", color=self.COLORS["text_dim"][:3])
        
        dpg.add_spacer(height=5)
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        # Canvas with drawlist
        with dpg.child_window(border=False, height=-1) as canvas_window:
            self.canvas_window_tag = canvas_window
            
            # Create drawlist for the visualization
            self.drawlist_tag = dpg.add_drawlist(
                width=self.canvas_width,
                height=self.canvas_height
            )
            
            # Resize handle (corner grip)
            self.resize_handle_tag = dpg.add_drawlist(
                width=20,
                height=20,
                parent=canvas_window
            )
            # Draw resize handle in bottom-right corner
            self._draw_resize_handle()
            
            # Register mouse handlers
            # #region agent log
            import time
            import json
            log_entry = {
                "timestamp": int(time.time() * 1000),
                "location": "embedding_map_panel.py:197",
                "message": "Creating handler_registry for mouse handlers including click",
                "data": {
                    "canvas_window": str(canvas_window),
                    "drawlist_tag": str(self.drawlist_tag)
                },
                "sessionId": "debug-session",
                "runId": "run3",
                "hypothesisId": "G"
            }
            with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            # #endregion
            # Register mouse handlers (global handlers, we check bounds in callbacks)
            with dpg.handler_registry(tag=f"embedding_map_handlers_{id(self)}"):
                # Mouse move for hover effects
                dpg.add_mouse_move_handler(callback=self._on_mouse_move)
                # Mouse wheel for zooming
                dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
                # Right-click drag for panning
                dpg.add_mouse_drag_handler(
                    button=dpg.mvMouseButton_Right,
                    callback=self._on_mouse_drag
                )
                # Middle-click drag for resizing
                dpg.add_mouse_drag_handler(
                    button=dpg.mvMouseButton_Middle,
                    callback=self._on_resize_drag
                )
                # Left mouse click handler
                dpg.add_mouse_click_handler(
                    button=dpg.mvMouseButton_Left,
                    callback=self._on_canvas_click
                )
            
            # Draw initial empty state
            self._draw_empty_state()
    
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
            
            # Number of PSEs
            dpg.add_text("Number of PSEs:", color=self.COLORS["text_dim"][:3])
            with dpg.group(horizontal=True):
                self.pse_count_input = dpg.add_slider_int(
                    default_value=5,
                    min_value=1,
                    max_value=50,
                    width=-1,
                    callback=self._on_pse_count_changed
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Value: ", color=self.COLORS["text_dim"][:3])
                self.pse_count_display = dpg.add_text("5", color=self.COLORS["text"][:3])
            
            dpg.add_spacer(height=5)
            
            # Noise ratio
            dpg.add_text("Noise Ratio (%):", color=self.COLORS["text_dim"][:3])
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=20)
                self.noise_ratio_input = dpg.add_input_float(
                    default_value=0.1,
                    min_value=0.01,
                    max_value=1.0,
                    format="%.2f",
                    width=120
                )
                dpg.add_spacer(width=20)
            
            dpg.add_spacer(height=5)
            
            # Noise types checkboxes
            dpg.add_text("Noise Types:", color=self.COLORS["text_dim"][:3])
            self.typo_checkbox = dpg.add_checkbox(label="Typo", default_value=True)
            self.semantic_checkbox = dpg.add_checkbox(label="Semantic", default_value=True)
            self.structural_checkbox = dpg.add_checkbox(label="Structural", default_value=True)
            
            dpg.add_spacer(height=5)
            
            # Distribution mode
            dpg.add_text("Distribution:", color=self.COLORS["text_dim"][:3])
            self.dist_uniform_radio = dpg.add_radio_button(
                ["Uniform", "Targeted"],
                default_value=0,
                horizontal=True
            )
            
            dpg.add_spacer(height=5)
            
            # Target columns (shown when targeted mode selected)
            with dpg.group(horizontal=True):
                dpg.add_text("Target Columns:", color=self.COLORS["text_dim"][:3])
                dpg.add_spacer(width=10)
                self.target_cols_input = dpg.add_input_text(
                    default_value="",
                    hint="comma-separated column names",
                    width=-1
                )
            
            dpg.add_spacer(height=5)
            
            # Dynamic noise density checkbox
            self.dynamic_noise_checkbox = dpg.add_checkbox(
                label="Dynamic Noise Density",
                default_value=False,
                callback=self._on_dynamic_noise_changed
            )
            dpg.add_text(
                "Creates PSEs with varying noise ratios\n(min 1% steps, min 100 PSEs)",
                color=self.COLORS["text_dim"][:3],
                wrap=250
            )
            
            # DND-specific options (hidden by default)
            with dpg.group() as dnd_options_group:
                self.dnd_options_group = dnd_options_group
                dpg.hide_item(self.dnd_options_group)
                
                dpg.add_spacer(height=5)
                dpg.add_text("DND Settings:", color=self.COLORS["accent"][:3])
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
    
    def _world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        """Convert world coordinates to screen coordinates."""
        sx = (wx * self.zoom) + self.view_offset_x + self.canvas_width / 2
        sy = (wy * self.zoom) + self.view_offset_y + self.canvas_height / 2
        return sx, sy
    
    def _screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        wx = (sx - self.view_offset_x - self.canvas_width / 2) / self.zoom
        wy = (sy - self.view_offset_y - self.canvas_height / 2) / self.zoom
        return wx, wy
    
    def _draw_map(self) -> None:
        """Draw the full embedding map."""
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
            self._draw_empty_state()
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
        for node_id, node in self.nodes.items():
            sx, sy = self._world_to_screen(node.x, node.y)
            
            # Skip if off-screen
            if not self._is_visible(sx, sy):
                continue
            
            # Determine color based on status and state
            if node_id == self.selected_node:
                color = self.COLORS["node_selected"]
                radius = self.NODE_RADIUS + 4
            elif node_id == self.hovered_node:
                color = self.COLORS["node_hover"]
                radius = self.NODE_RADIUS + 2
            else:
                if node.status == "SUCCESS":
                    color = self.COLORS["node_success"]
                elif node.status == "FAILURE":
                    color = self.COLORS["node_failure"]
                else:
                    color = self.COLORS["node_pending"]
                radius = self.NODE_RADIUS
            
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
            
            # Node label (truncated)
            label = node.label[:10] + "..." if len(node.label) > 10 else node.label
            dpg.draw_text(
                pos=[sx - len(label) * 3, sy + radius + 5],
                text=label,
                color=self.COLORS["text"],
                size=10,
                parent=self.drawlist_tag
            )
    
    def _is_visible(self, sx: float, sy: float) -> bool:
        """Check if a screen position is visible in the canvas."""
        margin = self.NODE_RADIUS * 2
        return (
            -margin <= sx <= self.canvas_width + margin and
            -margin <= sy <= self.canvas_height + margin
        )
    
    def _on_canvas_click(self, sender, app_data) -> None:
        """Handle canvas click events."""
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
            mouse_pos = dpg.get_mouse_pos(local=False)
            drawlist_pos = dpg.get_item_pos(self.drawlist_tag)
            
            if not drawlist_pos:
                return
            
            # Convert to local coordinates
            local_x = mouse_pos[0] - drawlist_pos[0]
            local_y = mouse_pos[1] - drawlist_pos[1]
            
            # Find clicked node using local coordinates
            clicked_node = self._get_node_at_position_local(local_x, local_y)
            
            if clicked_node:
                self.selected_node = clicked_node
                self._show_node_details(self.nodes[clicked_node])
            else:
                self.selected_node = None
                self._clear_node_details()
            
            self._draw_map()
        except Exception as e:
            # Silently fail if there's an error
            pass
    
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
        """Find a node at the given local drawlist coordinates."""
        if not self.nodes:
            return None
        
        # Check bounds
        if local_x < 0 or local_y < 0 or local_x > self.canvas_width or local_y > self.canvas_height:
            return None
        
        # Find closest node
        closest_node = None
        closest_distance = float('inf')
        
        for node_id, node in self.nodes.items():
            sx, sy = self._world_to_screen(node.x, node.y)
            
            distance = math.sqrt((local_x - sx) ** 2 + (local_y - sy) ** 2)
            if distance <= self.NODE_RADIUS + 5 and distance < closest_distance:
                closest_distance = distance
                closest_node = node_id
        
        return closest_node
    
    def _on_mouse_move(self, sender, app_data) -> None:
        """Handle mouse move for hover effects and resize handle."""
        # Update resize handle position if drawlist size changed
        if hasattr(self, 'resize_handle_tag') and self.resize_handle_tag:
            self._draw_resize_handle()
        
        # Check if dragging ended (right button released)
        if self.is_dragging:
            if not dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
                self.is_dragging = False
                # Final redraw when drag ends
                self._draw_map()
                return
        
        # Handle node hover (only if mouse is over drawlist)
        if not self.nodes or not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        if not dpg.is_item_hovered(self.drawlist_tag):
            if self.hovered_node:
                self.hovered_node = None
                if not self.is_dragging:
                    self._draw_map()
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
                        self._draw_map()
        except Exception:
            pass
        
        # Check if we should stop resizing (mouse button released)
        if self.is_resizing:
            # Check if middle button is still pressed
            if not dpg.is_mouse_button_down(dpg.mvMouseButton_Middle):
                self.is_resizing = False
    
    def _on_mouse_wheel(self, sender, app_data) -> None:
        """Handle mouse wheel for zooming."""
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
                if not (drawlist_pos[0] <= mouse_pos[0] <= drawlist_pos[0] + drawlist_size[0] and
                        drawlist_pos[1] <= mouse_pos[1] <= drawlist_pos[1] + drawlist_size[1]):
                    return  # Mouse not over map, ignore scroll
        except Exception:
            # If we can't check bounds, still try to process (fallback)
            pass
        
        delta = app_data
        
        # Zoom in/out
        zoom_factor = 1.1 if delta > 0 else 0.9
        new_zoom = self.zoom * zoom_factor
        
        # Clamp zoom
        new_zoom = max(0.2, min(3.0, new_zoom))
        
        self.zoom = new_zoom
        self._draw_map()
    
    def _on_mouse_drag(self, sender, app_data) -> None:
        """Handle right-click drag for panning."""
        # Skip if we're resizing
        if self.is_resizing:
            return
        
        # Mark that we're dragging
        if not self.is_dragging:
            self.is_dragging = True
            self.drag_redraw_timer = 0
        
        # app_data is [dx, dy]
        dx, dy = app_data[1], app_data[2]
        
        # Decrease drag sensitivity (multiply by 0.3 for smoother, less sensitive dragging)
        drag_sensitivity = 0.3
        self.view_offset_x += dx * drag_sensitivity
        self.view_offset_y += dy * drag_sensitivity
        
        # Only redraw every few frames to reduce lag (throttle redraws)
        self.drag_redraw_timer += 1
        if self.drag_redraw_timer >= 3:  # Redraw every 3rd drag event for smoother performance
            self._draw_map()
            self.drag_redraw_timer = 0
    
    def _on_resize_drag(self, sender, app_data) -> None:
        """Handle middle-click drag for resizing the map."""
        # Check if mouse is near the bottom-right corner of the drawlist
        if not self.drawlist_tag or not dpg.does_item_exist(self.drawlist_tag):
            return
        
        try:
            mouse_pos = dpg.get_mouse_pos(local=False)
            drawlist_pos = dpg.get_item_pos(self.drawlist_tag)
            drawlist_size = dpg.get_item_rect_size(self.drawlist_tag)
            
            if not drawlist_pos or not drawlist_size:
                return
            
            # Check if we're starting a resize (near bottom-right corner) or continuing one
            corner_threshold = 30  # pixels
            corner_x = drawlist_pos[0] + drawlist_size[0]
            corner_y = drawlist_pos[1] + drawlist_size[1]
            
            distance_to_corner = math.sqrt(
                (mouse_pos[0] - corner_x) ** 2 + 
                (mouse_pos[1] - corner_y) ** 2
            )
            
            # Start resize if near corner and not already resizing
            if not self.is_resizing and distance_to_corner <= corner_threshold:
                self.is_resizing = True
                self.resize_start_width = self.canvas_width
                self.resize_start_height = self.canvas_height
                self.resize_start_mouse_x = mouse_pos[0]
                self.resize_start_mouse_y = mouse_pos[1]
            
            # Continue resize if already resizing
            if self.is_resizing:
                # app_data is [dx, dy]
                dx, dy = app_data[1], app_data[2]
                
                # Calculate new size
                new_width = max(200, min(2000, self.resize_start_width + dx))
                new_height = max(200, min(2000, self.resize_start_height + dy))
                
                # Update canvas size
                self.canvas_width = int(new_width)
                self.canvas_height = int(new_height)
                
                # Update drawlist size
                dpg.configure_item(self.drawlist_tag, width=self.canvas_width, height=self.canvas_height)
                
                # Redraw
                self._draw_map()
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
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.zoom = 1.0
        self._draw_map()
    
    def _recenter_view(self) -> None:
        """Re-center the view on the map content."""
        if not self.nodes:
            self._reset_view()
            return
        
        # Calculate center of all nodes in world coordinates
        avg_x = sum(node.x for node in self.nodes.values()) / len(self.nodes)
        avg_y = sum(node.y for node in self.nodes.values()) / len(self.nodes)
        
        # Convert to screen coordinates to find current offset needed
        # We want the center of nodes to be at screen center
        # screen_center_x = (avg_x * self.zoom) + self.view_offset_x + self.canvas_width / 2
        # screen_center_y = (avg_y * self.zoom) + self.view_offset_y + self.canvas_height / 2
        # We want screen_center_x = canvas_width / 2, so:
        # (avg_x * self.zoom) + self.view_offset_x + self.canvas_width / 2 = self.canvas_width / 2
        # => self.view_offset_x = -avg_x * self.zoom
        
        self.view_offset_x = -avg_x * self.zoom
        self.view_offset_y = -avg_y * self.zoom
        
        self._draw_map()
    
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
        
        self._draw_map()
    
    def _show_node_details(self, node: MapNode) -> None:
        """Display details for a selected node."""
        # Clear info container
        dpg.delete_item(self.info_container_tag, children_only=True)
        
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
            
            # Preview Embedding button
            dpg.add_button(
                label="Preview Embedding",
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
                self._draw_empty_state()
                dpg.set_value(self.stats_text_tag, "Nodes: 0 | Connections: 0")
                return
            
            # Get embeddings and project to 2D
            embeddings = all_results.get("embeddings", [])
            # #region agent log
            import time
            import json
            emb_type = type(embeddings).__name__
            emb_len = len(embeddings) if hasattr(embeddings, "__len__") else "N/A"
            log_entry = {
                "timestamp": int(time.time() * 1000),
                "location": "embedding_map_panel.py:694",
                "message": "Checking embeddings type",
                "data": {"type": emb_type, "length": str(emb_len)},
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "B"
            }
            with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            # #endregion
            # Handle numpy array or list - check length first to avoid truth value ambiguity
            if len(embeddings) == 0:
                self._draw_empty_state()
                return
            
            # Convert to list if it's a numpy array
            if isinstance(embeddings, np.ndarray):
                # #region agent log
                log_entry2 = {
                    "timestamp": int(time.time() * 1000),
                    "location": "embedding_map_panel.py:711",
                    "message": "Converting numpy array to list",
                    "data": {"shape": str(embeddings.shape)},
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B"
                }
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps(log_entry2) + '\n')
                # #endregion
                embeddings = embeddings.tolist()
            
            # Project to 2D using t-SNE or simple PCA fallback
            positions = self._project_embeddings(np.array(embeddings))
            
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
                
                node = MapNode(
                    id=entry_id,
                    x=x,
                    y=y,
                    label=metadata.get("error_type", "unknown")[:15],
                    status=metadata.get("status", "PENDING"),
                    script_content=document,
                    metadata=metadata,
                    fingerprint_tokens=metadata.get("fingerprint_tokens", "").split()[:20],
                    connections=connections.get(i, [])
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
            
            # Draw the map
            self._draw_map()
            
        except Exception as e:
            print(f"Error loading from vector store: {e}")
            self._draw_empty_state()
            dpg.set_value(self.stats_text_tag, f"Error: {str(e)[:30]}")
    
    def _project_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Project high-dimensional embeddings to 2D."""
        if len(embeddings) == 0:
            return np.array([])
        
        if len(embeddings) == 1:
            return np.array([[0, 0]])
        
        try:
            # Use t-SNE for small datasets, PCA for larger
            if len(embeddings) <= 50:
                from sklearn.manifold import TSNE
                perplexity = min(30, max(5, len(embeddings) - 1))
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                positions = tsne.fit_transform(embeddings)
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                positions = pca.fit_transform(embeddings)
            
            # Normalize to [-1, 1]
            if positions.max() != positions.min():
                positions = (positions - positions.min()) / (positions.max() - positions.min()) * 2 - 1
            
            return positions
            
        except Exception as e:
            print(f"Projection error: {e}")
            # Fallback: random positions
            return np.random.randn(len(embeddings), 2)
    
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
        
        # Update stats and redraw
        total_connections = sum(len(n.connections) for n in self.nodes.values()) // 2
        dpg.set_value(
            self.stats_text_tag,
            f"Nodes: {len(self.nodes)} | Connections: {total_connections}"
        )
        
        self._draw_map()
    
    def _on_pse_count_changed(self, sender, app_data) -> None:
        """Update the display text when the PSE count slider changes."""
        if self.pse_count_display:
            dpg.set_value(self.pse_count_display, str(int(app_data)))
    
    def _on_dynamic_noise_changed(self, sender, app_data) -> None:
        """Handle dynamic noise density checkbox changes."""
        dnd_enabled = app_data
        
        # Show/hide DND-specific options
        if hasattr(self, 'dnd_options_group'):
            if dnd_enabled:
                dpg.show_item(self.dnd_options_group)
            else:
                dpg.hide_item(self.dnd_options_group)
        
        # Lock/unlock non-DND options
        # Lock: noise ratio, distribution mode, target columns
        # Keep enabled: noise types, PSE count
        if hasattr(self, 'noise_ratio_input'):
            dpg.configure_item(self.noise_ratio_input, enabled=not dnd_enabled)
        
        if hasattr(self, 'dist_uniform_radio'):
            dpg.configure_item(self.dist_uniform_radio, enabled=not dnd_enabled)
        
        if hasattr(self, 'target_cols_input'):
            dpg.configure_item(self.target_cols_input, enabled=not dnd_enabled)
    
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
        
        # Get configuration
        dynamic_noise = dpg.get_value(self.dynamic_noise_checkbox)
        
        if dynamic_noise:
            # DND mode: use DND-specific options
            min_noise = dpg.get_value(self.dnd_min_noise_input) if hasattr(self, 'dnd_min_noise_input') else 0.01
            max_noise = dpg.get_value(self.dnd_max_noise_input) if hasattr(self, 'dnd_max_noise_input') else 0.1
            step_size = dpg.get_value(self.dnd_step_size_input) if hasattr(self, 'dnd_step_size_input') else 0.01
            pses_per_step = int(dpg.get_value(self.dnd_pses_per_step_input)) if hasattr(self, 'dnd_pses_per_step_input') else 10
            
            # Calculate total PSEs
            num_steps = int((max_noise - min_noise) / step_size) + 1
            num_pses = num_steps * pses_per_step
            
            # For DND, use uniform distribution and no target columns
            dist_mode = 0  # Uniform
            target_cols_str = ""
            noise_ratio = max_noise  # Used as fallback, but DND will override
        else:
            # Standard mode: use regular options
            num_pses = int(dpg.get_value(self.pse_count_input))
            noise_ratio = dpg.get_value(self.noise_ratio_input)
            dist_mode = dpg.get_value(self.dist_uniform_radio)
            target_cols_str = dpg.get_value(self.target_cols_input)
            min_noise = None
            max_noise = None
            step_size = None
            pses_per_step = None
        
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
        
        # Parse target columns
        target_columns = None
        if dist_mode == 1:  # Targeted mode
            if target_cols_str.strip():
                target_columns = [col.strip() for col in target_cols_str.split(",") if col.strip()]
                # Validate columns exist
                missing_cols = [col for col in target_columns if col not in current_df.columns]
                if missing_cols:
                    dpg.set_value(
                        self.pse_status_text,
                        f"Error: Columns not found: {', '.join(missing_cols)}"
                    )
                    return
            else:
                dpg.set_value(self.pse_status_text, "Error: Please specify target columns for targeted mode")
                return
        
        # Disable button and show status
        self.pse_generating = True
        dpg.configure_item(self.generate_pse_button, enabled=False)
        dpg.set_value(self.pse_status_text, f"Generating {num_pses} PSEs...")
        
        # Run generation in background thread
        import threading
        if dynamic_noise:
            thread = threading.Thread(
                target=self._generate_pse_thread,
                args=(current_df, num_pses, noise_ratio, noise_types, target_columns, dynamic_noise, min_noise, max_noise, step_size, pses_per_step),
                daemon=True
            )
        else:
            thread = threading.Thread(
                target=self._generate_pse_thread,
                args=(current_df, num_pses, noise_ratio, noise_types, target_columns, dynamic_noise, None, None, None, None),
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
            
            # Calculate PSE generation parameters
            if dynamic_noise and dnd_min_noise is not None and dnd_max_noise is not None and dnd_step_size is not None and dnd_pses_per_step is not None:
                # Dynamic noise: use DND-specific parameters
                # Generate noise ratios from min to max in steps
                noise_ratios = []
                current_noise = dnd_min_noise
                while current_noise <= dnd_max_noise:
                    noise_ratios.append(current_noise)
                    current_noise += dnd_step_size
                
                # Distribute PSEs across noise ratios
                pses_per_ratio = dnd_pses_per_step
                remaining = 0
                actual_num_pses = len(noise_ratios) * pses_per_ratio
                
                # Create list of (noise_ratio, count) pairs
                pse_configs = []
                for i, nr in enumerate(noise_ratios):
                    count = pses_per_ratio + (1 if i < remaining else 0)
                    pse_configs.append((nr, count))
            else:
                # Standard mode: all PSEs use the same noise ratio
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
                # Create noise config with current noise ratio
                noise_type_enums = [NoiseType(nt) for nt in noise_types]
                dist_mode = DistributionMode.TARGETED if target_columns else DistributionMode.UNIFORM
                
                config = NoiseConfig(
                    noise_ratio=current_noise_ratio,
                    distribution_mode=dist_mode,
                    target_columns=target_columns or [],
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
                
                # Determine error type label
                if dynamic_noise:
                    error_type_label = f"PSE_{noise_types[0] if noise_types else 'mixed'}_{current_noise_ratio:.0%}"
                else:
                    error_type_label = f"PSE_{noise_types[0] if noise_types else 'mixed'}"
                
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
            if dynamic_noise:
                dpg.set_value(
                    self.pse_status_text,
                    f"Successfully generated {generated_count} PSEs with dynamic noise density! Click 'Refresh' to see them on the map."
                )
            else:
                dpg.set_value(
                    self.pse_status_text,
                    f"Successfully generated {generated_count} PSEs! Click 'Refresh' to see them on the map."
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
            dpg.add_text("⚠ Warning: This will delete ALL embeddings!", color=self.COLORS["node_failure"][:3])
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
            
            if vector_store.delete_entry(node_id):
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
            dpg.add_text("⚠ WARNING: This will delete ALL embeddings!", color=self.COLORS["node_failure"][:3])
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
                self._draw_empty_state()
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

