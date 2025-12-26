"""
Cycle Visualization Panel for BIDS GUI

Shows a visual diagram of the processing cycle and current stage.
"""

import dearpygui.dearpygui as dpg
from typing import Optional, Dict, Any
from ..utils.checkpoint_manager import ProcessingStage, CheckpointManager


class CycleVisualizationPanel:
    """
    Panel showing the processing cycle visualization.
    
    Features:
    - Visual diagram of processing stages
    - Highlighting of current stage
    - Progress indicators
    """
    
    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        """
        Initialize cycle visualization panel.
        
        Args:
            checkpoint_manager: Checkpoint manager instance
        """
        self.checkpoint_manager = checkpoint_manager
        self.current_stage = ProcessingStage.IDLE
        
        # UI tags
        self.drawing_tag: Optional[int] = None
        self.stage_text_tag: Optional[int] = None
        self.canvas_width = 800
        self.canvas_height = 400
        
        # Stage positions and info (now includes PREPROCESSING stage)
        self.stages = [
            {
                "id": ProcessingStage.IDLE,
                "label": "Idle",
                "x": 50,
                "y": 200,
                "color": (149, 165, 166),
                "active_color": (100, 149, 237)
            },
            {
                "id": ProcessingStage.PREPROCESSING,
                "label": "Prep",
                "x": 150,
                "y": 200,
                "color": (149, 165, 166),
                "active_color": (52, 152, 219)  # Blue for preprocessing
            },
            {
                "id": ProcessingStage.LOADING_LLM,
                "label": "Load",
                "x": 260,
                "y": 200,
                "color": (149, 165, 166),
                "active_color": (100, 149, 237)
            },
            {
                "id": ProcessingStage.DIAGNOSING,
                "label": "Diag",
                "x": 380,
                "y": 200,
                "color": (149, 165, 166),
                "active_color": (100, 149, 237)
            },
            {
                "id": ProcessingStage.FIX_ATTEMPT,
                "label": "Anal",
                "x": 510,
                "y": 200,
                "color": (149, 165, 166),
                "active_color": (241, 196, 15)
            },
            {
                "id": ProcessingStage.VALIDATING,
                "label": "Vali",
                "x": 640,
                "y": 200,
                "color": (149, 165, 166),
                "active_color": (100, 149, 237)
            },
            {
                "id": ProcessingStage.COMPLETE,
                "label": "Comp",
                "x": 750,
                "y": 200,
                "color": (149, 165, 166),
                "active_color": (46, 204, 113)
            },
            {
                "id": ProcessingStage.ERROR,
                "label": "Error",
                "x": 400,
                "y": 100,
                "color": (149, 165, 166),
                "active_color": (231, 76, 60)
            }
        ]
    
    def create(self) -> None:
        """Create the cycle visualization panel UI."""
        dpg.add_text("Processing Cycle", color=(100, 149, 237))
        dpg.add_spacer(height=10)
        
        # Current stage indicator
        with dpg.group(horizontal=True):
            dpg.add_text("Current Stage: ", color=(149, 165, 166))
            self.stage_text_tag = dpg.add_text(
                "Idle",
                color=(100, 149, 237)
            )
        
        dpg.add_spacer(height=10)
        
        # Drawing canvas
        with dpg.drawlist(width=self.canvas_width, height=self.canvas_height):
            self.drawing_tag = dpg.last_item()
        
        dpg.add_spacer(height=10)
        
        # Checkpoint info
        with dpg.group():
            dpg.add_text("Checkpoint Info", color=(149, 165, 166))
            self.checkpoint_info_tag = dpg.add_text(
                "No checkpoints yet",
                color=(149, 165, 166),
                wrap=-1
            )
        
        # Initial render
        self.update_stage(ProcessingStage.IDLE)
        
        # Set initial stage text
        if self.stage_text_tag:
            dpg.set_value(self.stage_text_tag, "Idle")
    
    def update_stage(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the current stage in the visualization.
        
        Args:
            stage: Current processing stage
            metadata: Optional metadata about the stage
        """
        self.current_stage = stage
        
        # Update stage text
        if self.stage_text_tag:
            stage_label = next(
                (s["label"].replace("\n", " ") for s in self.stages if s["id"] == stage),
                stage.replace("_", " ").title()
            )
            dpg.set_value(self.stage_text_tag, stage_label)
        
        # Update checkpoint info
        if self.checkpoint_manager:
            checkpoint_count = self.checkpoint_manager.get_checkpoint_count()
            cache_size = self.checkpoint_manager.get_cache_size()
            cache_size_mb = cache_size / (1024 * 1024)
            
            latest = self.checkpoint_manager.get_latest_checkpoint()
            if latest:
                info_text = (
                    f"Checkpoints: {checkpoint_count}\n"
                    f"Cache size: {cache_size_mb:.2f} MB\n"
                    f"Latest: {latest['stage']} at {latest['timestamp'][:19]}"
                )
            else:
                info_text = (
                    f"Checkpoints: {checkpoint_count}\n"
                    f"Cache size: {cache_size_mb:.2f} MB"
                )
            
            if self.checkpoint_info_tag:
                dpg.set_value(self.checkpoint_info_tag, info_text)
        
        # Redraw visualization
        self._draw_cycle()
    
    def _draw_cycle(self) -> None:
        """Draw the cycle visualization."""
        if not self.drawing_tag:
            return
        
        # Clear previous drawing
        dpg.delete_item(self.drawing_tag, children_only=True)
        
        # Draw connections (arrows) - now includes PREPROCESSING stage
        connections = [
            (ProcessingStage.IDLE, ProcessingStage.PREPROCESSING),
            (ProcessingStage.PREPROCESSING, ProcessingStage.LOADING_LLM),
            (ProcessingStage.LOADING_LLM, ProcessingStage.DIAGNOSING),
            (ProcessingStage.DIAGNOSING, ProcessingStage.FIX_ATTEMPT),
            (ProcessingStage.FIX_ATTEMPT, ProcessingStage.VALIDATING),
            (ProcessingStage.VALIDATING, ProcessingStage.COMPLETE),
        ]
        
        for start_stage, end_stage in connections:
            start_pos = self._get_stage_position(start_stage)
            end_pos = self._get_stage_position(end_stage)
            
            if start_pos and end_pos:
                # Draw arrow line - ensure correct direction (left to right)
                color = (100, 149, 237) if self._is_stage_active(end_stage) else (60, 60, 80)
                # Arrow from right edge of start circle to left edge of end circle
                start_x = start_pos["x"] + 30  # Right edge of circle (radius ~25-30)
                end_x = end_pos["x"] - 30      # Left edge of circle
                dpg.draw_arrow(
                    (start_x, start_pos["y"]),
                    (end_x, end_pos["y"]),
                    color=color,
                    thickness=2,
                    size=10,
                    parent=self.drawing_tag
                )
        
        # Draw stages (circles with labels)
        for stage_info in self.stages:
            stage_id = stage_info["id"]
            x, y = stage_info["x"], stage_info["y"]
            
            # Skip ERROR node unless we're in error state
            if stage_id == ProcessingStage.ERROR and self.current_stage != ProcessingStage.ERROR:
                continue
            
            # Determine if this stage is active
            is_active = self._is_stage_active(stage_id)
            is_current = (stage_id == self.current_stage)
            
            # Choose color
            if is_current:
                color = stage_info["active_color"]
                radius = 35
            elif is_active:
                color = stage_info["active_color"]
                radius = 30
            else:
                color = stage_info["color"]
                radius = 25
            
            # Draw circle
            dpg.draw_circle(
                (x, y),
                radius=radius,
                color=color,
                fill=color,
                thickness=2 if is_current else 1,
                parent=self.drawing_tag
            )
            
            # Draw label - center text inside the circle
            label = stage_info["label"]
            # All labels are now single-line, so center both horizontally and vertically
            # Estimate text width (rough approximation: ~8 pixels per character)
            text_width = len(label) * 8
            text_x = x - (text_width // 2)
            text_y = y - 7  # Center vertically (half of ~14px font size)
            dpg.draw_text(
                (text_x, text_y),
                text=label,
                color=(236, 240, 241) if is_current or is_active else (149, 165, 166),
                size=12,
                parent=self.drawing_tag
            )
        
        # Draw error connection if in error state
        if self.current_stage == ProcessingStage.ERROR:
            diag_pos = self._get_stage_position(ProcessingStage.DIAGNOSING)
            error_pos = self._get_stage_position(ProcessingStage.ERROR)
            
            if diag_pos and error_pos:
                dpg.draw_line(
                    (diag_pos["x"], diag_pos["y"] - 30),
                    (error_pos["x"], error_pos["y"] + 30),
                    color=(231, 76, 60),
                    thickness=2,
                    parent=self.drawing_tag
                )
    
    def _get_stage_position(self, stage_id: str) -> Optional[Dict[str, int]]:
        """Get position of a stage."""
        for stage_info in self.stages:
            if stage_info["id"] == stage_id:
                return {"x": stage_info["x"], "y": stage_info["y"]}
        return None
    
    def _is_stage_active(self, stage_id: str) -> bool:
        """Check if a stage is active (reached or current)."""
        # Updated stage order to include PREPROCESSING
        stage_order = [
            ProcessingStage.IDLE,
            ProcessingStage.PREPROCESSING,
            ProcessingStage.LOADING_LLM,
            ProcessingStage.DIAGNOSING,
            ProcessingStage.FIX_ATTEMPT,
            ProcessingStage.VALIDATING,
            ProcessingStage.COMPLETE
        ]
        
        try:
            current_index = stage_order.index(self.current_stage)
            stage_index = stage_order.index(stage_id)
            return stage_index <= current_index
        except ValueError:
            # Error stage is special
            if stage_id == ProcessingStage.ERROR:
                return self.current_stage == ProcessingStage.ERROR
            return False

