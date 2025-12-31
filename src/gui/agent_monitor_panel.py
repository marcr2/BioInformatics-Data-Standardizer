"""
Agent Monitor Panel for BIDS GUI

Real-time display of LLM thinking, review, and generation processes.
Features:
- Live streaming of agent thoughts
- Phase indicators (Analyzing, Diagnosing, Generating, Executing)
- Token/progress counters
- Thought history with timestamps
"""

import dearpygui.dearpygui as dpg
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import queue


class AgentPhase(Enum):
    """Current phase of agent processing."""
    IDLE = "idle"
    LOADING = "loading"
    ANALYZING = "analyzing"
    DIAGNOSING = "diagnosing"
    THINKING = "thinking"
    GENERATING = "generating"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ThoughtEntry:
    """A single thought/log entry from the agent."""
    timestamp: datetime
    phase: AgentPhase
    content: str
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentMonitorPanel:
    """
    Real-time monitor for LLM agent activity.
    
    Features:
    - Live thought streaming
    - Phase indicators with animations
    - Token usage tracking
    - Expandable thought history
    """
    
    # Theme colors
    COLORS = {
        "background": (25, 27, 35, 255),
        "panel": (35, 37, 50, 255),
        "accent": (100, 149, 237, 255),
        "text": (220, 225, 230, 255),
        "text_dim": (130, 140, 150, 255),
        "phase_idle": (80, 90, 100, 255),
        "phase_loading": (241, 196, 15, 255),
        "phase_analyzing": (52, 152, 219, 255),
        "phase_diagnosing": (155, 89, 182, 255),
        "phase_thinking": (241, 196, 15, 255),
        "phase_generating": (46, 204, 113, 255),
        "phase_executing": (230, 126, 34, 255),
        "phase_reviewing": (52, 152, 219, 255),
        "phase_complete": (46, 204, 113, 255),
        "phase_error": (231, 76, 60, 255),
        "stream_bg": (28, 30, 40, 255),
        "token_accent": (255, 193, 7, 255),
    }
    
    PHASE_ICONS = {
        AgentPhase.IDLE: "○",
        AgentPhase.LOADING: "◌",
        AgentPhase.ANALYZING: "◎",
        AgentPhase.DIAGNOSING: "◉",
        AgentPhase.THINKING: "●",
        AgentPhase.GENERATING: "◆",
        AgentPhase.EXECUTING: "▶",
        AgentPhase.REVIEWING: "◈",
        AgentPhase.COMPLETE: "✓",
        AgentPhase.ERROR: "✗",
    }
    
    def __init__(self, scale: float = 1.0):
        """Initialize the agent monitor panel."""
        self.current_phase = AgentPhase.IDLE
        self.thoughts: List[ThoughtEntry] = []
        self.total_tokens = 0
        self.current_stream = ""
        self.scale = scale
        
        # Thread-safe update queue
        self.update_queue: queue.Queue = queue.Queue()
        
        # UI element tags
        self.phase_indicator_tag: Optional[int] = None
        self.phase_text_tag: Optional[int] = None
        self.token_counter_tag: Optional[int] = None
        self.stream_text_tag: Optional[int] = None
        self.history_container_tag: Optional[int] = None
        self.progress_tag: Optional[int] = None
        
        # Animation state
        self._animation_frame = 0
        self._is_animating = False
        
        # Callbacks
        self._on_phase_change: Optional[Callable] = None
    
    def create(self) -> None:
        """Create the agent monitor panel UI."""
        # Main container
        with dpg.group():
            # Header with phase indicator
            self._create_header()
            
            dpg.add_spacer(height=int(10 * self.scale))
            
            # Main content split
            with dpg.group(horizontal=True):
                # Left: Live stream area
                right_panel_width = int(300 * self.scale)
                with dpg.child_window(width=-right_panel_width, height=-1, border=True):
                    self._create_stream_area()
                
                # Right: Stats and history
                with dpg.child_window(width=-1, height=-1, border=True):
                    self._create_stats_area()
    
    def _create_header(self) -> None:
        """Create the header with phase indicator."""
        with dpg.group(horizontal=True):
            dpg.add_text("Agent Monitor", color=self.COLORS["accent"][:3])
            
            dpg.add_spacer(width=30)
            
            # Phase indicator
            with dpg.group(horizontal=True):
                self.phase_indicator_tag = dpg.add_text(
                    self.PHASE_ICONS[AgentPhase.IDLE],
                    color=self.COLORS["phase_idle"][:3]
                )
                dpg.add_spacer(width=5)
                self.phase_text_tag = dpg.add_text(
                    "Idle",
                    color=self.COLORS["text_dim"][:3]
                )
            
            dpg.add_spacer(width=30)
            
            # Token counter
            with dpg.group(horizontal=True):
                dpg.add_text("Tokens:", color=self.COLORS["text_dim"][:3])
                dpg.add_spacer(width=5)
                self.token_counter_tag = dpg.add_text(
                    "0",
                    color=self.COLORS["token_accent"][:3]
                )
            
            dpg.add_spacer(width=-1)
            
            # Clear button
            dpg.add_button(
                label="Clear",
                callback=self._on_clear_click,
                width=60
            )
    
    def _create_stream_area(self) -> None:
        """Create the live streaming area."""
        dpg.add_text("Live Agent Output", color=self.COLORS["accent"][:3])
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        # Progress bar
        self.progress_tag = dpg.add_progress_bar(
            default_value=0.0,
            width=-1,
            overlay="Ready"
        )
        
        dpg.add_spacer(height=10)
        
        # Streaming text area
        with dpg.child_window(height=-1, border=True):
            self.stream_text_tag = dpg.add_input_text(
                default_value=self._get_welcome_message(),
                multiline=True,
                readonly=True,
                width=-1,
                height=-1,
                tab_input=True
            )
    
    def _create_stats_area(self) -> None:
        """Create the statistics and history area."""
        # Stats section
        dpg.add_text("Statistics", color=self.COLORS["accent"][:3])
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_text("Total Tokens:", color=self.COLORS["text_dim"][:3])
                self.total_tokens_tag = dpg.add_text("0", color=self.COLORS["text"][:3])
            
            with dpg.group(horizontal=True):
                dpg.add_text("Thoughts:", color=self.COLORS["text_dim"][:3])
                self.thought_count_tag = dpg.add_text("0", color=self.COLORS["text"][:3])
            
            with dpg.group(horizontal=True):
                dpg.add_text("Current Model:", color=self.COLORS["text_dim"][:3])
                self.model_tag = dpg.add_text("--", color=self.COLORS["text"][:3])
        
        dpg.add_spacer(height=15)
        dpg.add_text("Thought History", color=self.COLORS["accent"][:3])
        dpg.add_separator()
        dpg.add_spacer(height=5)
        
        # History container with scrollable list
        with dpg.child_window(height=-1, border=True) as history_container:
            self.history_container_tag = history_container
            dpg.add_text(
                "Agent thoughts will appear here...",
                color=self.COLORS["text_dim"][:3],
                wrap=250
            )
    
    def _get_welcome_message(self) -> str:
        """Get the welcome message for the stream area."""
        return """============================================================
                    BIDS Agent Monitor                          
============================================================
  This panel shows real-time LLM activity:                      
                                                                  
  • ANALYZING  - Examining data structure                       
  • DIAGNOSING - Identifying issues                             
  • THINKING   - Reasoning about solutions                      
  • GENERATING - Creating fix scripts                           
  • EXECUTING  - Running generated code                         
  • REVIEWING  - Validating results                             
                                                                  
  Start a diagnosis or fix process to see the agent in action!  
============================================================
"""
    
    def set_phase(self, phase: AgentPhase) -> None:
        """
        Set the current processing phase.
        
        Args:
            phase: New phase to set
        """
        self.current_phase = phase
        
        # Get phase color
        color_key = f"phase_{phase.value}"
        color = self.COLORS.get(color_key, self.COLORS["text"])[:3]
        
        # Update phase indicator
        if self.phase_indicator_tag:
            dpg.set_value(self.phase_indicator_tag, self.PHASE_ICONS[phase])
            dpg.configure_item(self.phase_indicator_tag, color=color)
        
        if self.phase_text_tag:
            dpg.set_value(self.phase_text_tag, phase.value.capitalize())
            dpg.configure_item(self.phase_text_tag, color=color)
        
        # Update progress bar overlay
        if self.progress_tag:
            dpg.configure_item(self.progress_tag, overlay=phase.value.capitalize())
        
        # Trigger callback if set
        if self._on_phase_change:
            self._on_phase_change(phase)
    
    def set_progress(self, progress: float) -> None:
        """
        Set the progress bar value.
        
        Args:
            progress: Progress value between 0 and 1
        """
        if self.progress_tag:
            dpg.set_value(self.progress_tag, progress)
    
    def add_thought(
        self,
        content: str,
        phase: Optional[AgentPhase] = None,
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new thought entry.
        
        Args:
            content: The thought content
            phase: Phase when this thought occurred
            token_count: Number of tokens used
            metadata: Additional metadata
        """
        if phase is None:
            phase = self.current_phase
        
        entry = ThoughtEntry(
            timestamp=datetime.now(),
            phase=phase,
            content=content,
            token_count=token_count,
            metadata=metadata or {}
        )
        
        self.thoughts.append(entry)
        self.total_tokens += token_count
        
        # Update displays
        self._update_stream(content)
        self._update_history(entry)
        self._update_stats()
    
    def stream_text(self, text: str, phase: Optional[AgentPhase] = None) -> None:
        """
        Stream text character by character (append to current stream).
        
        Args:
            text: Text to append to stream
            phase: Optional phase to set
        """
        if phase:
            self.set_phase(phase)
        
        self.current_stream += text
        
        if self.stream_text_tag:
            dpg.set_value(self.stream_text_tag, self.current_stream)
    
    def _update_stream(self, content: str) -> None:
        """Update the stream display with new content."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        phase_icon = self.PHASE_ICONS.get(self.current_phase, "○")
        
        new_text = f"\n[{timestamp}] {phase_icon} {self.current_phase.value.upper()}\n"
        new_text += "─" * 60 + "\n"
        new_text += content + "\n"
        
        self.current_stream += new_text
        
        # Keep only last 5000 chars to prevent slowdown
        if len(self.current_stream) > 5000:
            self.current_stream = self.current_stream[-5000:]
        
        if self.stream_text_tag:
            dpg.set_value(self.stream_text_tag, self.current_stream)
    
    def _update_history(self, entry: ThoughtEntry) -> None:
        """Add an entry to the history display."""
        if not self.history_container_tag:
            return
        
        # Clear welcome message on first entry
        if len(self.thoughts) == 1:
            dpg.delete_item(self.history_container_tag, children_only=True)
        
        # Get phase color
        color_key = f"phase_{entry.phase.value}"
        color = self.COLORS.get(color_key, self.COLORS["text"])[:3]
        
        # Create history entry
        with dpg.group(parent=self.history_container_tag):
            with dpg.group(horizontal=True):
                # Timestamp
                dpg.add_text(
                    entry.timestamp.strftime("%H:%M:%S"),
                    color=self.COLORS["text_dim"][:3]
                )
                # Phase icon
                dpg.add_text(
                    self.PHASE_ICONS.get(entry.phase, "○"),
                    color=color
                )
            
            # Content preview
            preview = entry.content[:100] + "..." if len(entry.content) > 100 else entry.content
            dpg.add_text(
                preview,
                wrap=250,
                color=self.COLORS["text_dim"][:3]
            )
            
            # Token count if present
            if entry.token_count > 0:
                dpg.add_text(
                    f"[{entry.token_count} tokens]",
                    color=self.COLORS["token_accent"][:3]
                )
            
            dpg.add_spacer(height=5)
            dpg.add_separator()
    
    def _update_stats(self) -> None:
        """Update the statistics display."""
        if hasattr(self, 'total_tokens_tag'):
            dpg.set_value(self.total_tokens_tag, str(self.total_tokens))
        
        if hasattr(self, 'thought_count_tag'):
            dpg.set_value(self.thought_count_tag, str(len(self.thoughts)))
    
    def set_model(self, model_name: str) -> None:
        """
        Set the current model name display.
        
        Args:
            model_name: Name of the current model
        """
        if hasattr(self, 'model_tag'):
            # Truncate long model names
            display_name = model_name.split("/")[-1] if "/" in model_name else model_name
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."
            dpg.set_value(self.model_tag, display_name)
    
    def _on_clear_click(self) -> None:
        """Handle clear button click."""
        self.thoughts = []
        self.total_tokens = 0
        self.current_stream = ""
        
        # Reset displays
        if self.stream_text_tag:
            dpg.set_value(self.stream_text_tag, self._get_welcome_message())
        
        if self.history_container_tag:
            dpg.delete_item(self.history_container_tag, children_only=True)
            dpg.add_text(
                "Agent thoughts will appear here...",
                color=self.COLORS["text_dim"][:3],
                wrap=250,
                parent=self.history_container_tag
            )
        
        self._update_stats()
        self.set_phase(AgentPhase.IDLE)
        self.set_progress(0.0)
    
    def process_updates(self) -> None:
        """Process queued updates from background threads."""
        try:
            while True:
                update_type, *args = self.update_queue.get_nowait()
                
                if update_type == 'phase':
                    self.set_phase(args[0])
                elif update_type == 'progress':
                    self.set_progress(args[0])
                elif update_type == 'thought':
                    self.add_thought(*args)
                elif update_type == 'stream':
                    self.stream_text(*args)
                elif update_type == 'model':
                    self.set_model(args[0])
                    
        except queue.Empty:
            pass
    
    # Thread-safe methods for background threads
    def queue_phase(self, phase: AgentPhase) -> None:
        """Queue a phase change from a background thread."""
        self.update_queue.put(('phase', phase))
    
    def queue_progress(self, progress: float) -> None:
        """Queue a progress update from a background thread."""
        self.update_queue.put(('progress', progress))
    
    def queue_thought(
        self,
        content: str,
        phase: Optional[AgentPhase] = None,
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Queue a thought from a background thread."""
        self.update_queue.put(('thought', content, phase, token_count, metadata))
    
    def queue_stream(self, text: str, phase: Optional[AgentPhase] = None) -> None:
        """Queue streamed text from a background thread."""
        self.update_queue.put(('stream', text, phase))
    
    def queue_model(self, model_name: str) -> None:
        """Queue a model name update from a background thread."""
        self.update_queue.put(('model', model_name))


# Global monitor instance for easy access
_global_monitor: Optional[AgentMonitorPanel] = None


def get_agent_monitor() -> Optional[AgentMonitorPanel]:
    """Get the global agent monitor instance."""
    global _global_monitor
    return _global_monitor


def set_agent_monitor(monitor: AgentMonitorPanel) -> None:
    """Set the global agent monitor instance."""
    global _global_monitor
    _global_monitor = monitor


def notify_phase(phase: AgentPhase) -> None:
    """Notify the monitor of a phase change (thread-safe)."""
    monitor = get_agent_monitor()
    if monitor:
        monitor.queue_phase(phase)


def notify_progress(progress: float) -> None:
    """Notify the monitor of progress (thread-safe)."""
    monitor = get_agent_monitor()
    if monitor:
        monitor.queue_progress(progress)


def notify_thought(
    content: str,
    phase: Optional[AgentPhase] = None,
    token_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Notify the monitor of a new thought (thread-safe)."""
    monitor = get_agent_monitor()
    if monitor:
        monitor.queue_thought(content, phase, token_count, metadata)


def notify_model(model_name: str) -> None:
    """Notify the monitor of the current model (thread-safe)."""
    monitor = get_agent_monitor()
    if monitor:
        monitor.queue_model(model_name)

