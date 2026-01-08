"""
Process Panel for BIDS GUI

Handles diagnostic and fix processing with status display.
"""

import dearpygui.dearpygui as dpg
from typing import Optional, Callable, Dict, Any, List
import pandas as pd
import threading
from datetime import datetime
from pathlib import Path
import json
import csv
import io

from ..utils.logger import get_logger, info, debug, warning, error

# Lazy imports
DiagnosticAgent = None
FixingAgent = None
AgentOrchestrator = None


def get_agent_classes():
    """Lazy import of agent classes."""
    global DiagnosticAgent, FixingAgent, AgentOrchestrator
    if DiagnosticAgent is None:
        from ..agents import (
            DiagnosticAgent as DA,
            FixingAgent as FA,
            AgentOrchestrator as AO
        )
        DiagnosticAgent = DA
        FixingAgent = FA
        AgentOrchestrator = AO
    return DiagnosticAgent, FixingAgent, AgentOrchestrator


class ProcessPanel:
    """
    Processing status and control panel.
    
    Features:
    - Run diagnosis
    - Run full fix process
    - Display logs and status
    - Show issues found
    - Display generated fix scripts
    """
    
    def __init__(self, on_process_complete: Optional[Callable] = None, main_app=None, scale: float = 1.0):
        """
        Initialize ProcessPanel.
        
        Args:
            on_process_complete: Callback when processing completes
            main_app: Reference to main app for button callbacks
            scale: Scaling factor for UI elements
        """
        self.on_process_complete = on_process_complete
        self.main_app = main_app  # Reference to main app
        self.scale = scale
        self.orchestrator = None
        self.is_processing = False
        self.logs: List[str] = []
        
        # Data references (set by main app)
        self.current_df: Optional[pd.DataFrame] = None
        self.current_schema = None
        
        # Checkpoint manager (will be set by main app if available)
        self.checkpoint_manager = None
        
        # Store last diagnosis result for reuse
        self.last_diagnosis = None
        
        # UI tags
        self.log_text_tag: Optional[int] = None
        self.issues_text_tag: Optional[int] = None  # Copy-pastable issues text
        self.progress_tag: Optional[int] = None
        self.status_text_tag: Optional[int] = None
        self.load_prev_diag_checkbox: Optional[int] = None
        
        # Context input UI tags
        self.context_input_tag: Optional[int] = None
        self.context_diagnosis_checkbox: Optional[int] = None
        self.context_fixing_checkbox: Optional[int] = None
        self.token_remaining_tag: Optional[int] = None
        self.token_used_tag: Optional[int] = None
        
        # Uploaded files storage: {filename: {"path": str, "content": str, "type": str}}
        self.uploaded_files: Dict[str, Dict[str, Any]] = {}
        self.uploaded_files_list_tag: Optional[int] = None
        self.uploaded_files_group_tag: Optional[int] = None
    
    def _scale(self, size: int) -> int:
        """Scale a size value."""
        return int(size * self.scale)
    
    def set_checkpoint_manager(self, checkpoint_manager):
        """Set checkpoint manager and update orchestrator if it exists."""
        self.checkpoint_manager = checkpoint_manager
        # Update orchestrator if it exists
        if self.orchestrator:
            self.orchestrator.checkpoint_manager = checkpoint_manager
    
    def create(self) -> None:
        """Create the process panel UI."""
        # Initialize orchestrator
        _, _, AO = get_agent_classes()
        try:
            self.orchestrator = AO(
                vector_store_path="data/vector_store",
                schemas_dir="schemas",
                checkpoint_manager=self.checkpoint_manager
            )
        except Exception as e:
            self.orchestrator = None
            self._log(f"Warning: Could not initialize orchestrator: {e}")
        
        # Status section
        with dpg.group(horizontal=True):
            dpg.add_text("Status:", color=(100, 149, 237))
            self.status_text_tag = dpg.add_text("Ready", color=(149, 165, 166))
        
        dpg.add_spacer(height=self._scale(5))
        
        self.progress_tag = dpg.add_progress_bar(
            default_value=0.0,
            width=-1,
            overlay="Idle"
        )
        
        dpg.add_spacer(height=self._scale(10))
        dpg.add_separator()
        dpg.add_spacer(height=self._scale(10))
        
        # Tabs for different views
        with dpg.tab_bar():
            # Issues tab - using input_text for copy-paste support
            with dpg.tab(label="Issues Found"):
                with dpg.child_window(height=self._scale(250), border=True):
                    self.issues_text_tag = dpg.add_input_text(
                        default_value="Run diagnosis to see issues...",
                        multiline=True,
                        readonly=True,
                        width=-1,
                        height=-1,
                        enabled=True  # Enable for text selection and copy
                    )
            
            # Log tab
            with dpg.tab(label="Processing Log"):
                with dpg.child_window(height=self._scale(250), border=True):
                    # Use input_text with readonly for selectable text
                    self.log_text_tag = dpg.add_input_text(
                        default_value="Processing log will appear here...",
                        multiline=True,
                        readonly=True,
                        width=-1,
                        height=-1,
                        enabled=False  # Disable editing but allow selection
                    )
        
        dpg.add_spacer(height=self._scale(10))
        dpg.add_separator()
        dpg.add_spacer(height=self._scale(10))
        
        # Additional Context section
        dpg.add_text("Additional Context (Optional)", color=(100, 149, 237))
        dpg.add_spacer(height=self._scale(5))
        
        with dpg.group(horizontal=True):
            self.context_diagnosis_checkbox = dpg.add_checkbox(
                label="Apply to Diagnosis",
                default_value=True
            )
            dpg.add_spacer(width=20)
            self.context_fixing_checkbox = dpg.add_checkbox(
                label="Apply to Fixing",
                default_value=True
            )
        
        dpg.add_spacer(height=self._scale(5))
        
        # File upload button
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Upload File",
                callback=self._on_upload_file_click,
                width=self._scale(100)
            )
            dpg.add_text(
                "Upload .txt, .json, .csv, or other text files",
                color=(149, 165, 166),
                wrap=-1
            )
        
        dpg.add_spacer(height=self._scale(5))
        
        # Uploaded files list
        dpg.add_text("Uploaded Files:", color=(149, 165, 166))
        with dpg.child_window(
            height=self._scale(80),
            border=True,
            tag="uploaded_files_container"
        ):
            # Use a group that we can rebuild
            self.uploaded_files_group_tag = dpg.add_group(tag="uploaded_files_group")
            self.uploaded_files_list_tag = dpg.add_text(
                "No files uploaded",
                color=(149, 165, 166),
                parent=self.uploaded_files_group_tag
            )
        
        dpg.add_spacer(height=self._scale(5))
        
        self.context_input_tag = dpg.add_input_text(
            default_value="",
            multiline=True,
            width=-1,
            height=self._scale(80),
            hint="Enter additional context for the LLM (e.g., domain-specific terminology, expected formats, special handling instructions). You can reference uploaded files by name (e.g., 'see config.json' or 'use the mapping in data.csv')...",
            callback=self._on_context_text_changed
        )
        
        dpg.add_spacer(height=self._scale(5))
        
        # Token estimation display
        with dpg.group(horizontal=True):
            self.token_remaining_tag = dpg.add_text(
                "Remaining: -- tokens",
                color=(149, 165, 166)
            )
            dpg.add_spacer(width=20)
            self.token_used_tag = dpg.add_text(
                "Used: -- / -- tokens",
                color=(149, 165, 166)
            )
        
        dpg.add_spacer(height=self._scale(10))
        dpg.add_separator()
        dpg.add_spacer(height=self._scale(10))
        
        # Create file dialog for context file uploads
        self._create_context_file_dialog()
        
        # Action buttons - styled like header buttons and call main app handlers
        with dpg.group(horizontal=True):
            # Get themes from main app if available
            success_theme = None
            if self.main_app and hasattr(self.main_app, 'success_theme'):
                success_theme = self.main_app.success_theme
            
            # "Run Full Process" - Green (success theme) - matches "Fix & Export"
            self.btn_process = dpg.add_button(
                label="Run Full Process",
                callback=self._on_process_click,
                width=150
            )
            if success_theme:
                dpg.bind_item_theme(self.btn_process, success_theme)
            
            dpg.add_spacer(width=5)
            
            # Checkbox to load previous diagnosis
            self.load_prev_diag_checkbox = dpg.add_checkbox(
                label="Use Previous Diagnosis",
                default_value=True
            )
            
            dpg.add_spacer(width=10)
            
            # "Diagnose" - Blue (accent color) - matches header button
            # Uses default theme which is already blue (accent color)
            self.btn_diagnose = dpg.add_button(
                label="Diagnose",
                callback=self._on_diagnose_click,
                width=140
            )
            # No theme binding needed - default button uses accent blue color
            
            dpg.add_spacer(width=-1)
            dpg.add_button(
                label="Clear Log",
                callback=self._clear_log,
                width=100
            )
    
    def _log(self, message: str, level: str = "info") -> None:
        """Add a log message to both GUI and console logger."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # Also output to console logger
        if level == "debug":
            debug(message, context="ProcessPanel")
        elif level == "warning":
            warning(message, context="ProcessPanel")
        elif level == "error":
            error(message, context="ProcessPanel")
        else:
            info(message, context="ProcessPanel")
        
        # Update log display
        if self.log_text_tag:
            log_text = "\n".join(self.logs[-50:])  # Keep last 50 entries
            dpg.set_value(self.log_text_tag, log_text)
            # Re-enable to allow text selection
            dpg.configure_item(self.log_text_tag, enabled=True)
    
    def _clear_log(self) -> None:
        """Clear the log."""
        self.logs = []
    
    def _create_context_file_dialog(self) -> None:
        """Create the file dialog for context file uploads."""
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_context_file_dialog_ok,
            cancel_callback=self._on_context_file_dialog_cancel,
            width=700,
            height=400,
            modal=True,
            tag="context_file_dialog"
        ):
            # Add file type filters for text-based files
            dpg.add_file_extension(".*", color=(255, 255, 255))
            dpg.add_file_extension(".txt", color=(0, 255, 0))
            dpg.add_file_extension(".json", color=(0, 200, 255))
            dpg.add_file_extension(".csv", color=(255, 200, 0))
            dpg.add_file_extension(".tsv", color=(255, 200, 0))
            dpg.add_file_extension(".md", color=(200, 200, 255))
            dpg.add_file_extension(".yaml", color=(255, 100, 100))
            dpg.add_file_extension(".yml", color=(255, 100, 100))
            dpg.add_file_extension(".xml", color=(255, 150, 100))
    
    def _on_upload_file_click(self) -> None:
        """Handle upload file button click."""
        if not dpg.does_item_exist("context_file_dialog"):
            self._create_context_file_dialog()
        
        # Show the dialog
        inputs_dir = Path("inputs").absolute()
        inputs_dir.mkdir(exist_ok=True)
        
        try:
            dpg.configure_item("context_file_dialog", default_path=str(inputs_dir))
        except:
            pass
        
        dpg.show_item("context_file_dialog")
    
    def _on_context_file_dialog_ok(self, sender, app_data) -> None:
        """Handle context file dialog OK callback."""
        file_path = app_data.get("file_path_name", "")
        
        if not file_path:
            return
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self._log(f"Error: File not found: {file_path}")
                return
            
            # Read and parse file
            file_content = self._read_context_file(file_path_obj)
            if file_content is None:
                self._log(f"Error: Could not read file: {file_path}")
                return
            
            # Store file
            filename = file_path_obj.name
            file_ext = file_path_obj.suffix.lower()
            
            self.uploaded_files[filename] = {
                "path": str(file_path_obj),
                "content": file_content,
                "type": file_ext,
                "size": file_path_obj.stat().st_size
            }
            
            self._update_uploaded_files_display()
            self._update_token_estimation()
            self._log(f"Added context file: {filename}")
            
        except Exception as e:
            self._log(f"Error loading file: {e}")
            error(f"Error loading context file: {e}", context="ProcessPanel")
    
    def _on_context_file_dialog_cancel(self) -> None:
        """Handle context file dialog cancel."""
        pass
    
    def _read_context_file(self, file_path: Path) -> Optional[str]:
        """
        Read and format a context file based on its type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Formatted file content as string, or None if error
        """
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == ".json":
                # Read and pretty-print JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2, ensure_ascii=False)
            
            elif file_ext in [".csv", ".tsv"]:
                # Read CSV and format as readable text
                delimiter = "\t" if file_ext == ".tsv" else ","
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    rows = list(reader)
                    
                    if not rows:
                        return "Empty CSV file"
                    
                    # Format as a readable table-like structure
                    output = []
                    output.append("CSV Data:")
                    output.append("=" * 50)
                    
                    # Show headers
                    if reader.fieldnames:
                        output.append("Columns: " + ", ".join(reader.fieldnames))
                        output.append("")
                    
                    # Show first few rows (limit to avoid huge context)
                    max_rows = 20
                    for i, row in enumerate(rows[:max_rows]):
                        output.append(f"Row {i+1}:")
                        for key, value in row.items():
                            output.append(f"  {key}: {value}")
                        output.append("")
                    
                    if len(rows) > max_rows:
                        output.append(f"... ({len(rows) - max_rows} more rows)")
                    
                    return "\n".join(output)
            
            elif file_ext in [".yaml", ".yml"]:
                # Read YAML as text (could use yaml library if available)
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            else:
                # Read as plain text
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    # Limit size to avoid huge context (first 50KB)
                    max_size = 50000
                    if len(content) > max_size:
                        content = content[:max_size] + f"\n\n... (file truncated, {len(content)} total chars)"
                    return content
        
        except json.JSONDecodeError as e:
            warning(f"Invalid JSON in {file_path.name}: {e}", context="ProcessPanel")
            return None
        except Exception as e:
            error(f"Error reading file {file_path.name}: {e}", context="ProcessPanel")
            return None
    
    def _update_uploaded_files_display(self) -> None:
        """Update the uploaded files display."""
        if not hasattr(self, 'uploaded_files_group_tag') or not dpg.does_item_exist(self.uploaded_files_group_tag):
            return
        
        # Clear existing items
        dpg.delete_item(self.uploaded_files_group_tag, children_only=True)
        
        if not self.uploaded_files:
            dpg.add_text(
                "No files uploaded",
                color=(149, 165, 166),
                parent=self.uploaded_files_group_tag
            )
            return
        
        # Add each file with a remove button
        for filename, file_info in self.uploaded_files.items():
            size_kb = file_info["size"] / 1024
            file_type = file_info["type"]
            
            with dpg.group(horizontal=True, parent=self.uploaded_files_group_tag):
                # File icon/indicator based on type
                if file_type == ".json":
                    icon_color = (0, 200, 255)
                elif file_type in [".csv", ".tsv"]:
                    icon_color = (255, 200, 0)
                elif file_type == ".txt":
                    icon_color = (0, 255, 0)
                else:
                    icon_color = (200, 200, 200)
                
                dpg.add_text("•", color=icon_color)
                dpg.add_text(
                    f"{filename} ({size_kb:.1f} KB)",
                    color=(149, 165, 166)
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Remove",
                    callback=lambda s, a, f=filename: self._remove_uploaded_file(f),
                    width=60,
                    height=20
                )
    
    def _remove_uploaded_file(self, filename: str) -> None:
        """Remove an uploaded file."""
        if filename in self.uploaded_files:
            del self.uploaded_files[filename]
            self._update_uploaded_files_display()
            self._update_token_estimation()
            self._log(f"Removed context file: {filename}")
    
    def _on_context_text_changed(self, sender, app_data) -> None:
        """Handle context text input changes."""
        self._update_token_estimation()
    
    def _update_token_estimation(self) -> None:
        """Update token estimation display based on current context and data."""
        try:
            from ..utils.preferences import get_preferences
            prefs = get_preferences()
            max_tokens = prefs.get_max_tokens()
            
            # Get current context (including files)
            context_text = self._build_full_context(include_files=True) or ""
            
            # Estimate tokens for context (rough estimate: ~4 chars per token)
            context_tokens = len(context_text) // 4
            
            # Estimate base tokens (for prompt template, schema, etc.)
            # This is a rough estimate - actual will vary by data size
            base_tokens = 2000  # Approximate base prompt size
            if self.current_df is not None:
                # Add estimate for DataFrame sample
                sample_size = min(10, len(self.current_df))
                base_tokens += sample_size * len(self.current_df.columns) * 10
            
            used_tokens = base_tokens + context_tokens
            remaining_tokens = max(0, max_tokens - used_tokens)
            
            # Update display
            if self.token_remaining_tag:
                dpg.set_value(self.token_remaining_tag, f"Remaining: ~{remaining_tokens} tokens")
            if self.token_used_tag:
                dpg.set_value(self.token_used_tag, f"Used: ~{used_tokens} / {max_tokens} tokens")
            
        except Exception as e:
            # Silently handle errors in token estimation
            if self.token_remaining_tag:
                dpg.set_value(self.token_remaining_tag, "Remaining: -- tokens")
            if self.token_used_tag:
                dpg.set_value(self.token_used_tag, "Used: -- / -- tokens")
    
    def _build_full_context(self, include_files: bool = True) -> Optional[str]:
        """
        Build full context including text and uploaded files.
        
        Args:
            include_files: Whether to include uploaded file contents
            
        Returns:
            Combined context string or None if empty
        """
        # Get text context
        text_context = ""
        if self.context_input_tag:
            text_context = dpg.get_value(self.context_input_tag) or ""
        
        # Get file contents
        file_contexts = []
        if include_files and self.uploaded_files:
            for filename, file_info in self.uploaded_files.items():
                file_content = file_info["content"]
                file_type = file_info["type"]
                
                # Format file content with clear markers
                file_section = f"\n\n--- UPLOADED FILE: {filename} ({file_type}) ---\n"
                file_section += file_content
                file_section += f"\n--- END OF FILE: {filename} ---\n"
                file_contexts.append(file_section)
        
        # Combine text and files
        if not text_context and not file_contexts:
            return None
        
        # Build combined context
        combined = []
        
        if text_context:
            combined.append("USER PROVIDED CONTEXT:")
            combined.append(text_context)
        
        if file_contexts:
            if text_context:
                combined.append("\n\nUPLOADED FILES:")
            else:
                combined.append("UPLOADED FILES:")
            
            # Check if text context references any files
            referenced_files = []
            if text_context:
                for filename in self.uploaded_files.keys():
                    # Check if filename (with or without extension) is mentioned
                    filename_base = Path(filename).stem
                    if filename in text_context or filename_base in text_context:
                        referenced_files.append(filename)
            
            if referenced_files:
                combined.append(f"\nNote: The following files are referenced in your context: {', '.join(referenced_files)}")
                combined.append("Their contents are included below for reference.\n")
            
            combined.extend(file_contexts)
        
        result = "\n".join(combined)
        return result if result.strip() else None
    
    def get_context_for_diagnosis(self) -> Optional[str]:
        """Get context text if diagnosis checkbox is checked."""
        if self.context_diagnosis_checkbox and dpg.get_value(self.context_diagnosis_checkbox):
            return self._build_full_context(include_files=True)
        return None
    
    def get_context_for_fixing(self) -> Optional[str]:
        """Get context text if fixing checkbox is checked."""
        if self.context_fixing_checkbox and dpg.get_value(self.context_fixing_checkbox):
            return self._build_full_context(include_files=True)
        return None
        if self.log_text_tag:
            dpg.set_value(self.log_text_tag, "Log cleared")
    
    def _update_status(self, status: str, progress: float = None) -> None:
        """Update status display."""
        if self.status_text_tag:
            dpg.set_value(self.status_text_tag, status)
        
        if progress is not None and self.progress_tag:
            dpg.set_value(self.progress_tag, progress)
            dpg.configure_item(self.progress_tag, overlay=status)
    
    def _display_issues(self, issues: List[Dict[str, Any]]) -> None:
        """Display found issues as copy-pastable text."""
        if not issues:
            if self.issues_text_tag:
                dpg.set_value(self.issues_text_tag, "[OK] No issues found! Data is valid.")
            return
        
        # Format issues as text for copy-paste
        lines = []
        lines.append(f"Found {len(issues)} issue(s):")
        lines.append("=" * 50)
        lines.append("")
        
        for i, issue in enumerate(issues, 1):
            severity = issue.get("severity", "info").upper()
            column = issue.get("column", "Unknown")
            description = issue.get("description", "No description")
            suggested_fix = issue.get("suggested_fix", "")
            
            lines.append(f"Issue #{i}")
            lines.append(f"  Severity: [{severity}]")
            lines.append(f"  Column:   {column}")
            lines.append(f"  Problem:  {description}")
            if suggested_fix:
                lines.append(f"  Fix:      {suggested_fix}")
            lines.append("-" * 50)
            lines.append("")
        
        issues_text = "\n".join(lines)
        
        if self.issues_text_tag:
            dpg.set_value(self.issues_text_tag, issues_text)
    
    def _on_diagnose_click(self) -> None:
        """Handle diagnose button click - delegates to main app."""
        if self.main_app:
            # Call main app's handler which updates both locations
            self.main_app._on_diagnose_click()
        else:
            # Fallback if no main app reference
            if self.current_df is None:
                self._log("Error: No data loaded. Please load a file first.")
                return
            
            if self.current_schema is None:
                self._log("Error: No schema selected. Please select a schema first.")
                return
            
            self.run_diagnosis(self.current_df, self.current_schema)
    
    def _on_process_click(self) -> None:
        """Handle process button click - delegates to main app."""
        if self.main_app:
            # Call main app's handler which updates both locations
            self.main_app._on_fix_click()
        else:
            # Fallback if no main app reference
            if self.current_df is None:
                self._log("Error: No data loaded. Please load a file first.")
                return
            
            if self.current_schema is None:
                self._log("Error: No schema selected. Please select a schema first.")
                return
            
            self.run_full_process(self.current_df, self.current_schema)
    
    def set_data(self, df: pd.DataFrame, schema) -> None:
        """
        Set the current data and schema for processing.
        
        Args:
            df: Current DataFrame
            schema: Current schema
        """
        self.current_df = df
        self.current_schema = schema
    
    def run_diagnosis(self, df: pd.DataFrame, schema) -> None:
        """
        Run diagnosis on a DataFrame.
        
        Args:
            df: DataFrame to diagnose
            schema: Target schema
        """
        if self.is_processing:
            self._log("Already processing...")
            return
        
        self.is_processing = True
        # Status already updated by main app handler
        # self._update_status("Diagnosing...", 0.3)
        # self._log("Starting diagnosis...")
        
        # Run in background thread
        thread = threading.Thread(
            target=self._diagnosis_thread,
            args=(df, schema)
        )
        thread.daemon = True
        thread.start()
    
    def _diagnosis_thread(self, df: pd.DataFrame, schema) -> None:
        """Background thread for diagnosis."""
        try:
            DA, _, _ = get_agent_classes()
            
            if self.orchestrator:
                # Ensure LLM is loaded (lazy loading)
                self._log("Loading LLM model (first time may take 10-30 seconds)...")
                self._update_status("Loading LLM...", 0.1)
                self.orchestrator._ensure_llm_loaded()
                agent = self.orchestrator.diagnostic_agent
            else:
                agent = DA()
            
            self._log("Analyzing data structure...")
            self._update_status("Analyzing...", 0.5)
            
            # Save checkpoint before diagnosis
            from ..utils.checkpoint_manager import ProcessingStage
            if self.checkpoint_manager:
                self.checkpoint_manager.create_checkpoint(
                    ProcessingStage.DIAGNOSING,
                    df=df,
                    metadata={"action": "before_diagnosis"}
                )
                if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                    self.main_app.cycle_viz_panel.update_stage(ProcessingStage.DIAGNOSING)
                    # Refresh data state panel
                    if hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                        self.main_app.data_state_panel._load_latest()
            
            # Get additional context for diagnosis
            diagnosis_context = self.get_context_for_diagnosis()
            if diagnosis_context:
                self._log(f"Using additional context ({len(diagnosis_context)} chars)")
            
            # Run diagnosis
            result = agent.diagnose(
                df,
                schema,
                self.orchestrator.schema_manager if self.orchestrator else None,
                additional_context=diagnosis_context
            )
            
            self._log(f"Diagnosis complete. Valid: {result.is_valid}")
            self._log(f"Quality score: {result.quality_score:.2f}")
            self._log(f"Issues found: {len(result.issues)}")
            
            # Log warning if diagnosis failed
            if not result.is_valid or result.quality_score == 0.0:
                if result.summary and "failed" in result.summary.lower():
                    self._log(f"⚠️ Warning: {result.summary}")
                    self._log("The LLM may have returned an invalid response. Check the debug logs for details.")
                    
                    # Check if there are parse errors
                    parse_errors = [issue for issue in result.issues if issue.get('issue_type') == 'parse_error']
                    if parse_errors:
                        self._log(f"Found {len(parse_errors)} parse error(s) - LLM response may not have been valid JSON")
            
            # Store diagnosis for potential reuse in full processing
            self.last_diagnosis = result
            
            # Save checkpoint after diagnosis
            if self.checkpoint_manager:
                self.checkpoint_manager.create_checkpoint(
                    ProcessingStage.DIAGNOSING,
                    df=df,
                    metadata={
                        "action": "after_diagnosis",
                        "is_valid": result.is_valid,
                        "quality_score": result.quality_score,
                        "issues_count": len(result.issues)
                    }
                )
                # Refresh data state panel
                if self.main_app and hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                    self.main_app.data_state_panel._load_latest()
            
            # Update UI in both locations
            self._update_status("Diagnosis complete", 1.0)
            self._display_issues(result.issues)
            
            # Update main app status bar
            if self.main_app:
                dpg.set_value(self.main_app.status_text, "Diagnosis complete")
                dpg.set_value(self.main_app.progress_bar, 1.0)
                dpg.configure_item(self.main_app.progress_bar, overlay="Complete")
            
        except Exception as e:
            self._log(f"Diagnosis error: {str(e)}")
            self._update_status("Error", 0.0)
            # Update main app status bar
            if self.main_app:
                dpg.set_value(self.main_app.status_text, f"Diagnosis error: {str(e)[:50]}")
                dpg.set_value(self.main_app.progress_bar, 0.0)
                dpg.configure_item(self.main_app.progress_bar, overlay="Error")
        
        finally:
            self.is_processing = False
    
    def run_full_process(self, df: pd.DataFrame, schema) -> None:
        """
        Run full diagnosis and fix process.
        
        Args:
            df: DataFrame to process
            schema: Target schema
        """
        if self.is_processing:
            self._log("Already processing...")
            return
        
        self.is_processing = True
        # Status already updated by main app handler
        # self._update_status("Processing...", 0.1)
        # self._log("Starting full process...")
        
        # Run in background thread
        thread = threading.Thread(
            target=self._full_process_thread,
            args=(df, schema)
        )
        thread.daemon = True
        thread.start()
    
    def _full_process_thread(self, df: pd.DataFrame, schema) -> None:
        """Background thread for full processing."""
        result = {"success": False, "final_df": None}
        
        try:
            if not self.orchestrator:
                raise Exception("Orchestrator not initialized")
            
            # Import ProcessingStage for checkpoints
            from ..utils.checkpoint_manager import ProcessingStage
            
            # Get preferences
            from ..utils.preferences import get_preferences
            from ..utils.llm_check import is_llm_available
            prefs = get_preferences()
            auto_fix = prefs.get("auto_fix", True)
            max_attempts = prefs.get("max_fix_attempts", 3)
            processing_mode = prefs.get_processing_mode()
            llm_available = is_llm_available()
            
            # Log processing mode
            mode_names = {
                "auto": "Auto (use LLM if available)",
                "rules_only": "Rules-based only (no LLM)",
                "llm_required": "LLM required"
            }
            self._log(f"Processing mode: {mode_names.get(processing_mode, processing_mode)}")
            if processing_mode == "rules_only":
                self._log("Running in rules-based mode - LLM features disabled")
            elif not llm_available:
                if processing_mode == "llm_required":
                    self._log("ERROR: LLM required but not installed. Please install LLM support from Preferences.")
                    self._update_status("Error: LLM required but not available", 0.0)
                    self.is_processing = False
                    return
                else:
                    self._log("LLM not available - running in rules-based mode")
            
            # Check if we should use previous diagnosis
            use_prev_diagnosis = (
                self.load_prev_diag_checkbox and 
                dpg.get_value(self.load_prev_diag_checkbox) and 
                self.last_diagnosis is not None
            )
            if use_prev_diagnosis:
                self._log("Using previous diagnosis result...")
            
            # Get additional context for LLM
            diagnosis_context = self.get_context_for_diagnosis()
            fixing_context = self.get_context_for_fixing()
            if diagnosis_context:
                self._log(f"Using diagnosis context ({len(diagnosis_context)} chars)")
            if fixing_context:
                self._log(f"Using fixing context ({len(fixing_context)} chars)")
            
            # Phase 1: Rules-Based Preprocessing (BEFORE loading LLM)
            self._log("Phase 1: Rules-based preprocessing...")
            self._update_status("Preprocessing...", 0.05)
            
            # Update visualization - orchestrator will create checkpoints
            if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                self.main_app.cycle_viz_panel.update_stage(ProcessingStage.PREPROCESSING)
            
            # Run orchestrator with preferences
            # Note: The orchestrator now handles preprocessing internally and creates checkpoints at each stage
            self._log("Running orchestrated pipeline (preprocess -> diagnose -> fix)...")
            try:
                process_result = self.orchestrator.process(
                    df,
                    schema_name=schema.name if hasattr(schema, 'name') else "IPA Standard",
                    auto_fix=auto_fix,
                    max_attempts=max_attempts,
                    existing_diagnosis=self.last_diagnosis if use_prev_diagnosis else None,
                    diagnosis_context=diagnosis_context,
                    fixing_context=fixing_context
                )
            except Exception as e:
                error_msg = str(e)
                if "LLM is required" in error_msg or "LLM required" in error_msg:
                    self._log(f"ERROR: {error_msg}")
                    self._log("Please install LLM support from Preferences -> Install LLM")
                    self._update_status("Error: LLM required but not available", 0.0)
                    self.is_processing = False
                    return
                else:
                    raise
            
            # Check for error in result
            if "error" in process_result:
                error_msg = process_result["error"]
                if "LLM is required" in error_msg or "LLM required" in error_msg:
                    self._log(f"ERROR: {error_msg}")
                    self._log("Please install LLM support from Preferences -> Install LLM")
                    self._update_status("Error: LLM required but not available", 0.0)
                    self.is_processing = False
                    return
            
            # Log preprocessing results if available
            preprocessing = process_result.get("preprocessing", {})
            if preprocessing:
                changes = preprocessing.get("changes_made", [])
                remaining = preprocessing.get("issues_remaining", 0)
                if changes:
                    self._log(f"Preprocessing applied {len(changes)} automatic fix(es)")
                    for change in changes[:5]:  # Show first 5 changes
                        self._log(f"  - {change}")
                    if len(changes) > 5:
                        self._log(f"  ... and {len(changes) - 5} more")
                if remaining > 0:
                    if llm_available and processing_mode != "rules_only":
                        self._log(f"Preprocessing: {remaining} issue(s) remain for LLM")
                    else:
                        self._log(f"Preprocessing: {remaining} issue(s) remain (LLM required for complex fixes)")
                else:
                    self._log("Preprocessing resolved all issues!")
            
            # Refresh data state panel after orchestrator creates checkpoints
            if self.main_app and hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                self.main_app.data_state_panel._load_latest()
            
            # Update visualization for LLM loading phase
            if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                self.main_app.cycle_viz_panel.update_stage(ProcessingStage.LOADING_LLM)
            
            # Update status for diagnosis phase
            self._update_status("Diagnosing...", 0.25)
            
            # Update visualization - orchestrator will create checkpoints
            if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                self.main_app.cycle_viz_panel.update_stage(ProcessingStage.DIAGNOSING)
            
            # Refresh data state panel after diagnosis checkpoint
            if self.main_app and hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                self.main_app.data_state_panel._load_latest()
            
            # Log diagnosis results and update Issues tab
            diag = process_result.get("diagnosis", {})
            self._log(f"Quality score: {diag.get('quality_score', 0):.2f}")
            self._log(f"Issues found: {diag.get('issues_count', 0)}")
            
            # Update Issues Found tab with diagnosis results
            issues = diag.get("issues", [])
            self._display_issues(issues)
            
            # Log fix attempts - orchestrator creates checkpoints
            attempts = process_result.get("fix_attempts", [])
            for attempt in attempts:
                attempt_num = attempt['attempt']
                self._update_status(f"Fix attempt {attempt_num}...", 0.4 + 0.15 * attempt_num)
                self._log(f"Fix attempt {attempt_num}: {'Success' if attempt['success'] else 'Failed'}")
                if attempt.get("error"):
                    self._log(f"  Error: {attempt['error'][:100]}...")
                
                # Refresh data state panel after each fix attempt (orchestrator creates checkpoints)
                if self.main_app and hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                    self.main_app.data_state_panel._load_latest()
                
                # Note: Checkpoints are created by orchestrator, so we don't create them here
            
            # Get final result
            final_df = process_result.get("final_df")
            if process_result.get("success"):
                self._log("Processing successful!")
                self._update_status("Success", 1.0)
                
                # Update visualization - orchestrator already created completion checkpoint
                if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                    self.main_app.cycle_viz_panel.update_stage(ProcessingStage.COMPLETE)
                    # Refresh data state panel
                    if hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                        self.main_app.data_state_panel._load_latest()
                
                # Export to exports/ directory
                if final_df is not None:
                    try:
                        # #region agent log
                        import json
                        try:
                            with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_C","timestamp":int(datetime.now().timestamp()*1000),"location":"process_panel.py:1009","message":"Before export - final_df columns vs schema","data":{"final_df_columns":list(final_df.columns),"schema_columns":schema.get_column_names() if hasattr(schema, 'get_column_names') else [],"columns_match":list(final_df.columns) == (schema.get_column_names() if hasattr(schema, 'get_column_names') else []),"hypothesisId":"C"},"sessionId":"debug-session","runId":"run1"}) + '\n')
                        except: pass
                        # #endregion
                        exports_dir = Path("exports")
                        exports_dir.mkdir(exist_ok=True)
                        
                        # Generate timestamped filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        schema_name = schema.name if hasattr(schema, 'name') else "output"
                        # Sanitize schema name for filename
                        safe_schema_name = "".join(c for c in schema_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
                        filename = f"bids_{safe_schema_name}_{timestamp}.csv"
                        export_path = exports_dir / filename
                        
                        # Save as CSV
                        final_df.to_csv(export_path, index=False)
                        self._log(f"Output saved to: {export_path}")
                    except Exception as e:
                        self._log(f"Warning: Failed to save export: {str(e)}", level="warning")
                
                result = {
                    "success": True,
                    "final_df": final_df
                }
                # Update main app status bar
                if self.main_app:
                    dpg.set_value(self.main_app.status_text, "Processing complete - Success!")
                    dpg.set_value(self.main_app.progress_bar, 1.0)
                    dpg.configure_item(self.main_app.progress_bar, overlay="Complete")
            else:
                self._log("Processing completed with issues remaining")
                self._update_status("Completed with issues", 1.0)
                
                # Update visualization - orchestrator already created completion checkpoint
                if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                        self.main_app.cycle_viz_panel.update_stage(ProcessingStage.COMPLETE)
                        # Refresh data state panel
                        if hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                            self.main_app.data_state_panel._load_latest()
                
                result = {
                    "success": False,
                    "final_df": final_df
                }
                # Update main app status bar
                if self.main_app:
                    dpg.set_value(self.main_app.status_text, "Processing complete - Issues remain")
                    dpg.set_value(self.main_app.progress_bar, 1.0)
                    dpg.configure_item(self.main_app.progress_bar, overlay="Complete")
            
            # Display last script if available
            if attempts:
                last_attempt = attempts[-1]
                # Would need to get script content from the result
                
        except Exception as e:
            self._log(f"Processing error: {str(e)}")
            self._update_status("Error", 0.0)
            
            # Save error checkpoint
            from ..utils.checkpoint_manager import ProcessingStage
            if self.checkpoint_manager and self.current_df is not None:
                self.checkpoint_manager.create_checkpoint(
                    ProcessingStage.ERROR,
                    df=self.current_df,
                    metadata={"error": str(e)}
                )
                if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                    self.main_app.cycle_viz_panel.update_stage(ProcessingStage.ERROR)
                    # Refresh data state panel
                    if hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                        self.main_app.data_state_panel._load_latest()
            
            result = {"success": False, "error": str(e)}
            # Update main app status bar
            if self.main_app:
                dpg.set_value(self.main_app.status_text, f"Processing error: {str(e)[:50]}")
                dpg.set_value(self.main_app.progress_bar, 0.0)
                dpg.configure_item(self.main_app.progress_bar, overlay="Error")
        
        finally:
            self.is_processing = False
            
            # Trigger callback
            if self.on_process_complete:
                self.on_process_complete(result)
    
    def get_logs(self) -> List[str]:
        """Get all log entries."""
        return self.logs.copy()

