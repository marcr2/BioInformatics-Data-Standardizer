"""
Process Panel for BIDS GUI

Handles diagnostic and fix processing with status display.
"""

import dearpygui.dearpygui as dpg
from typing import Optional, Callable, Dict, Any, List
import pandas as pd
import threading
from datetime import datetime

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
    
    def __init__(self, on_process_complete: Optional[Callable] = None, main_app=None):
        """
        Initialize ProcessPanel.
        
        Args:
            on_process_complete: Callback when processing completes
            main_app: Reference to main app for button callbacks
        """
        self.on_process_complete = on_process_complete
        self.main_app = main_app  # Reference to main app
        self.orchestrator = None
        self.is_processing = False
        self.logs: List[str] = []
        
        # Data references (set by main app)
        self.current_df: Optional[pd.DataFrame] = None
        self.current_schema = None
        
        # Checkpoint manager (will be set by main app if available)
        self.checkpoint_manager = None
        
        # UI tags
        self.log_text_tag: Optional[int] = None
        self.issues_container_tag: Optional[int] = None
        self.script_text_tag: Optional[int] = None
        self.progress_tag: Optional[int] = None
        self.status_text_tag: Optional[int] = None
    
    def create(self) -> None:
        """Create the process panel UI."""
        # Initialize orchestrator
        _, _, AO = get_agent_classes()
        try:
            self.orchestrator = AO(
                vector_store_path="data/vector_store",
                schemas_dir="schemas"
            )
        except Exception as e:
            self.orchestrator = None
            self._log(f"Warning: Could not initialize orchestrator: {e}")
        
        # Status section
        with dpg.group(horizontal=True):
            dpg.add_text("Status:", color=(100, 149, 237))
            self.status_text_tag = dpg.add_text("Ready", color=(149, 165, 166))
        
        dpg.add_spacer(height=5)
        
        self.progress_tag = dpg.add_progress_bar(
            default_value=0.0,
            width=-1,
            overlay="Idle"
        )
        
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)
        
        # Tabs for different views
        with dpg.tab_bar():
            # Issues tab
            with dpg.tab(label="Issues Found"):
                with dpg.child_window(height=250, border=True) as container:
                    self.issues_container_tag = container
                    dpg.add_text(
                        "Run diagnosis to see issues",
                        color=(149, 165, 166)
                    )
            
            # Log tab
            with dpg.tab(label="Processing Log"):
                with dpg.child_window(height=250, border=True):
                    # Use input_text with readonly for selectable text
                    self.log_text_tag = dpg.add_input_text(
                        default_value="Processing log will appear here...",
                        multiline=True,
                        readonly=True,
                        width=-1,
                        height=-1,
                        enabled=False  # Disable editing but allow selection
                    )
            
            # Script tab
            with dpg.tab(label="Fix Script"):
                with dpg.child_window(height=250, border=True):
                    self.script_text_tag = dpg.add_text(
                        "Generated fix script will appear here...",
                        wrap=-1,
                        color=(149, 165, 166)
                    )
        
        dpg.add_spacer(height=10)
        
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
    
    def _log(self, message: str) -> None:
        """Add a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # Update log display
        if self.log_text_tag:
            log_text = "\n".join(self.logs[-50:])  # Keep last 50 entries
            dpg.set_value(self.log_text_tag, log_text)
            # Re-enable to allow text selection
            dpg.configure_item(self.log_text_tag, enabled=True)
    
    def _clear_log(self) -> None:
        """Clear the log."""
        self.logs = []
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
        """Display found issues."""
        # Clear container
        dpg.delete_item(self.issues_container_tag, children_only=True)
        
        if not issues:
            dpg.add_text(
                "No issues found! Data is valid.",
                color=(46, 204, 113),
                parent=self.issues_container_tag
            )
            return
        
        for i, issue in enumerate(issues):
            severity = issue.get("severity", "info")
            color = {
                "critical": (231, 76, 60),
                "warning": (241, 196, 15),
                "info": (149, 165, 166)
            }.get(severity, (149, 165, 166))
            
            with dpg.group(parent=self.issues_container_tag):
                with dpg.group(horizontal=True):
                    # Severity badge
                    dpg.add_text(
                        f"[{severity.upper()}]",
                        color=color
                    )
                    dpg.add_text(
                        issue.get("column", "Unknown"),
                        color=(100, 149, 237)
                    )
                
                dpg.add_text(
                    f"  {issue.get('description', 'No description')}",
                    wrap=-1,
                    color=(236, 240, 241)
                )
                
                if issue.get("suggested_fix"):
                    dpg.add_text(
                        f"  Fix: {issue['suggested_fix']}",
                        wrap=-1,
                        color=(149, 165, 166)
                    )
                
                dpg.add_spacer(height=5)
    
    def _display_script(self, script: str) -> None:
        """Display generated fix script."""
        if self.script_text_tag:
            dpg.set_value(self.script_text_tag, script)
    
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
            
            # Run diagnosis
            result = agent.diagnose(
                df,
                schema,
                self.orchestrator.schema_manager if self.orchestrator else None
            )
            
            self._log(f"Diagnosis complete. Valid: {result.is_valid}")
            self._log(f"Quality score: {result.quality_score:.2f}")
            self._log(f"Issues found: {len(result.issues)}")
            
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
            prefs = get_preferences()
            auto_fix = prefs.get("auto_fix", True)
            max_attempts = prefs.get("max_fix_attempts", 3)
            
            # Phase 1: Rules-Based Preprocessing (BEFORE loading LLM)
            self._log("Phase 1: Rules-based preprocessing...")
            self._update_status("Preprocessing...", 0.05)
            
            # Save checkpoint before preprocessing
            if self.checkpoint_manager:
                self.checkpoint_manager.create_checkpoint(
                    ProcessingStage.PREPROCESSING,
                    df=df,
                    metadata={"action": "before_preprocessing"}
                )
                if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                    self.main_app.cycle_viz_panel.update_stage(ProcessingStage.PREPROCESSING)
                    # Refresh data state panel
                    if hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                        self.main_app.data_state_panel._load_latest()
            
            # Run orchestrator with preferences
            # Note: The orchestrator now handles preprocessing internally before loading LLM
            self._log("Running orchestrated pipeline (preprocess -> diagnose -> fix)...")
            process_result = self.orchestrator.process(
                df,
                schema_name=schema.name if hasattr(schema, 'name') else "IPA Standard",
                auto_fix=auto_fix,
                max_attempts=max_attempts
            )
            
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
                    self._log(f"Preprocessing: {remaining} issue(s) remain for LLM")
                else:
                    self._log("Preprocessing resolved all issues!")
            
            # Save checkpoint after preprocessing
            if self.checkpoint_manager:
                preprocessed_df = process_result.get("final_df", df)
                self.checkpoint_manager.create_checkpoint(
                    ProcessingStage.PREPROCESSING,
                    df=preprocessed_df,
                    metadata={
                        "action": "after_preprocessing",
                        "changes_made": len(preprocessing.get("changes_made", [])),
                        "issues_remaining": preprocessing.get("issues_remaining", 0)
                    }
                )
                # Refresh data state panel
                if self.main_app and hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                    self.main_app.data_state_panel._load_latest()
            
            # Update visualization for LLM loading phase
            if self.checkpoint_manager:
                self.checkpoint_manager.create_checkpoint(
                    ProcessingStage.LOADING_LLM,
                    df=df,
                    metadata={"action": "llm_loaded"}
                )
                if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                    self.main_app.cycle_viz_panel.update_stage(ProcessingStage.LOADING_LLM)
            
            # Update status for diagnosis phase
            self._update_status("Diagnosing...", 0.25)
            
            # Save checkpoint before diagnosis
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
            
            # Log diagnosis results
            diag = process_result.get("diagnosis", {})
            self._log(f"Quality score: {diag.get('quality_score', 0):.2f}")
            self._log(f"Issues found: {diag.get('issues_count', 0)}")
            
            # Log fix attempts and save checkpoints
            attempts = process_result.get("fix_attempts", [])
            for attempt in attempts:
                attempt_num = attempt['attempt']
                self._update_status(f"Fix attempt {attempt_num}...", 0.4 + 0.15 * attempt_num)
                self._log(f"Fix attempt {attempt_num}: {'Success' if attempt['success'] else 'Failed'}")
                if attempt.get("error"):
                    self._log(f"  Error: {attempt['error'][:100]}...")
                
                # Save checkpoint for each fix attempt
                if self.checkpoint_manager:
                    fixed_df = process_result.get("final_df")
                    if fixed_df is not None:
                        self.checkpoint_manager.create_checkpoint(
                            ProcessingStage.FIX_ATTEMPT,
                            df=fixed_df,
                            metadata={
                                "attempt": attempt_num,
                                "success": attempt.get("success", False),
                                "error": attempt.get("error")
                            }
                        )
                        if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                            self.main_app.cycle_viz_panel.update_stage(ProcessingStage.FIX_ATTEMPT)
                            # Refresh data state panel
                            if hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                                self.main_app.data_state_panel._load_latest()
            
            # Get final result
            final_df = process_result.get("final_df")
            if process_result.get("success"):
                self._log("Processing successful!")
                self._update_status("Success", 1.0)
                
                # Save completion checkpoint
                if self.checkpoint_manager and final_df is not None:
                    self.checkpoint_manager.create_checkpoint(
                        ProcessingStage.COMPLETE,
                        df=final_df,
                        metadata={"success": True}
                    )
                    if self.main_app and hasattr(self.main_app, 'cycle_viz_panel'):
                        self.main_app.cycle_viz_panel.update_stage(ProcessingStage.COMPLETE)
                        # Refresh data state panel
                        if hasattr(self.main_app, 'data_state_panel') and self.main_app.data_state_panel:
                            self.main_app.data_state_panel._load_latest()
                
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
                
                # Save checkpoint even if not fully successful
                if self.checkpoint_manager and final_df is not None:
                    self.checkpoint_manager.create_checkpoint(
                        ProcessingStage.COMPLETE,
                        df=final_df,
                        metadata={"success": False}
                    )
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

