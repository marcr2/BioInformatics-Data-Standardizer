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
    
    def __init__(self, on_process_complete: Optional[Callable] = None):
        """
        Initialize ProcessPanel.
        
        Args:
            on_process_complete: Callback when processing completes
        """
        self.on_process_complete = on_process_complete
        self.orchestrator = None
        self.is_processing = False
        self.logs: List[str] = []
        
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
                    self.log_text_tag = dpg.add_text(
                        "Processing log will appear here...",
                        wrap=-1,
                        color=(149, 165, 166)
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
        
        # Action buttons
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Run Diagnosis",
                callback=self._on_diagnose_click,
                width=120
            )
            dpg.add_button(
                label="Run Full Process",
                callback=self._on_process_click,
                width=120
            )
            dpg.add_spacer(width=-80)
            dpg.add_button(
                label="Clear Log",
                callback=self._clear_log,
                width=70
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
        """Handle diagnose button click."""
        # This would be called from main app with data
        self._log("Diagnosis requested - use run_diagnosis() with data")
    
    def _on_process_click(self) -> None:
        """Handle process button click."""
        self._log("Full process requested - use run_full_process() with data")
    
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
        self._update_status("Diagnosing...", 0.3)
        self._log("Starting diagnosis...")
        
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
                agent = self.orchestrator.diagnostic_agent
            else:
                agent = DA()
            
            self._log("Analyzing data structure...")
            self._update_status("Analyzing...", 0.5)
            
            # Run diagnosis
            result = agent.diagnose(
                df,
                schema,
                self.orchestrator.schema_manager if self.orchestrator else None
            )
            
            self._log(f"Diagnosis complete. Valid: {result.is_valid}")
            self._log(f"Quality score: {result.quality_score:.2f}")
            self._log(f"Issues found: {len(result.issues)}")
            
            # Update UI
            self._update_status("Diagnosis complete", 1.0)
            self._display_issues(result.issues)
            
        except Exception as e:
            self._log(f"Diagnosis error: {str(e)}")
            self._update_status("Error", 0.0)
        
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
        self._update_status("Processing...", 0.1)
        self._log("Starting full process...")
        
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
            
            self._log("Phase 1: Diagnosing data...")
            self._update_status("Diagnosing...", 0.2)
            
            # Run orchestrator
            process_result = self.orchestrator.process(
                df,
                schema_name=schema.name if hasattr(schema, 'name') else "IPA Standard",
                auto_fix=True,
                max_attempts=3
            )
            
            # Log diagnosis results
            diag = process_result.get("diagnosis", {})
            self._log(f"Quality score: {diag.get('quality_score', 0):.2f}")
            self._log(f"Issues found: {diag.get('issues_count', 0)}")
            
            # Log fix attempts
            attempts = process_result.get("fix_attempts", [])
            for attempt in attempts:
                self._update_status(f"Fix attempt {attempt['attempt']}...", 0.4 + 0.15 * attempt['attempt'])
                self._log(f"Fix attempt {attempt['attempt']}: {'Success' if attempt['success'] else 'Failed'}")
                if attempt.get("error"):
                    self._log(f"  Error: {attempt['error'][:100]}...")
            
            # Get final result
            if process_result.get("success"):
                self._log("Processing successful!")
                self._update_status("Success", 1.0)
                result = {
                    "success": True,
                    "final_df": process_result.get("final_df")
                }
            else:
                self._log("Processing completed with issues remaining")
                self._update_status("Completed with issues", 1.0)
                result = {
                    "success": False,
                    "final_df": process_result.get("final_df")
                }
            
            # Display last script if available
            if attempts:
                last_attempt = attempts[-1]
                # Would need to get script content from the result
                
        except Exception as e:
            self._log(f"Processing error: {str(e)}")
            self._update_status("Error", 0.0)
            result = {"success": False, "error": str(e)}
        
        finally:
            self.is_processing = False
            
            # Trigger callback
            if self.on_process_complete:
                self.on_process_complete(result)
    
    def get_logs(self) -> List[str]:
        """Get all log entries."""
        return self.logs.copy()

