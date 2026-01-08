"""
Agentic Loop for BIDS

Diagnostic Agent: Analyzes data issues, avoids failed approaches
Fixing Agent: Generates fix scripts using successful examples

Both agents use local LLM (100% private, no API calls)
"""

import io
import sys
import traceback
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

# Lazy import torch to avoid requiring it if not using GPU
try:
    import torch
except ImportError:
    torch = None

from .utils.llm_client import LocalLLMClient, get_gemini_client, get_claude_client
from .utils.logger import get_logger, info, debug, warning, error, log_pipeline_stage
from .vectorizer import NoiseVectorizer, DataFingerprint
from .vector_store import VectorStore, ScriptStatus
from .schema_manager import Schema, SchemaManager, ValidationResult
from .preprocessor import RulesBasedPreprocessor, PreprocessingResult

# Optional checkpoint manager import
try:
    from .utils.checkpoint_manager import CheckpointManager, ProcessingStage
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False
    CheckpointManager = None
    ProcessingStage = None

# Import agent monitor for real-time updates
try:
    from .gui.agent_monitor_panel import (
        notify_phase, notify_progress, notify_thought, notify_model, AgentPhase
    )
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    # Define stubs if GUI not available
    class AgentPhase:
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
    
    def notify_phase(phase): pass
    def notify_progress(progress): pass
    def notify_thought(content, phase=None, token_count=0, metadata=None): pass
    def notify_model(model_name): pass


@dataclass
class DiagnosisResult:
    """Result from the Diagnostic Agent."""
    is_valid: bool
    quality_score: float
    issues: List[Dict[str, Any]]
    summary: str
    schema_validation: Optional[ValidationResult] = None
    avoided_approaches: List[str] = field(default_factory=list)


@dataclass
class FixResult:
    """Result from the Fixing Agent."""
    success: bool
    fixed_df: Optional[pd.DataFrame]
    script_content: str
    execution_time: float
    error_message: Optional[str] = None
    changes_made: List[str] = field(default_factory=list)
    is_execution_error: bool = False  # True if error is syntax/runtime, False if logical error


class DiagnosticAgent:
    """
    Diagnostic Agent powered by local LLM.
    
    Responsibilities:
    - Analyze DataFrame for quality issues
    - Validate against target schema
    - Query vector store for failed approaches to avoid
    - Generate structured diagnosis report
    """
    
    def __init__(
        self,
        gemini_client: Optional[LocalLLMClient] = None,
        vector_store: Optional[VectorStore] = None,
        vectorizer: Optional[NoiseVectorizer] = None
    ):
        """
        Initialize DiagnosticAgent.
        
        Args:
            gemini_client: Local LLM client (backward compatible name)
            vector_store: Vector store for failed script lookup
            vectorizer: Vectorizer for fingerprinting
        """
        # Store client directly if provided, otherwise None for lazy loading
        self._llm_client = gemini_client
        self.vector_store = vector_store
        self.vectorizer = vectorizer or NoiseVectorizer()
    
    @property
    def llm(self) -> LocalLLMClient:
        """Lazy-load LLM client only when accessed."""
        if self._llm_client is None:
            self._llm_client = get_gemini_client()
        return self._llm_client
    
    @llm.setter
    def llm(self, value: Optional[LocalLLMClient]) -> None:
        """Set LLM client directly."""
        self._llm_client = value
    
    def diagnose(
        self,
        df: pd.DataFrame,
        schema: Schema,
        schema_manager: Optional[SchemaManager] = None,
        preprocessing_report: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> DiagnosisResult:
        """
        Perform comprehensive diagnosis on a DataFrame.
        
        Args:
            df: DataFrame to diagnose
            schema: Target output schema
            schema_manager: Schema manager for validation
            preprocessing_report: Optional report from rules-based preprocessing
            additional_context: Optional additional context from user
            
        Returns:
            DiagnosisResult object
        """
        # Notify monitor of phase change
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.ANALYZING)
            notify_progress(0.1)
            notify_thought(
                f"Starting analysis of DataFrame with shape {df.shape}\n"
                f"Target schema: {schema.name}",
                AgentPhase.ANALYZING
            )
        
        # Get schema validation
        schema_validation = None
        if schema_manager:
            if MONITOR_AVAILABLE:
                notify_thought("Validating against target schema...", AgentPhase.ANALYZING)
            schema_validation = schema_manager.validate_dataframe(df, schema)
        
        # Get failed scripts to avoid
        failed_scripts = []
        avoided_approaches = []
        if self.vector_store:
            if MONITOR_AVAILABLE:
                notify_thought("Checking vector store for failed approaches to avoid...", AgentPhase.ANALYZING)
                notify_progress(0.2)
            fingerprint = self.vectorizer.create_fingerprint(df)
            # Try to fit and transform if not already fitted
            try:
                self.vectorizer.fit([fingerprint])
                self.vectorizer.transform(fingerprint)
                failed_scripts = self.vector_store.get_failed_scripts(fingerprint, top_k=5)
                avoided_approaches = [s[:200] + "..." for s in failed_scripts]
                if MONITOR_AVAILABLE and avoided_approaches:
                    notify_thought(f"Found {len(avoided_approaches)} failed approaches to avoid", AgentPhase.ANALYZING)
            except Exception:
                pass
        
        # Prepare DataFrame info for LLM
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.DIAGNOSING)
            notify_progress(0.3)
            notify_thought("Preparing data structure analysis for LLM...", AgentPhase.DIAGNOSING)
        
        df_info = self._prepare_df_info(df)
        # #region agent log
        import json
        try:
            with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_O","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:197","message":"Prepared df_info for diagnostic","data":{"df_info_length":len(df_info),"df_shape":df.shape,"hypothesisId":"O"},"sessionId":"debug-session","runId":"run2"}) + '\n')
        except: pass
        # #endregion
        schema_rules = schema.get_validation_rules()
        
        # Include preprocessing context if available
        if preprocessing_report:
            df_info += f"\n\nPREPROCESSING ALREADY APPLIED:\n{preprocessing_report}"
            if MONITOR_AVAILABLE:
                notify_thought(
                    "Including preprocessing context in analysis.\n"
                    "The LLM will focus on remaining complex issues.",
                    AgentPhase.DIAGNOSING
                )
        
        # Include additional user context if provided
        if additional_context:
            # Check if context contains uploaded files
            if "--- UPLOADED FILE:" in additional_context:
                df_info += f"\n\nADDITIONAL USER CONTEXT AND UPLOADED FILES:\n"
                df_info += "The user has provided context and uploaded files. "
                df_info += "When the user references a file by name in their text context, "
                df_info += "refer to the corresponding file content below.\n"
            else:
                df_info += f"\n\nADDITIONAL USER CONTEXT:\n"
            df_info += additional_context
            if MONITOR_AVAILABLE:
                # Count files if present
                file_count = additional_context.count("--- UPLOADED FILE:")
                context_desc = f"user-provided context"
                if file_count > 0:
                    context_desc += f" with {file_count} uploaded file{'s' if file_count > 1 else ''}"
                notify_thought(
                    f"Including {context_desc} ({len(additional_context)} chars)",
                    AgentPhase.DIAGNOSING
                )
        
        # Call local LLM for diagnosis
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.THINKING)
            notify_progress(0.4)
            notify_thought(
                "Sending data to LLM for deep analysis...\n"
                "Analyzing column types, value distributions, and quality issues...",
                AgentPhase.THINKING
            )
            notify_model(self.llm.model_id if hasattr(self.llm, 'model_id') else "Local LLM")
        
        diagnosis = self.llm.diagnose_data(df_info, schema_rules, failed_scripts)
        
        if MONITOR_AVAILABLE:
            notify_progress(0.8)
            notify_thought(
                f"LLM Analysis Complete:\n"
                f"  Quality Score: {diagnosis.get('overall_quality_score', 0.0):.2f}\n"
                f"  Issues Found: {len(diagnosis.get('issues', []))}\n"
                f"  Summary: {diagnosis.get('summary', 'N/A')[:200]}",
                AgentPhase.REVIEWING
            )
        
        # Merge with schema validation
        issues = diagnosis.get("issues", [])
        if schema_validation and not schema_validation.is_valid:
            for error in schema_validation.errors:
                issues.append({
                    "column": "schema",
                    "issue_type": "schema_mismatch",
                    "severity": "critical",
                    "description": error,
                    "affected_rows": "all",
                    "suggested_fix": "Map or transform columns to match schema"
                })
        
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.COMPLETE)
            notify_progress(1.0)
            notify_thought(
                f"Diagnosis complete. Valid: {diagnosis.get('is_valid', False)}\n"
                f"Total issues: {len(issues)}",
                AgentPhase.COMPLETE
            )
        
        return DiagnosisResult(
            is_valid=diagnosis.get("is_valid", False) and (schema_validation.is_valid if schema_validation else True),
            quality_score=diagnosis.get("overall_quality_score", 0.0),
            issues=issues,
            summary=diagnosis.get("summary", "Diagnosis complete"),
            schema_validation=schema_validation,
            avoided_approaches=avoided_approaches
        )
    
    def _prepare_df_info(self, df: pd.DataFrame) -> str:
        """Prepare DataFrame information for LLM analysis."""
        buffer = io.StringIO()
        
        buffer.write(f"Shape: {df.shape}\n\n")
        buffer.write("Columns and Types:\n")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            buffer.write(f"  - {col}: {df[col].dtype}, nulls={null_count}, unique={unique_count}\n")
        
        buffer.write("\nSample Data (first 5 rows):\n")
        buffer.write(df.head().to_string())
        
        buffer.write("\n\nStatistics:\n")
        try:
            # Limit statistics to avoid huge outputs that cause OOM
            # Only describe numeric columns and limit categorical info
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                buffer.write("Numeric columns:\n")
                buffer.write(df[numeric_cols].describe().to_string())
            
            # For non-numeric columns, just show value counts for top values (limit to avoid huge output)
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                buffer.write("\n\nNon-numeric columns (top 3 values per column):\n")
                for col in non_numeric_cols[:10]:  # Limit to first 10 non-numeric columns
                    top_values = df[col].value_counts().head(3)
                    buffer.write(f"  {col}: {dict(top_values)}\n")
        except Exception as e:
            buffer.write(f"(Could not compute statistics: {str(e)})")
        
        return buffer.getvalue()
    
    def quick_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform a quick quality check without LLM.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with quick check results
        """
        issues = []
        
        # Check for nulls
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            issues.append({
                "type": "null_values",
                "columns": null_cols,
                "severity": "warning"
            })
        
        # Check for empty strings
        for col in df.select_dtypes(include=['object']).columns:
            empty_count = (df[col] == '').sum()
            if empty_count > 0:
                issues.append({
                    "type": "empty_strings",
                    "column": col,
                    "count": int(empty_count),
                    "severity": "warning"
                })
        
        # Check for duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append({
                "type": "duplicates",
                "count": int(dup_count),
                "severity": "info"
            })
        
        # Check numeric columns for outliers
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            if outliers > 0:
                issues.append({
                    "type": "outliers",
                    "column": col,
                    "count": int(outliers),
                    "severity": "info"
                })
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "issues": issues,
            "has_issues": len(issues) > 0
        }


def _create_restricted_import(sandbox_globals: dict, changes_log: list):
    """
    Create a restricted __import__ function for sandbox execution.
    
    Only allows importing modules that are:
    1. Already in the sandbox globals (pd, np, re)
    2. Safe data processing modules from the virtual environment
    
    Blocks dangerous modules like os, sys, subprocess, etc.
    """
    # Modules that are explicitly blocked for security
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        'importlib', 'builtins', '__builtins__',
        'ctypes', 'socket', 'http', 'urllib', 'requests',
        'pickle', 'marshal', 'shelve',
        'multiprocessing', 'threading', 'concurrent',
        'asyncio', 'signal', 'atexit',
        'code', 'codeop', 'compile', 'exec', 'eval',
        'gc', 'inspect', 'traceback', 'linecache',
        'tempfile', 'glob', 'fnmatch',
        'sqlite3', 'dbm', 'gdbm',
        'webbrowser', 'pdb', 'profile', 'pstats',
    }
    
    # Modules that are safe to import (data processing)
    ALLOWED_MODULES = {
        'pandas', 'pd',
        'numpy', 'np',
        're', 'regex',
        'math', 'statistics', 'random',
        'datetime', 'time', 'calendar',
        'collections', 'itertools', 'functools', 'operator',
        'string', 'textwrap',
        'json', 'csv',
        'decimal', 'fractions',
        'copy', 'typing',
        'warnings',
        'scipy', 'sklearn', 'statsmodels',
    }
    
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Restricted import that only allows safe modules."""
        # Get the base module name
        base_module = name.split('.')[0]
        
        # Check if module is blocked
        if base_module in BLOCKED_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed in sandbox for security reasons")
        
        # Check if module is in sandbox globals (already available)
        if name in sandbox_globals:
            changes_log.append(f"LOG: Using pre-loaded module '{name}'")
            return sandbox_globals[name]
        if name == 'pandas' and 'pd' in sandbox_globals:
            changes_log.append(f"LOG: Using pre-loaded module 'pandas' as 'pd'")
            return sandbox_globals['pd']
        if name == 'numpy' and 'np' in sandbox_globals:
            changes_log.append(f"LOG: Using pre-loaded module 'numpy' as 'np'")
            return sandbox_globals['np']
        
        # Check if module is explicitly allowed
        if base_module in ALLOWED_MODULES:
            try:
                changes_log.append(f"LOG: Importing allowed module '{name}'")
                return __import__(name, globals, locals, fromlist, level)
            except ImportError as e:
                raise ImportError(f"Failed to import '{name}': {e}")
        
        # For any other module, deny access
        raise ImportError(f"Import of '{name}' is not allowed in sandbox. Pre-loaded modules: pd, np, re")
    
    return restricted_import


class FixingAgent:
    """
    Fixing Agent powered by local LLM.
    
    Responsibilities:
    - Generate Python fix scripts based on diagnosis
    - Use successful scripts from vector store as examples
    - Execute scripts safely
    - Record success/failure in vector store
    """
    
    def __init__(
        self,
        claude_client: Optional[LocalLLMClient] = None,
        vector_store: Optional[VectorStore] = None,
        vectorizer: Optional[NoiseVectorizer] = None
    ):
        """
        Initialize FixingAgent.
        
        Args:
            claude_client: Local LLM client (backward compatible name)
            vector_store: Vector store for script lookup and storage
            vectorizer: Vectorizer for fingerprinting
        """
        # Store client directly if provided, otherwise None for lazy loading
        self._llm_client = claude_client
        self.vector_store = vector_store
        self.vectorizer = vectorizer or NoiseVectorizer()
        
        # Sandbox for executing generated code
        self._sandbox_globals = {
            'pd': pd,
            'np': np,
            're': re,
        }
    
    @property
    def llm(self) -> LocalLLMClient:
        """Lazy-load LLM client only when accessed."""
        if self._llm_client is None:
            self._llm_client = get_claude_client()
        return self._llm_client
    
    @llm.setter
    def llm(self, value: Optional[LocalLLMClient]) -> None:
        """Set LLM client directly."""
        self._llm_client = value
    
    def fix(
        self,
        df: pd.DataFrame,
        diagnosis: DiagnosisResult,
        schema: Schema,
        additional_context: Optional[str] = None
    ) -> FixResult:
        """
        Generate and execute a fix script.
        
        Args:
            df: DataFrame to fix
            diagnosis: Diagnosis results
            schema: Target output schema
            additional_context: Optional additional context from user
            
        Returns:
            FixResult object
        """
        # Notify monitor of phase change
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.ANALYZING)
            notify_progress(0.1)
            notify_thought(
                f"Preparing to fix DataFrame with {len(diagnosis.issues)} diagnosed issues\n"
                f"Target schema: {schema.name}",
                AgentPhase.ANALYZING
            )
        
        # Get successful scripts as examples
        successful_scripts = []
        fingerprint = None
        
        if self.vector_store:
            if MONITOR_AVAILABLE:
                notify_thought("Searching vector store for successful fix examples...", AgentPhase.ANALYZING)
            fingerprint = self.vectorizer.create_fingerprint(df)
            try:
                self.vectorizer.fit([fingerprint])
                self.vectorizer.transform(fingerprint)
                successful_scripts = self.vector_store.get_successful_scripts(fingerprint, top_k=3)
                if MONITOR_AVAILABLE and successful_scripts:
                    notify_thought(f"Found {len(successful_scripts)} successful fix examples to use as reference", AgentPhase.ANALYZING)
            except Exception:
                pass
        
        # Prepare sample data for LLM
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.THINKING)
            notify_progress(0.2)
            notify_thought(
                "Analyzing diagnosed issues and preparing prompt for code generation...\n"
                f"Issues to fix:\n" + "\n".join(
                    f"  - {issue.get('column', 'Unknown')}: {issue.get('description', 'No description')[:60]}"
                    for issue in diagnosis.issues[:5]
                ),
                AgentPhase.THINKING
            )
        
        df_sample = df.head(10).to_string()
        
        # Generate fix script using local LLM
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.GENERATING)
            notify_progress(0.3)
            notify_thought(
                "Generating Python fix script using LLM...\n"
                "The script will:\n"
                "  - Handle all diagnosed issues\n"
                "  - Map columns to target schema\n"
                "  - Convert data types appropriately\n"
                "  - Handle missing values gracefully",
                AgentPhase.GENERATING
            )
            notify_model(self.llm.model_id if hasattr(self.llm, 'model_id') else "Local LLM")
        
        script = self.llm.generate_fix_script(
            df_sample=df_sample,
            diagnosis={
                "is_valid": diagnosis.is_valid,
                "quality_score": diagnosis.quality_score,
                "issues": diagnosis.issues,
                "summary": diagnosis.summary
            },
            target_schema=schema.to_dict(),
            successful_scripts=successful_scripts,
            additional_context=additional_context
        )
        
        if MONITOR_AVAILABLE:
            notify_progress(0.6)
            notify_thought(
                f"Generated fix script ({len(script)} chars):\n"
                f"```python\n{script[:500]}{'...' if len(script) > 500 else ''}\n```",
                AgentPhase.GENERATING
            )
        
        # Execute the script
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.EXECUTING)
            notify_progress(0.7)
            notify_thought("Executing generated fix script in sandbox environment...", AgentPhase.EXECUTING)
        
        start_time = datetime.now()
        fixed_df, error_message, changes, is_execution_error = self._execute_fix(df, script)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        success = fixed_df is not None and error_message is None
        
        # If execution failed, try to debug it
        if not success and is_execution_error and self.llm:
            debug(f"Execution error detected, attempting to debug script...", context="FixingAgent")
            debugged_result = self._debug_and_retry_execution(
                df, script, error_message
            )
            fixed_df, error_message, changes, is_execution_error = debugged_result
            if fixed_df is not None:
                success = True
                # Update script to the debugged version
                # Note: We'll track the original script separately if needed
                execution_time = (datetime.now() - start_time).total_seconds()
                debug(f"Script debugged and executed successfully", context="FixingAgent")
        
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.REVIEWING)
            notify_progress(0.9)
            if success:
                notify_thought(
                    f"Script executed successfully in {execution_time:.2f}s\n"
                    f"Changes made:\n" + "\n".join(f"  - {c}" for c in changes[:10]),
                    AgentPhase.REVIEWING
                )
            else:
                notify_thought(
                    f"Script execution failed:\n{error_message[:300]}",
                    AgentPhase.ERROR
                )
        
        # Store result in vector store
        if self.vector_store and fingerprint:
            error_type = diagnosis.issues[0]["issue_type"] if diagnosis.issues else "unknown"
            self.vector_store.add_entry(
                fingerprint=fingerprint,
                script_content=script,
                status=ScriptStatus.SUCCESS if success else ScriptStatus.FAILURE,
                error_type=error_type,
                source_file="runtime",
                execution_time=execution_time,
                error_message=error_message
            )
            if MONITOR_AVAILABLE:
                notify_thought(
                    f"Stored {'successful' if success else 'failed'} script in vector store for future reference",
                    AgentPhase.REVIEWING
                )
        
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.COMPLETE if success else AgentPhase.ERROR)
            notify_progress(1.0)
            notify_thought(
                f"Fix attempt {'succeeded' if success else 'failed'}.\n"
                f"Execution time: {execution_time:.2f}s",
                AgentPhase.COMPLETE if success else AgentPhase.ERROR
            )
        
        return FixResult(
            success=success,
            fixed_df=fixed_df,
            script_content=script,
            execution_time=execution_time,
            error_message=error_message,
            changes_made=changes,
            is_execution_error=is_execution_error if not success else False
        )
    
    def _execute_fix(
        self,
        df: pd.DataFrame,
        script: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str], bool]:
        """
        Execute a fix script safely.
        
        Args:
            df: Input DataFrame
            script: Python script to execute
            
        Returns:
            Tuple of (fixed_df, error_message, changes_made, is_execution_error)
            is_execution_error: True if error is syntax/runtime (should be debugged), 
                              False if logical error (script ran but didn't fix data)
        """
        changes = []
        
        # Pre-validate script for common issues
        if not script or not script.strip():
            return None, "Script is empty", changes, True  # Execution error
        
        # Check if script appears truncated/incomplete
        if hasattr(self.llm, '_is_code_truncated') and self.llm._is_code_truncated(script, "python"):
            return None, "Generated script appears truncated or incomplete. The LLM may have hit the token limit. Try reducing the number of issues or simplifying the fix requirements.", changes, True  # Execution error
        
        # Check for common syntax issues
        try:
            compile(script, '<string>', 'exec')
        except SyntaxError as e:
            # Provide helpful error message for common issues
            error_msg = f"Syntax error in script: {str(e)}"
            if 'invalid escape sequence' in str(e).lower():
                error_msg += "\nHint: Use raw strings (r'...') for regex patterns containing backslashes, e.g., r'^[A-Za-z0-9\\-]+$'"
            return None, error_msg, changes, True  # Execution error
        
        try:
            # Prepare sandbox environment
            sandbox = self._sandbox_globals.copy()
            sandbox['df'] = df.copy()
            
            # Create restricted import function for this execution
            restricted_import = _create_restricted_import(sandbox, changes)
            
            sandbox['__builtins__'] = {
                '__import__': restricted_import,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'all': all,
                'any': any,
                'ord': ord,
                'chr': chr,
                'hex': hex,
                'oct': oct,
                'bin': bin,
                'pow': pow,
                'divmod': divmod,
                'repr': repr,
                'hash': hash,
                'id': id,
                'callable': callable,
                'iter': iter,
                'next': next,
                'slice': slice,
                'reversed': reversed,
                'print': lambda *args: changes.append(f"LOG: {' '.join(map(str, args))}"),
                'isinstance': isinstance,
                'issubclass': issubclass,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'delattr': delattr,
                'type': type,
                'object': object,
                'property': property,
                'staticmethod': staticmethod,
                'classmethod': classmethod,
                'super': super,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'AttributeError': AttributeError,
                'RuntimeError': RuntimeError,
                'StopIteration': StopIteration,
                'Exception': Exception,
                'True': True,
                'False': False,
                'None': None,
            }
            
            # Execute the script
            exec(script, sandbox)
            
            # Look for the result
            result = None
            if 'fix_dataframe' in sandbox and callable(sandbox['fix_dataframe']):
                # If a function was defined, call it
                try:
                    result = sandbox['fix_dataframe'](df.copy())
                except Exception as e:
                    # Error calling function - execution error
                    return None, f"Error calling fix_dataframe function: {type(e).__name__}: {str(e)}", changes, True
            elif 'result' in sandbox and isinstance(sandbox['result'], pd.DataFrame):
                result = sandbox['result']
            elif 'df' in sandbox and isinstance(sandbox['df'], pd.DataFrame):
                result = sandbox['df']
            else:
                # Script executed but didn't produce expected result - this is a logical error, not execution error
                return None, "Script did not produce a DataFrame result. Expected 'fix_dataframe' function, 'result' variable, or modified 'df' variable.", changes, False
            
            # Validate result
            if not isinstance(result, pd.DataFrame):
                # Script executed but returned wrong type - logical error
                return None, f"Result is not a DataFrame: {type(result)}", changes, False
            
            if len(result) == 0:
                # Script executed but produced empty result - logical error
                return None, "Result DataFrame is empty", changes, False
            
            # Validate column structure
            if len(result.columns) == 0:
                # Script executed but produced invalid result - logical error
                return None, "Result DataFrame has no columns", changes, False
            
            # Track changes
            if len(result.columns) != len(df.columns):
                changes.append(f"Columns changed: {len(df.columns)} -> {len(result.columns)}")
            if len(result) != len(df):
                changes.append(f"Rows changed: {len(df)} -> {len(result)}")
            
            # Script executed successfully and produced valid result
            return result, None, changes, False
            
        except Exception as e:
            # Any exception during execution is an execution error
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg, changes, True
    
    def apply_manual_fix(
        self,
        df: pd.DataFrame,
        script: str
    ) -> FixResult:
        """
        Apply a manually provided fix script.
        
        Args:
            df: DataFrame to fix
            script: Python script to execute
            
        Returns:
            FixResult object
        """
        start_time = datetime.now()
        fixed_df, error_message, changes, is_execution_error = self._execute_fix(df, script)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # If execution failed, try to debug it
        if fixed_df is None and is_execution_error and self.llm:
            debug(f"Execution error detected, attempting to debug script...", context="FixingAgent")
            fixed_df, error_message, changes, is_execution_error = self._debug_and_retry_execution(
                df, script, error_message
            )
            execution_time = (datetime.now() - start_time).total_seconds()
        
        return FixResult(
            success=fixed_df is not None,
            fixed_df=fixed_df,
            script_content=script,
            execution_time=execution_time,
            error_message=error_message,
            changes_made=changes,
            is_execution_error=is_execution_error if fixed_df is None else False
        )
    
    def _debug_and_retry_execution(
        self,
        df: pd.DataFrame,
        script: str,
        error_message: str,
        max_debug_attempts: int = 5
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str], bool]:
        """
        Debug a script execution error using LLM and retry until it executes successfully.
        
        Args:
            df: Input DataFrame
            script: Original script that failed
            error_message: Error message from execution
            max_debug_attempts: Maximum number of debug attempts
            
        Returns:
            Tuple of (fixed_df, error_message, changes_made, is_execution_error)
        """
        current_script = script
        debug_attempt = 0
        
        while debug_attempt < max_debug_attempts:
            debug_attempt += 1
            info(f"Debug attempt {debug_attempt}/{max_debug_attempts} for script execution error", context="FixingAgent")
            
            # Get debugged script from LLM
            try:
                debugged_script = self.llm.debug_script_execution(
                    script=current_script,
                    error_message=error_message,
                    previous_errors=[] if debug_attempt == 1 else [error_message]
                )
                
                if not debugged_script or debugged_script.strip() == "":
                    warning(f"Debug LLM returned empty script on attempt {debug_attempt}", context="FixingAgent")
                    break
                
                # Try executing the debugged script
                fixed_df, new_error, changes, is_exec_error = self._execute_fix(df, debugged_script)
                
                if fixed_df is not None:
                    # Success! Script executed
                    info(f"Script debugged successfully on attempt {debug_attempt}", context="FixingAgent")
                    return fixed_df, None, changes, False
                elif not is_exec_error:
                    # Script executed but produced invalid result - this is a logical error, not execution error
                    # Return it as a logical failure (not an execution error)
                    return None, new_error, changes, False
                else:
                    # Still an execution error, try again
                    error_message = new_error
                    current_script = debugged_script
                    debug(f"Debug attempt {debug_attempt} still has execution error, retrying...", context="FixingAgent")
                    
            except Exception as e:
                error(f"Error in debug LLM call: {e}", context="FixingAgent")
                break
        
        # All debug attempts failed
        warning(f"Could not debug script after {debug_attempt} attempts", context="FixingAgent")
        return None, f"Script execution failed after {debug_attempt} debug attempts. Last error: {error_message[:200]}", [], True


class AgentOrchestrator:
    """
    Orchestrates the diagnostic and fixing agents.
    
    Provides a high-level interface for the complete
    diagnosis -> fix -> validate cycle.
    """
    
    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        schemas_dir: Optional[str] = None,
        checkpoint_manager: Optional[Any] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            vector_store_path: Path for persistent vector store
            schemas_dir: Directory for schemas
            checkpoint_manager: Optional checkpoint manager for saving data states
        """
        from .vector_store import get_vector_store
        from .schema_manager import get_schema_manager
        
        self.vector_store = get_vector_store(vector_store_path)
        self.schema_manager = get_schema_manager(schemas_dir)
        self.vectorizer = NoiseVectorizer()
        self.checkpoint_manager = checkpoint_manager
        
        # Rules-based preprocessor (runs before LLM)
        self.preprocessor = RulesBasedPreprocessor(self.schema_manager)
        
        # Lazy loading: Don't load LLM until first use
        # This saves startup time and GPU memory until actually needed
        self._shared_llm_client: Optional[LocalLLMClient] = None
        self._llm_loading = False
        
        # Initialize agents without LLM clients (will be set lazily)
        self.diagnostic_agent = DiagnosticAgent(
            gemini_client=None,  # Will be set on first use
            vector_store=self.vector_store,
            vectorizer=self.vectorizer
        )
        
        self.fixing_agent = FixingAgent(
            claude_client=None,  # Will be set on first use
            vector_store=self.vector_store,
            vectorizer=self.vectorizer
        )
    
    def _ensure_llm_loaded(self) -> LocalLLMClient:
        """
        Ensure the shared LLM client is loaded (lazy loading).
        
        Returns:
            The shared LLM client instance
        """
        if self._shared_llm_client is None:
            if self._llm_loading:
                # Another thread is loading, wait a bit
                import time
                while self._llm_loading:
                    time.sleep(0.1)
                return self._shared_llm_client
            
            self._llm_loading = True
            try:
                # Notify monitor of loading phase
                if MONITOR_AVAILABLE:
                    notify_phase(AgentPhase.LOADING)
                    notify_progress(0.0)
                    notify_thought(
                        "Loading LLM model...\n"
                        "This may take 10-30 seconds on first use.\n"
                        "The model will be cached for subsequent operations.",
                        AgentPhase.LOADING
                    )
                
                # Share a single LLM client between agents to save GPU memory
                # Both agents can use the same model instance since they're not used simultaneously
                info("Loading LLM model (this may take 10-30 seconds on first use)...", context="AgentOrchestrator")
                self._shared_llm_client = get_gemini_client()
                
                # Update agents with the shared client
                self.diagnostic_agent.llm = self._shared_llm_client
                self.fixing_agent.llm = self._shared_llm_client
                
                info("LLM model loaded successfully!", context="AgentOrchestrator")
                
                # Notify monitor of successful load
                if MONITOR_AVAILABLE:
                    model_id = self._shared_llm_client.model_id if hasattr(self._shared_llm_client, 'model_id') else "Unknown"
                    notify_model(model_id)
                    notify_thought(
                        f"LLM model loaded successfully!\n"
                        f"Model: {model_id}\n"
                        f"Ready for diagnosis and fix operations.",
                        AgentPhase.LOADING
                    )
            finally:
                self._llm_loading = False
        
        return self._shared_llm_client
    
    def _llm_rewrite(
        self,
        df: pd.DataFrame,
        schema: Schema
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform LLM-based structural rewrite of data to match schema.
        
        This is a STRUCTURAL transformation only - it reorders/reorganizes data
        to match schema structure. It does NOT transform data values.
        Data value transformations are handled by fix scripts, not this method.
        
        Args:
            df: Input DataFrame (preprocessed)
            schema: Target schema
            
        Returns:
            Tuple of (rewritten_df, rewrite_report)
            rewrite_report contains: success, column_mapping, validation_results, error
        """
        rewrite_report = {
            "success": False,
            "column_mapping": {},
            "validation_results": {},
            "error": None,
            "original_row_count": len(df),
            "original_columns": list(df.columns)
        }
        
        try:
            # Ensure LLM is loaded
            self._ensure_llm_loaded()
            
            if MONITOR_AVAILABLE:
                notify_phase(AgentPhase.THINKING)
                notify_thought(
                    "Starting LLM structural rewrite step...\n"
                    "This step will reorganize data structure to match schema.\n"
                    "Data values will NOT be modified - only column names/organization.",
                    AgentPhase.THINKING
                )
            
            # Prepare DataFrame sample for LLM
            sample_rows = min(10, len(df))
            df_sample = df.head(sample_rows).to_string()
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_K","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1078","message":"LLM rewrite - dataframe sample prepared","data":{"df_columns":list(df.columns),"df_shape":df.shape,"sample_rows":sample_rows,"sample_length":len(df_sample),"hypothesisId":"K"},"sessionId":"debug-session","runId":"run2"}) + '\n')
            except: pass
            # #endregion
            
            # Get schema as dict
            schema_dict = {
                "columns": [
                    {
                        "name": col.name,
                        "type": col.type.value if hasattr(col.type, 'value') else str(col.type),
                        "description": col.description or ""
                    }
                    for col in schema.columns
                ]
            }
            
            # Generate rewrite script
            if MONITOR_AVAILABLE:
                notify_thought(
                    "Generating structural rewrite script...\n"
                    "The script will only rename/reorder columns, not modify values.",
                    AgentPhase.GENERATING
                )
            
            script = self._shared_llm_client.rewrite_to_schema(
                df_sample=df_sample,
                target_schema=schema_dict,
                source_columns=list(df.columns)
            )
            
            # Execute the rewrite script using the same sandbox as fix scripts
            # but with additional validation for data preservation
            changes = []
            
            # Use the fixing agent's sandbox globals
            sandbox = self.fixing_agent._sandbox_globals.copy()
            sandbox['df'] = df.copy()
            
            # Create restricted import function
            restricted_import = _create_restricted_import(sandbox, changes)
            
            sandbox['__builtins__'] = {
                '__import__': restricted_import,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'all': all,
                'any': any,
                'print': lambda *args: changes.append(f"LOG: {' '.join(map(str, args))}"),
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'type': type,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'Exception': Exception,
                'True': True,
                'False': False,
                'None': None,
            }
            
            # Execute the script
            exec(script, sandbox)
            
            # Get the result
            if 'rewrite_dataframe' in sandbox and callable(sandbox['rewrite_dataframe']):
                rewritten_df = sandbox['rewrite_dataframe'](df.copy())
            elif 'result' in sandbox and isinstance(sandbox['result'], pd.DataFrame):
                rewritten_df = sandbox['result']
            elif 'df' in sandbox and isinstance(sandbox['df'], pd.DataFrame):
                rewritten_df = sandbox['df']
            else:
                rewrite_report["error"] = "Rewrite script did not produce a DataFrame result"
                return df, rewrite_report
            
            # CRITICAL: Validate data preservation
            validation = self._validate_rewrite_preservation(df, rewritten_df)
            rewrite_report["validation_results"] = validation
            
            if not validation["row_count_valid"]:
                rewrite_report["error"] = f"Row count changed: {validation['original_rows']} -> {validation['rewritten_rows']}. Rewrite rejected."
                if MONITOR_AVAILABLE:
                    notify_thought(
                        f"REWRITE REJECTED: Row count changed ({validation['original_rows']} -> {validation['rewritten_rows']})\n"
                        "Using original DataFrame instead.",
                        AgentPhase.ERROR
                    )
                return df, rewrite_report
            
            if not validation["values_preserved"]:
                rewrite_report["error"] = f"Data values were modified. Rewrite rejected. Details: {validation['value_changes']}"
                if MONITOR_AVAILABLE:
                    notify_thought(
                        f"REWRITE REJECTED: Data values were modified\n"
                        f"Details: {validation['value_changes'][:200]}\n"
                        "Using original DataFrame instead.",
                        AgentPhase.ERROR
                    )
                return df, rewrite_report
            
            # Track column mapping
            rewrite_report["column_mapping"] = validation.get("column_mapping", {})
            rewrite_report["success"] = True
            
            if MONITOR_AVAILABLE:
                notify_thought(
                    f"Structural rewrite successful!\n"
                    f"Row count preserved: {len(rewritten_df)}\n"
                    f"Columns: {list(rewritten_df.columns)[:5]}{'...' if len(rewritten_df.columns) > 5 else ''}\n"
                    f"Data values verified unchanged.",
                    AgentPhase.COMPLETE
                )
            
            return rewritten_df, rewrite_report
            
        except Exception as e:
            rewrite_report["error"] = f"Rewrite failed: {str(e)}"
            if MONITOR_AVAILABLE:
                notify_thought(
                    f"Rewrite step failed: {str(e)}\n"
                    "Using original DataFrame instead.",
                    AgentPhase.ERROR
                )
            return df, rewrite_report
    
    def _validate_rewrite_preservation(
        self,
        original_df: pd.DataFrame,
        rewritten_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate that the rewrite preserved all data values.
        
        Checks:
        1. Row count is identical
        2. Data values are unchanged (sample comparison)
        3. No data was lost
        
        Args:
            original_df: Original DataFrame before rewrite
            rewritten_df: DataFrame after rewrite
            
        Returns:
            Dict with validation results
        """
        validation = {
            "row_count_valid": False,
            "values_preserved": False,
            "original_rows": len(original_df),
            "rewritten_rows": len(rewritten_df),
            "column_mapping": {},
            "value_changes": ""
        }
        
        # Check row count
        if len(original_df) != len(rewritten_df):
            validation["row_count_valid"] = False
            return validation
        validation["row_count_valid"] = True
        
        # Build column mapping (which original columns are in rewritten df)
        for orig_col in original_df.columns:
            for new_col in rewritten_df.columns:
                # Check if columns have same values
                try:
                    orig_values = original_df[orig_col].fillna("__NA__").astype(str).tolist()
                    new_values = rewritten_df[new_col].fillna("__NA__").astype(str).tolist()
                    if orig_values == new_values:
                        validation["column_mapping"][orig_col] = new_col
                        break
                except Exception:
                    continue
        
        # Sample comparison to verify values weren't modified
        # Check if at least 80% of original columns have matching values in rewritten df
        matched_columns = len(validation["column_mapping"])
        total_columns = len(original_df.columns)
        
        if total_columns == 0:
            validation["values_preserved"] = True
            return validation
        
        match_ratio = matched_columns / total_columns
        
        if match_ratio < 0.8:
            validation["values_preserved"] = False
            missing_cols = [c for c in original_df.columns if c not in validation["column_mapping"]]
            validation["value_changes"] = f"Only {matched_columns}/{total_columns} columns preserved their values. Missing: {missing_cols[:5]}"
        else:
            validation["values_preserved"] = True
        
        return validation
    
    def process(
        self,
        df: pd.DataFrame,
        schema_name: str = "IPA Standard",
        auto_fix: bool = True,
        max_attempts: int = 3,
        existing_diagnosis: Optional[DiagnosisResult] = None,
        diagnosis_context: Optional[str] = None,
        fixing_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a DataFrame through the complete pipeline.
        
        Pipeline order:
        1. Rules-based preprocessing (before LLM loading)
        2. Optional LLM-Rewrite step (structural reorganization only, no value changes)
        3. Load LLM model
        4. Agentic diagnosis (or use existing_diagnosis if provided)
        5. Agentic fix loop (if needed) - can transform values
        
        Args:
            df: Input DataFrame
            schema_name: Name of target schema
            auto_fix: Whether to attempt automatic fixing
            max_attempts: Maximum fix attempts
            existing_diagnosis: Optional pre-existing diagnosis result to reuse
            diagnosis_context: Optional additional context for diagnosis
            fixing_context: Optional additional context for fixing
            
        Returns:
            Dict with processing results
        """
        schema = self.schema_manager.get_schema(schema_name)
        if not schema:
            return {"error": f"Schema '{schema_name}' not found"}
        
        results = {
            "original_shape": df.shape,
            "schema": schema_name,
            "preprocessing": None,
            "diagnosis": None,
            "fix_attempts": [],
            "final_df": None,
            "success": False
        }
        
        # Phase 1: Rules-based preprocessing (BEFORE loading LLM)
        # This saves time by fixing simple issues without needing the LLM
        log_pipeline_stage("PREPROCESSING", f"DataFrame shape: {df.shape}, Target schema: {schema_name}")
        debug(f"Starting rules-based preprocessing on DataFrame with {len(df)} rows, {len(df.columns)} columns", context="Pipeline")
        
        # Create checkpoint before preprocessing (original data)
        if CHECKPOINT_AVAILABLE and self.checkpoint_manager:
            self.checkpoint_manager.create_checkpoint(
                ProcessingStage.PREPROCESSING,
                df=df,
                metadata={"action": "before_preprocessing", "stage": "input"}
            )
        
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.ANALYZING)
            notify_progress(0.05)
            notify_thought(
                "Starting rules-based preprocessing...\n"
                "Fixing simple issues before loading LLM.",
                AgentPhase.ANALYZING
            )
        
        preprocessing_result = self.preprocessor.preprocess(df, schema, self.schema_manager)
        
        debug(f"Preprocessing complete: {len(preprocessing_result.changes_made)} changes made, {len(preprocessing_result.issues_remaining)} issues remaining", context="Pipeline")
        
        results["preprocessing"] = {
            "changes_made": preprocessing_result.changes_made,
            "issues_remaining": len(preprocessing_result.issues_remaining),
            "has_changes": preprocessing_result.has_changes,
            "report": preprocessing_result.preprocessing_report
        }
        
        if MONITOR_AVAILABLE:
            notify_progress(0.1)
            if preprocessing_result.has_changes:
                notify_thought(
                    f"Preprocessing complete:\n"
                    f"  Changes applied: {len(preprocessing_result.changes_made)}\n"
                    f"  Remaining issues: {len(preprocessing_result.issues_remaining)}\n"
                    f"Using preprocessed data for further processing.",
                    AgentPhase.ANALYZING
                )
            else:
                notify_thought(
                    "Preprocessing complete: No automatic fixes needed.",
                    AgentPhase.ANALYZING
                )
        
        # Use preprocessed DataFrame for further processing
        current_df = preprocessing_result.preprocessed_df
        # #region agent log
        import json
        try:
            with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_L","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1384","message":"After preprocessing - setting current_df","data":{"current_df_columns":list(current_df.columns),"current_df_shape":current_df.shape,"original_df_columns":list(df.columns),"original_df_shape":df.shape,"schema_columns":schema.get_column_names(),"hypothesisId":"L"},"sessionId":"debug-session","runId":"run2"}) + '\n')
        except: pass
        # #endregion
        
        # Create checkpoint after preprocessing (preprocessed data)
        if CHECKPOINT_AVAILABLE and self.checkpoint_manager:
            self.checkpoint_manager.create_checkpoint(
                ProcessingStage.PREPROCESSING,
                df=current_df,
                metadata={
                    "action": "after_preprocessing",
                    "stage": "preprocessed",
                    "changes_made": len(preprocessing_result.changes_made),
                    "issues_remaining": len(preprocessing_result.issues_remaining)
                }
            )
        
        # Note: Even if preprocessing says it's valid, we still run the diagnostic agent
        # to ensure the output is fully correct before marking as finished
        if preprocessing_result.is_valid:
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_G","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1400","message":"Preprocessing valid - continuing to diagnostic agent","data":{"preprocessing_valid":True,"current_df_columns":list(current_df.columns),"schema_columns":schema.get_column_names(),"hypothesisId":"G"},"sessionId":"debug-session","runId":"run2"}) + '\n')
            except: pass
            # #endregion
            if MONITOR_AVAILABLE:
                notify_thought(
                    "Preprocessing completed successfully.\n"
                    "Continuing to LLM diagnostic agent for final verification...",
                    AgentPhase.ANALYZING
                )
        
        # Check processing mode and LLM availability
        try:
            from .utils.preferences import get_preferences
            from .utils.llm_check import is_llm_available
            prefs = get_preferences()
            processing_mode = prefs.get_processing_mode()
            llm_available = is_llm_available()
            should_use_llm = prefs.should_use_llm()
        except Exception:
            processing_mode = "auto"
            llm_available = False
            should_use_llm = False
        
        # If LLM is required but not available, raise error
        if processing_mode == "llm_required" and not llm_available:
            error_msg = "LLM is required but not installed. Please install LLM support from Preferences."
            if MONITOR_AVAILABLE:
                notify_thought(
                    f"Error: {error_msg}",
                    AgentPhase.ERROR
                )
            return {
                "error": error_msg,
                "original_shape": df.shape,
                "schema": schema_name,
                "preprocessing": results["preprocessing"],
                "success": False
            }
        
        # Phase 2: Optional LLM-Rewrite step (structural reorganization only)
        # Check if llm_rewrite is enabled in preferences (only if LLM is available)
        llm_rewrite_enabled = False
        if should_use_llm:
            try:
                llm_rewrite_enabled = prefs.get("llm_rewrite_enabled", False)
            except Exception:
                llm_rewrite_enabled = False
        
        results["llm_rewrite"] = None
        
        if llm_rewrite_enabled and should_use_llm:
            if MONITOR_AVAILABLE:
                notify_phase(AgentPhase.ANALYZING)
                notify_progress(0.15)
                notify_thought(
                    "LLM-Rewrite step enabled.\n"
                    "This will reorganize data structure to match schema.\n"
                    "Data values will NOT be modified - only column names/organization.",
                    AgentPhase.ANALYZING
                )
            
            # Perform LLM rewrite (structural transformation only)
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_J","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1439","message":"Before LLM rewrite - checking dataframe","data":{"current_df_columns":list(current_df.columns),"current_df_shape":current_df.shape,"schema_columns":schema.get_column_names(),"hypothesisId":"J"},"sessionId":"debug-session","runId":"run2"}) + '\n')
            except: pass
            # #endregion
            rewritten_df, rewrite_report = self._llm_rewrite(current_df, schema)
            
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_E","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1441","message":"After LLM rewrite","data":{"rewrite_success":rewrite_report.get("success",False),"rewritten_columns":list(rewritten_df.columns) if rewrite_report.get("success") else [],"schema_columns":schema.get_column_names(),"hypothesisId":"E"},"sessionId":"debug-session","runId":"run1"}) + '\n')
            except: pass
            # #endregion
            
            results["llm_rewrite"] = rewrite_report
            
            if rewrite_report["success"]:
                # Use rewritten DataFrame
                current_df = rewritten_df
                
                # Create checkpoint after LLM rewrite (rewritten data)
                if CHECKPOINT_AVAILABLE and self.checkpoint_manager:
                    self.checkpoint_manager.create_checkpoint(
                        ProcessingStage.PREPROCESSING,
                        df=current_df,
                        metadata={
                            "action": "after_llm_rewrite",
                            "stage": "rewritten",
                            "column_mapping": len(rewrite_report.get('column_mapping', {}))
                        }
                    )
                
                if MONITOR_AVAILABLE:
                    notify_progress(0.2)
                    notify_thought(
                        f"LLM-Rewrite successful!\n"
                        f"Column mappings: {len(rewrite_report.get('column_mapping', {}))} columns mapped\n"
                        f"Data values verified unchanged.",
                        AgentPhase.ANALYZING
                    )
            else:
                # Rewrite failed validation or had error - use preprocessed DataFrame
                if MONITOR_AVAILABLE:
                    notify_thought(
                        f"LLM-Rewrite skipped or failed: {rewrite_report.get('error', 'Unknown error')}\n"
                        "Continuing with preprocessed data.",
                        AgentPhase.ANALYZING
                    )
        else:
            if MONITOR_AVAILABLE:
                notify_thought(
                    "LLM-Rewrite step disabled (can be enabled in preferences).",
                    AgentPhase.ANALYZING
                )
        
        # Phase 3: Load LLM for remaining complex issues (only if should use LLM)
        if should_use_llm:
            try:
                self._ensure_llm_loaded()
            except Exception as e:
                # If LLM loading fails and we're in auto mode, fall back to rules-only
                if processing_mode == "auto":
                    warning(f"LLM loading failed, falling back to rules-only mode: {e}", context="Pipeline")
                    should_use_llm = False
                    if MONITOR_AVAILABLE:
                        notify_thought(
                            f"LLM loading failed: {str(e)}\n"
                            "Falling back to rules-based processing only.",
                            AgentPhase.ANALYZING
                        )
                else:
                    # In llm_required mode, we already checked above, so this shouldn't happen
                    raise
        else:
            if MONITOR_AVAILABLE:
                mode_name = "rules-only" if processing_mode == "rules_only" else "auto (LLM not available)"
                notify_thought(
                    f"Processing in {mode_name} mode.\n"
                    "Skipping LLM-based diagnosis and fixing.",
                    AgentPhase.ANALYZING
                )
        
        # Transform dataframe to match schema structure before diagnostic agent
        # This ensures the diagnostic agent sees the data in its final form
        # (unless LLM rewrite already succeeded, in which case it's already transformed)
        llm_rewrite_succeeded = results.get("llm_rewrite") and results["llm_rewrite"].get("success", False)
        if not llm_rewrite_succeeded:
            # LLM rewrite didn't succeed or wasn't enabled - transform to schema now
            if MONITOR_AVAILABLE:
                notify_thought(
                    "Transforming data structure to match schema before diagnostic verification...",
                    AgentPhase.ANALYZING
                )
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_M","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1511","message":"Before schema transform for diagnostic","data":{"current_df_columns":list(current_df.columns),"schema_columns":schema.get_column_names(),"hypothesisId":"M"},"sessionId":"debug-session","runId":"run2"}) + '\n')
            except: pass
            # #endregion
            current_df = self.schema_manager.transform_dataframe(current_df, schema)
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_N","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1520","message":"After schema transform for diagnostic","data":{"transformed_df_columns":list(current_df.columns),"schema_columns":schema.get_column_names(),"columns_match":list(current_df.columns) == schema.get_column_names(),"hypothesisId":"N"},"sessionId":"debug-session","runId":"run2"}) + '\n')
            except: pass
            # #endregion
        
        # Create checkpoint before diagnosis (current data state)
        if CHECKPOINT_AVAILABLE and self.checkpoint_manager:
            self.checkpoint_manager.create_checkpoint(
                ProcessingStage.DIAGNOSING,
                df=current_df,
                metadata={"action": "before_diagnosis", "stage": "pre_diagnosis"}
            )
        
        # Phase 4: Agentic diagnosis (or use existing diagnosis if provided)
        # Skip if not using LLM
        if not should_use_llm:
            # In rules-only mode, create a simple diagnosis from preprocessing results
            # DiagnosisResult is defined in this file
            diagnosis = DiagnosisResult(
                is_valid=preprocessing_result.is_valid,
                quality_score=1.0 if preprocessing_result.is_valid else 0.5,
                issues=preprocessing_result.issues_remaining,
                summary="Rules-based preprocessing completed. " + 
                       ("All issues resolved." if preprocessing_result.is_valid 
                        else f"{len(preprocessing_result.issues_remaining)} issues remain (LLM required for complex fixes).")
            )
            if MONITOR_AVAILABLE:
                notify_thought(
                    "Rules-based processing complete.\n"
                    f"  Valid: {diagnosis.is_valid}\n"
                    f"  Remaining issues: {len(diagnosis.issues)}",
                    AgentPhase.DIAGNOSING
                )
        elif existing_diagnosis is not None:
            # Use the provided existing diagnosis
            diagnosis = existing_diagnosis
            info("Using existing diagnosis result (skipping re-diagnosis)", context="Pipeline")
            if MONITOR_AVAILABLE:
                notify_thought(
                    "Using previously computed diagnosis result.\n"
                    f"  Quality Score: {diagnosis.quality_score:.2f}\n"
                    f"  Issues: {len(diagnosis.issues)}",
                    AgentPhase.DIAGNOSING
                )
        else:
            # Run new diagnosis
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_H","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1533","message":"Running diagnostic agent","data":{"current_df_columns":list(current_df.columns),"current_df_shape":current_df.shape,"schema_columns":schema.get_column_names(),"preprocessing_valid":preprocessing_result.is_valid,"hypothesisId":"H"},"sessionId":"debug-session","runId":"run2"}) + '\n')
            except: pass
            # #endregion
            diagnosis = self.diagnostic_agent.diagnose(
                current_df, 
                schema, 
                self.schema_manager,
                preprocessing_report=preprocessing_result.preprocessing_report,
                additional_context=diagnosis_context
            )
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_I","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1540","message":"Diagnostic agent completed","data":{"diagnosis_valid":diagnosis.is_valid,"quality_score":diagnosis.quality_score,"issues_count":len(diagnosis.issues),"hypothesisId":"I"},"sessionId":"debug-session","runId":"run2"}) + '\n')
            except: pass
            # #endregion
        
        results["diagnosis"] = {
            "is_valid": diagnosis.is_valid,
            "quality_score": diagnosis.quality_score,
            "issues_count": len(diagnosis.issues),
            "issues": diagnosis.issues,  # Include full issues list for UI display
            "summary": diagnosis.summary
        }
        
        # Create checkpoint after diagnosis (current data state)
        if CHECKPOINT_AVAILABLE and self.checkpoint_manager:
            self.checkpoint_manager.create_checkpoint(
                ProcessingStage.DIAGNOSING,
                df=current_df,
                metadata={
                    "action": "after_diagnosis",
                    "stage": "post_diagnosis",
                    "quality_score": diagnosis.quality_score,
                    "issues_count": len(diagnosis.issues),
                    "is_valid": diagnosis.is_valid
                }
            )
        
        log_pipeline_stage("DIAGNOSIS COMPLETE", f"Valid: {diagnosis.is_valid}, Quality: {diagnosis.quality_score:.2f}, Issues: {len(diagnosis.issues)}")
        debug(f"Diagnosis summary: {diagnosis.summary[:200] if diagnosis.summary else 'No summary'}", context="Pipeline")
        
        # Clear CUDA cache after diagnosis to free memory
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if diagnosis.is_valid:
            info("Data validation passed - no fixes needed", context="Pipeline")
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_F","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1548","message":"Diagnosis valid - setting final_df without fixes","data":{"current_df_columns":list(current_df.columns),"schema_columns":schema.get_column_names(),"hypothesisId":"F"},"sessionId":"debug-session","runId":"run1"}) + '\n')
            except: pass
            # #endregion
            # Transform to match schema before setting final_df
            transformed_df = self.schema_manager.transform_dataframe(current_df, schema)
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_FIX","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1570","message":"After transform - diagnosis valid case","data":{"transformed_columns":list(transformed_df.columns),"schema_columns":schema.get_column_names(),"columns_match":list(transformed_df.columns) == schema.get_column_names(),"runId":"post-fix"},"sessionId":"debug-session"}) + '\n')
            except: pass
            # #endregion
            results["final_df"] = transformed_df
            results["success"] = True
            return results
        
        if not auto_fix:
            info("Auto-fix disabled, returning diagnosis results only", context="Pipeline")
            return results
        
        # Skip fix loop if LLM is not available
        if not should_use_llm:
            info("LLM not available - skipping fix loop. Remaining issues require LLM support.", context="Pipeline")
            if MONITOR_AVAILABLE:
                notify_thought(
                    "Fix loop skipped: LLM required for complex fixes.\n"
                    f"  Remaining issues: {len(diagnosis.issues)}\n"
                    "  Install LLM support to enable automatic fixing.",
                    AgentPhase.FIXING
                )
            # Return preprocessed data as final result
            transformed_df = self.schema_manager.transform_dataframe(current_df, schema)
            results["final_df"] = transformed_df
            results["success"] = diagnosis.is_valid  # Success only if preprocessing resolved all issues
            return results
        
        # Phase 5: Agentic fix loop
        log_pipeline_stage("FIX LOOP", f"Starting fix loop with max {max_attempts} attempts")
        attempt_count = 0  # Only count successful script executions
        
        for attempt in range(max_attempts * 2):  # Allow more iterations for debug retries
            debug(f"Fix iteration {attempt + 1} (attempt count: {attempt_count}/{max_attempts})", context="FixLoop")
            fix_result = self.fixing_agent.fix(current_df, diagnosis, schema, additional_context=fixing_context)
            
            # Clear CUDA cache after each fix attempt to free memory
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Only count as an attempt if script executed successfully (even if it didn't fix the data)
            # Execution failures are debugged automatically and don't count as attempts
            if not fix_result.is_execution_error:
                # Script executed (either successfully or with logical error) - count as attempt
                attempt_count += 1
                results["fix_attempts"].append({
                    "attempt": attempt_count,
                    "success": fix_result.success,
                    "execution_time": fix_result.execution_time,
                    "error": fix_result.error_message,
                    "changes": fix_result.changes_made
                })
                
                if fix_result.success:
                    info(f"Fix attempt {attempt_count} succeeded in {fix_result.execution_time:.2f}s", context="FixLoop")
                else:
                    warning(f"Fix attempt {attempt_count} executed but didn't fix data: {fix_result.error_message[:100] if fix_result.error_message else 'Unknown error'}", context="FixLoop")
            else:
                # Execution error - was debugged automatically, don't count as attempt
                debug(f"Script execution error (was debugged, not counting as attempt): {fix_result.error_message[:100] if fix_result.error_message else 'Unknown error'}", context="FixLoop")
            
            # Only update dataframe and re-diagnose if fix was successful
            if fix_result.success and fix_result.fixed_df is not None:
                current_df = fix_result.fixed_df
                
                # Create checkpoint after each successful fix
                if CHECKPOINT_AVAILABLE and self.checkpoint_manager:
                    self.checkpoint_manager.create_checkpoint(
                        ProcessingStage.FIX_ATTEMPT,
                        df=current_df,
                        metadata={
                            "action": "after_fix",
                            "stage": f"fix_attempt_{attempt_count}",
                            "attempt": attempt_count,
                            "success": True
                        }
                    )
                
                # Clear CUDA cache before re-diagnosis
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Re-diagnose to check if issues are resolved
                diagnosis = self.diagnostic_agent.diagnose(
                    current_df, schema, self.schema_manager
                )
                
                # Clear CUDA cache after re-diagnosis
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # If all issues are resolved, we're done
                if diagnosis.is_valid:
                    # #region agent log
                    import json
                    try:
                        with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_A","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1621","message":"Setting final_df before transform","data":{"current_df_columns":list(current_df.columns),"schema_columns":schema.get_column_names(),"hypothesisId":"A"},"sessionId":"debug-session","runId":"run1"}) + '\n')
                    except: pass
                    # #endregion
                    # Transform to match schema before setting final_df
                    transformed_df = self.schema_manager.transform_dataframe(current_df, schema)
                    # #region agent log
                    import json
                    try:
                        with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_FIX","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1650","message":"After transform - fix loop success case","data":{"transformed_columns":list(transformed_df.columns),"schema_columns":schema.get_column_names(),"columns_match":list(transformed_df.columns) == schema.get_column_names(),"runId":"post-fix"},"sessionId":"debug-session"}) + '\n')
                    except: pass
                    # #endregion
                    results["final_df"] = transformed_df
                    results["success"] = True
                    
                    # Create final checkpoint
                    if CHECKPOINT_AVAILABLE and self.checkpoint_manager:
                        self.checkpoint_manager.create_checkpoint(
                            ProcessingStage.COMPLETE,
                            df=current_df,
                            metadata={
                                "action": "complete",
                                "stage": "final",
                                "attempts": attempt_count,
                                "success": True
                            }
                        )
                    
                    info(f"All issues resolved after {attempt_count} attempt(s)", context="FixLoop")
                    break
                else:
                    # Fix succeeded but issues remain, continue to next attempt
                    info(f"Fix attempt {attempt_count} succeeded but {len(diagnosis.issues)} issue(s) remain, continuing...", context="FixLoop")
            
            # Check if we've reached max attempts (only counting successful executions)
            if attempt_count >= max_attempts:
                info(f"Reached maximum fix attempts ({max_attempts})", context="FixLoop")
                break
            
            # If fix failed (logical error, not execution error), continue to next attempt
            # The loop will naturally end when max_attempts is reached or max iterations exceeded
        
        if not results["success"]:
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_A","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1652","message":"Setting final_df (not successful) before transform","data":{"current_df_columns":list(current_df.columns),"schema_columns":schema.get_column_names(),"hypothesisId":"A"},"sessionId":"debug-session","runId":"run1"}) + '\n')
            except: pass
            # #endregion
            # Transform to match schema before setting final_df (even if not successful, still transform)
            transformed_df = self.schema_manager.transform_dataframe(current_df, schema)
            # #region agent log
            import json
            try:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_FIX","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1688","message":"After transform - not successful case","data":{"transformed_columns":list(transformed_df.columns),"schema_columns":schema.get_column_names(),"columns_match":list(transformed_df.columns) == schema.get_column_names(),"runId":"post-fix"},"sessionId":"debug-session"}) + '\n')
            except: pass
            # #endregion
            results["final_df"] = transformed_df
            # Create checkpoint for final state even if not successful
            if CHECKPOINT_AVAILABLE and self.checkpoint_manager:
                self.checkpoint_manager.create_checkpoint(
                    ProcessingStage.COMPLETE,
                    df=current_df,
                    metadata={
                        "action": "complete",
                        "stage": "final",
                        "attempts": attempt_count,
                        "success": False
                    }
                )
        
        # #region agent log
        import json
        try:
            final_df_before_transform = results.get("final_df")
            if final_df_before_transform is not None:
                with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_B","timestamp":int(datetime.now().timestamp()*1000),"location":"agents.py:1666","message":"Before return - checking if transform_dataframe called","data":{"final_df_columns":list(final_df_before_transform.columns),"schema_columns":schema.get_column_names(),"columns_match":list(final_df_before_transform.columns) == schema.get_column_names(),"hypothesisId":"B"},"sessionId":"debug-session","runId":"run1"}) + '\n')
        except: pass
        # #endregion
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the system."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "available_schemas": self.schema_manager.list_schemas()
        }
