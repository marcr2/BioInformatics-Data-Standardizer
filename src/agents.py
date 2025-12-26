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
from .vectorizer import NoiseVectorizer, DataFingerprint
from .vector_store import VectorStore, ScriptStatus
from .schema_manager import Schema, SchemaManager, ValidationResult
from .preprocessor import RulesBasedPreprocessor, PreprocessingResult

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
        preprocessing_report: Optional[str] = None
    ) -> DiagnosisResult:
        """
        Perform comprehensive diagnosis on a DataFrame.
        
        Args:
            df: DataFrame to diagnose
            schema: Target output schema
            schema_manager: Schema manager for validation
            preprocessing_report: Optional report from rules-based preprocessing
            
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
            buffer.write(df.describe(include='all').to_string())
        except Exception:
            buffer.write("(Could not compute statistics)")
        
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
        schema: Schema
    ) -> FixResult:
        """
        Generate and execute a fix script.
        
        Args:
            df: DataFrame to fix
            diagnosis: Diagnosis results
            schema: Target output schema
            
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
            successful_scripts=successful_scripts
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
        fixed_df, error_message, changes = self._execute_fix(df, script)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        success = fixed_df is not None and error_message is None
        
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
            changes_made=changes
        )
    
    def _execute_fix(
        self,
        df: pd.DataFrame,
        script: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str]]:
        """
        Execute a fix script safely.
        
        Args:
            df: Input DataFrame
            script: Python script to execute
            
        Returns:
            Tuple of (fixed_df, error_message, changes_made)
        """
        changes = []
        
        try:
            # Prepare sandbox environment
            sandbox = self._sandbox_globals.copy()
            sandbox['df'] = df.copy()
            sandbox['__builtins__'] = {
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
            }
            
            # Execute the script
            exec(script, sandbox)
            
            # Look for the result
            if 'fix_dataframe' in sandbox and callable(sandbox['fix_dataframe']):
                # If a function was defined, call it
                result = sandbox['fix_dataframe'](df.copy())
            elif 'result' in sandbox and isinstance(sandbox['result'], pd.DataFrame):
                result = sandbox['result']
            elif 'df' in sandbox and isinstance(sandbox['df'], pd.DataFrame):
                result = sandbox['df']
            else:
                return None, "Script did not produce a DataFrame result", changes
            
            # Validate result
            if not isinstance(result, pd.DataFrame):
                return None, f"Result is not a DataFrame: {type(result)}", changes
            
            if len(result) == 0:
                return None, "Result DataFrame is empty", changes
            
            # Track changes
            if len(result.columns) != len(df.columns):
                changes.append(f"Columns changed: {len(df.columns)} -> {len(result.columns)}")
            if len(result) != len(df):
                changes.append(f"Rows changed: {len(df)} -> {len(result)}")
            
            return result, None, changes
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg, changes
    
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
        fixed_df, error_message, changes = self._execute_fix(df, script)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return FixResult(
            success=fixed_df is not None,
            fixed_df=fixed_df,
            script_content=script,
            execution_time=execution_time,
            error_message=error_message,
            changes_made=changes
        )


class AgentOrchestrator:
    """
    Orchestrates the diagnostic and fixing agents.
    
    Provides a high-level interface for the complete
    diagnosis -> fix -> validate cycle.
    """
    
    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        schemas_dir: Optional[str] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            vector_store_path: Path for persistent vector store
            schemas_dir: Directory for schemas
        """
        from .vector_store import get_vector_store
        from .schema_manager import get_schema_manager
        
        self.vector_store = get_vector_store(vector_store_path)
        self.schema_manager = get_schema_manager(schemas_dir)
        self.vectorizer = NoiseVectorizer()
        
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
                print("Loading LLM model (this may take 10-30 seconds on first use)...")
                self._shared_llm_client = get_gemini_client()
                
                # Update agents with the shared client
                self.diagnostic_agent.llm = self._shared_llm_client
                self.fixing_agent.llm = self._shared_llm_client
                
                print("LLM model loaded successfully!")
                
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
    
    def process(
        self,
        df: pd.DataFrame,
        schema_name: str = "IPA Standard",
        auto_fix: bool = True,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Process a DataFrame through the complete pipeline.
        
        Pipeline order:
        1. Rules-based preprocessing (before LLM loading)
        2. Load LLM model
        3. Agentic diagnosis
        4. Agentic fix loop (if needed)
        
        Args:
            df: Input DataFrame
            schema_name: Name of target schema
            auto_fix: Whether to attempt automatic fixing
            max_attempts: Maximum fix attempts
            
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
        if MONITOR_AVAILABLE:
            notify_phase(AgentPhase.ANALYZING)
            notify_progress(0.05)
            notify_thought(
                "Starting rules-based preprocessing...\n"
                "Fixing simple issues before loading LLM.",
                AgentPhase.ANALYZING
            )
        
        preprocessing_result = self.preprocessor.preprocess(df, schema, self.schema_manager)
        
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
        
        # If preprocessing resolved all issues, we might not need LLM at all
        if preprocessing_result.is_valid:
            results["final_df"] = current_df
            results["success"] = True
            results["diagnosis"] = {
                "is_valid": True,
                "quality_score": 1.0,
                "issues_count": 0,
                "summary": "All issues resolved by rules-based preprocessing"
            }
            if MONITOR_AVAILABLE:
                notify_phase(AgentPhase.COMPLETE)
                notify_progress(1.0)
                notify_thought(
                    "All issues resolved by preprocessing! No LLM needed.",
                    AgentPhase.COMPLETE
                )
            return results
        
        # Phase 2: Load LLM for remaining complex issues
        self._ensure_llm_loaded()
        
        # Phase 3: Agentic diagnosis (with preprocessing context)
        diagnosis = self.diagnostic_agent.diagnose(
            current_df, 
            schema, 
            self.schema_manager,
            preprocessing_report=preprocessing_result.preprocessing_report
        )
        results["diagnosis"] = {
            "is_valid": diagnosis.is_valid,
            "quality_score": diagnosis.quality_score,
            "issues_count": len(diagnosis.issues),
            "summary": diagnosis.summary
        }
        
        # Clear CUDA cache after diagnosis to free memory
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if diagnosis.is_valid:
            results["final_df"] = current_df
            results["success"] = True
            return results
        
        if not auto_fix:
            return results
        
        # Phase 4: Agentic fix loop
        for attempt in range(max_attempts):
            fix_result = self.fixing_agent.fix(current_df, diagnosis, schema)
            
            # Clear CUDA cache after each fix attempt to free memory
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            results["fix_attempts"].append({
                "attempt": attempt + 1,
                "success": fix_result.success,
                "execution_time": fix_result.execution_time,
                "error": fix_result.error_message,
                "changes": fix_result.changes_made
            })
            
            if fix_result.success and fix_result.fixed_df is not None:
                current_df = fix_result.fixed_df
                
                # Clear CUDA cache before re-diagnosis
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Re-diagnose
                diagnosis = self.diagnostic_agent.diagnose(
                    current_df, schema, self.schema_manager
                )
                
                # Clear CUDA cache after re-diagnosis
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if diagnosis.is_valid:
                    results["final_df"] = current_df
                    results["success"] = True
                    break
            else:
                break
        
        if not results["success"]:
            results["final_df"] = current_df
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the system."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "available_schemas": self.schema_manager.list_schemas()
        }
