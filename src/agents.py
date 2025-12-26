"""
Agentic Loop for BIDS

Diagnostic Agent (Gemini): Analyzes data issues, avoids failed approaches
Fixing Agent (Claude Opus): Generates fix scripts using successful examples
"""

import io
import sys
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

from .utils.llm_client import GeminiClient, ClaudeClient, get_gemini_client, get_claude_client
from .vectorizer import NoiseVectorizer, DataFingerprint
from .vector_store import VectorStore, ScriptStatus
from .schema_manager import Schema, SchemaManager, ValidationResult


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
    Diagnostic Agent powered by Gemini.
    
    Responsibilities:
    - Analyze DataFrame for quality issues
    - Validate against target schema
    - Query vector store for failed approaches to avoid
    - Generate structured diagnosis report
    """
    
    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        vector_store: Optional[VectorStore] = None,
        vectorizer: Optional[NoiseVectorizer] = None
    ):
        """
        Initialize DiagnosticAgent.
        
        Args:
            gemini_client: Gemini LLM client
            vector_store: Vector store for failed script lookup
            vectorizer: Vectorizer for fingerprinting
        """
        self.llm = gemini_client or get_gemini_client()
        self.vector_store = vector_store
        self.vectorizer = vectorizer or NoiseVectorizer()
    
    def diagnose(
        self,
        df: pd.DataFrame,
        schema: Schema,
        schema_manager: Optional[SchemaManager] = None
    ) -> DiagnosisResult:
        """
        Perform comprehensive diagnosis on a DataFrame.
        
        Args:
            df: DataFrame to diagnose
            schema: Target output schema
            schema_manager: Schema manager for validation
            
        Returns:
            DiagnosisResult object
        """
        # Get schema validation
        schema_validation = None
        if schema_manager:
            schema_validation = schema_manager.validate_dataframe(df, schema)
        
        # Get failed scripts to avoid
        failed_scripts = []
        avoided_approaches = []
        if self.vector_store:
            fingerprint = self.vectorizer.create_fingerprint(df)
            # Try to fit and transform if not already fitted
            try:
                self.vectorizer.fit([fingerprint])
                self.vectorizer.transform(fingerprint)
                failed_scripts = self.vector_store.get_failed_scripts(fingerprint, top_k=5)
                avoided_approaches = [s[:200] + "..." for s in failed_scripts]
            except Exception:
                pass
        
        # Prepare DataFrame info for LLM
        df_info = self._prepare_df_info(df)
        schema_rules = schema.get_validation_rules()
        
        # Call Gemini for diagnosis
        diagnosis = self.llm.diagnose_data(df_info, schema_rules, failed_scripts)
        
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
    Fixing Agent powered by Claude Opus.
    
    Responsibilities:
    - Generate Python fix scripts based on diagnosis
    - Use successful scripts from vector store as examples
    - Execute scripts safely
    - Record success/failure in vector store
    """
    
    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
        vector_store: Optional[VectorStore] = None,
        vectorizer: Optional[NoiseVectorizer] = None
    ):
        """
        Initialize FixingAgent.
        
        Args:
            claude_client: Claude LLM client
            vector_store: Vector store for script lookup and storage
            vectorizer: Vectorizer for fingerprinting
        """
        self.llm = claude_client or get_claude_client()
        self.vector_store = vector_store
        self.vectorizer = vectorizer or NoiseVectorizer()
        
        # Sandbox for executing generated code
        self._sandbox_globals = {
            'pd': pd,
            'np': np,
            're': __import__('re'),
        }
    
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
        # Get successful scripts as examples
        successful_scripts = []
        fingerprint = None
        
        if self.vector_store:
            fingerprint = self.vectorizer.create_fingerprint(df)
            try:
                self.vectorizer.fit([fingerprint])
                self.vectorizer.transform(fingerprint)
                successful_scripts = self.vector_store.get_successful_scripts(fingerprint, top_k=3)
            except Exception:
                pass
        
        # Prepare sample data for LLM
        df_sample = df.head(10).to_string()
        
        # Generate fix script using Claude Opus
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
        
        # Execute the script
        start_time = datetime.now()
        fixed_df, error_message, changes = self._execute_fix(df, script)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        success = fixed_df is not None and error_message is None
        
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
        
        self.diagnostic_agent = DiagnosticAgent(
            vector_store=self.vector_store,
            vectorizer=self.vectorizer
        )
        
        self.fixing_agent = FixingAgent(
            vector_store=self.vector_store,
            vectorizer=self.vectorizer
        )
    
    def process(
        self,
        df: pd.DataFrame,
        schema_name: str = "IPA Standard",
        auto_fix: bool = True,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Process a DataFrame through the complete pipeline.
        
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
            "diagnosis": None,
            "fix_attempts": [],
            "final_df": None,
            "success": False
        }
        
        # Initial diagnosis
        diagnosis = self.diagnostic_agent.diagnose(df, schema, self.schema_manager)
        results["diagnosis"] = {
            "is_valid": diagnosis.is_valid,
            "quality_score": diagnosis.quality_score,
            "issues_count": len(diagnosis.issues),
            "summary": diagnosis.summary
        }
        
        if diagnosis.is_valid:
            results["final_df"] = df
            results["success"] = True
            return results
        
        if not auto_fix:
            return results
        
        # Attempt fixes
        current_df = df
        for attempt in range(max_attempts):
            fix_result = self.fixing_agent.fix(current_df, diagnosis, schema)
            
            results["fix_attempts"].append({
                "attempt": attempt + 1,
                "success": fix_result.success,
                "execution_time": fix_result.execution_time,
                "error": fix_result.error_message,
                "changes": fix_result.changes_made
            })
            
            if fix_result.success and fix_result.fixed_df is not None:
                current_df = fix_result.fixed_df
                
                # Re-diagnose
                diagnosis = self.diagnostic_agent.diagnose(
                    current_df, schema, self.schema_manager
                )
                
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
