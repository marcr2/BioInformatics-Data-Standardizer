"""
Rules-Based Preprocessor for BIDS

Performs comprehensive data cleaning and automatic fixes before the agentic loop.
This runs BEFORE the LLM is loaded to fix simple issues quickly and save resources.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from .schema_manager import Schema, SchemaManager, ColumnDefinition, ColumnType, TransformType, ValidationResult


@dataclass
class PreprocessingResult:
    """Result of rules-based preprocessing."""
    preprocessed_df: pd.DataFrame
    changes_made: List[str] = field(default_factory=list)
    issues_remaining: List[Dict[str, Any]] = field(default_factory=list)
    preprocessing_report: str = ""
    column_mapping: Dict[str, str] = field(default_factory=dict)
    
    @property
    def has_changes(self) -> bool:
        """Check if any changes were made."""
        return len(self.changes_made) > 0
    
    @property
    def is_valid(self) -> bool:
        """Check if data is now valid (no remaining issues)."""
        return len(self.issues_remaining) == 0


class RulesBasedPreprocessor:
    """
    Rules-based preprocessing engine.
    
    Performs automatic fixes based on schema rules without needing an LLM:
    - Basic cleaning (empty columns, whitespace, duplicates)
    - Schema transformations (uppercase, lowercase, log transforms, etc.)
    - Type conversions
    - Range clamping
    - Pattern fixes (remove invalid characters)
    - Null handling
    """
    
    def __init__(self, schema_manager: Optional[SchemaManager] = None):
        """
        Initialize preprocessor.
        
        Args:
            schema_manager: Schema manager instance
        """
        self.schema_manager = schema_manager
    
    def preprocess(
        self,
        df: pd.DataFrame,
        schema: Schema,
        schema_manager: Optional[SchemaManager] = None
    ) -> PreprocessingResult:
        """
        Perform comprehensive preprocessing on a DataFrame.
        
        Args:
            df: Input DataFrame
            schema: Target schema
            schema_manager: Optional schema manager (overrides instance)
            
        Returns:
            PreprocessingResult with preprocessed DataFrame and report
        """
        sm = schema_manager or self.schema_manager
        changes_made = []
        issues_remaining = []
        
        # Work with a copy
        result_df = df.copy()
        
        # Step 1: Basic cleaning
        result_df, cleaning_changes = self._clean_dataframe(result_df)
        changes_made.extend(cleaning_changes)
        
        # Step 2: Get column mapping
        if sm:
            validation = sm.validate_dataframe(result_df, schema)
            column_mapping = validation.column_mapping
        else:
            column_mapping = self._auto_map_columns(result_df, schema)
        
        # Step 3: Apply schema transformations
        result_df, transform_changes = self._apply_schema_transforms(
            result_df, schema, column_mapping
        )
        changes_made.extend(transform_changes)
        
        # Step 4: Type conversions
        result_df, type_changes = self._convert_types(
            result_df, schema, column_mapping
        )
        changes_made.extend(type_changes)
        
        # Step 5: Fix pattern violations
        result_df, pattern_changes, pattern_issues = self._fix_pattern_violations(
            result_df, schema, column_mapping
        )
        changes_made.extend(pattern_changes)
        issues_remaining.extend(pattern_issues)
        
        # Step 6: Fix range violations
        result_df, range_changes, range_issues = self._fix_range_violations(
            result_df, schema, column_mapping
        )
        changes_made.extend(range_changes)
        issues_remaining.extend(range_issues)
        
        # Step 7: Handle null values
        result_df, null_changes, null_issues = self._handle_null_values(
            result_df, schema, column_mapping
        )
        changes_made.extend(null_changes)
        issues_remaining.extend(null_issues)
        
        # Generate report
        report = self._generate_report(changes_made, issues_remaining, schema)
        
        return PreprocessingResult(
            preprocessed_df=result_df,
            changes_made=changes_made,
            issues_remaining=issues_remaining,
            preprocessing_report=report,
            column_mapping=column_mapping
        )
    
    def _clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Basic DataFrame cleaning.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned_df, list of changes)
        """
        changes = []
        result = df.copy()
        original_shape = result.shape
        
        # Remove columns that are all NULL
        null_cols = result.columns[result.isnull().all()].tolist()
        if null_cols:
            result = result.drop(columns=null_cols)
            changes.append(f"Removed {len(null_cols)} all-null column(s): {', '.join(null_cols[:5])}")
        
        # Remove columns that are all empty (null, empty strings, or whitespace only)
        empty_cols = []
        for col in result.columns:
            # Check if column is effectively empty
            is_empty = True
            
            try:
                # For object/string columns: check for null, empty strings, or whitespace
                if result[col].dtype == 'object':
                    # Convert to string and check
                    col_str = result[col].astype(str)
                    # Remove 'nan' strings that came from actual NaN values
                    # Check if all values are null, empty, or whitespace
                    mask = ~col_str.isin(['nan', 'None', '']) & (col_str.str.strip() != '')
                    non_empty = col_str[mask]
                    if len(non_empty) > 0:
                        is_empty = False
                else:
                    # For numeric columns: check if all are null
                    if not result[col].isnull().all():
                        is_empty = False
            except Exception:
                # If checking fails, assume column has data (don't drop it)
                is_empty = False
            
            if is_empty:
                empty_cols.append(col)
        
        if empty_cols:
            result = result.drop(columns=empty_cols)
            changes.append(f"Removed {len(empty_cols)} empty column(s): {', '.join(empty_cols[:5])}")
        
        # Trim whitespace from string columns
        trimmed_count = 0
        for col in result.select_dtypes(include=['object']).columns:
            original = result[col].copy()
            result[col] = result[col].astype(str).str.strip()
            # Replace 'nan' string back to NaN
            result[col] = result[col].replace('nan', np.nan)
            if not original.equals(result[col]):
                trimmed_count += 1
        if trimmed_count > 0:
            changes.append(f"Trimmed whitespace in {trimmed_count} column(s)")
        
        # Remove duplicate rows
        dup_count = result.duplicated().sum()
        if dup_count > 0:
            result = result.drop_duplicates()
            changes.append(f"Removed {dup_count} duplicate row(s)")
        
        # Clean column names (remove leading/trailing spaces)
        cleaned_names = {col: col.strip() for col in result.columns if col != col.strip()}
        if cleaned_names:
            result = result.rename(columns=cleaned_names)
            changes.append(f"Cleaned {len(cleaned_names)} column name(s)")
        
        if result.shape != original_shape:
            changes.append(f"Shape changed: {original_shape} -> {result.shape}")
        
        return result, changes
    
    def _auto_map_columns(
        self,
        df: pd.DataFrame,
        schema: Schema
    ) -> Dict[str, str]:
        """
        Auto-map DataFrame columns to schema columns.
        
        Args:
            df: Input DataFrame
            schema: Target schema
            
        Returns:
            Dict mapping target column name -> source column name
        """
        column_mapping = {}
        
        for col_def in schema.columns:
            # Try exact match
            if col_def.name in df.columns:
                column_mapping[col_def.name] = col_def.name
                continue
            
            # Try case-insensitive match
            lower_name = col_def.name.lower()
            for col in df.columns:
                if col.lower() == lower_name:
                    column_mapping[col_def.name] = col
                    break
            
            if col_def.name in column_mapping:
                continue
            
            # Try pattern matching
            patterns = self._get_column_patterns(col_def.name)
            for col in df.columns:
                col_lower = col.lower()
                for pattern in patterns:
                    if re.search(pattern, col_lower):
                        column_mapping[col_def.name] = col
                        break
                if col_def.name in column_mapping:
                    break
        
        return column_mapping
    
    def _get_column_patterns(self, col_name: str) -> List[str]:
        """Get regex patterns for auto-detecting columns."""
        name_lower = col_name.lower()
        
        patterns_map = {
            "genesymbol": [r"gene", r"symbol", r"name", r"id"],
            "foldchange": [r"fold", r"fc", r"log2?fc", r"ratio", r"change"],
            "pvalue": [r"p[-_]?val", r"pvalue", r"significance", r"adj[-_]?p"],
        }
        
        return patterns_map.get(name_lower, [name_lower])
    
    def _apply_schema_transforms(
        self,
        df: pd.DataFrame,
        schema: Schema,
        column_mapping: Dict[str, str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply schema-defined transformations.
        
        Args:
            df: Input DataFrame
            schema: Target schema
            column_mapping: Column name mapping
            
        Returns:
            Tuple of (transformed_df, list of changes)
        """
        changes = []
        result = df.copy()
        
        for col_def in schema.columns:
            if col_def.name not in column_mapping:
                continue
            
            source_col = column_mapping[col_def.name]
            if source_col not in result.columns:
                continue
            
            if col_def.transform == TransformType.NONE:
                continue
            
            original = result[source_col].copy()
            
            try:
                if col_def.transform == TransformType.UPPERCASE:
                    result[source_col] = result[source_col].astype(str).str.upper()
                elif col_def.transform == TransformType.LOWERCASE:
                    result[source_col] = result[source_col].astype(str).str.lower()
                elif col_def.transform == TransformType.TRIM:
                    result[source_col] = result[source_col].astype(str).str.strip()
                elif col_def.transform == TransformType.LOG2:
                    numeric = pd.to_numeric(result[source_col], errors='coerce')
                    result[source_col] = np.log2(numeric.clip(lower=1e-10))
                elif col_def.transform == TransformType.LOG10:
                    numeric = pd.to_numeric(result[source_col], errors='coerce')
                    result[source_col] = np.log10(numeric.clip(lower=1e-10))
                elif col_def.transform == TransformType.ABS:
                    result[source_col] = pd.to_numeric(result[source_col], errors='coerce').abs()
                elif col_def.transform == TransformType.ROUND:
                    result[source_col] = pd.to_numeric(result[source_col], errors='coerce').round()
                elif col_def.transform == TransformType.NEGATE:
                    result[source_col] = -pd.to_numeric(result[source_col], errors='coerce')
                
                if not original.equals(result[source_col]):
                    changes.append(f"Applied {col_def.transform.value} transform to '{source_col}'")
                    
            except Exception as e:
                pass  # Skip transforms that fail
        
        return result, changes
    
    def _convert_types(
        self,
        df: pd.DataFrame,
        schema: Schema,
        column_mapping: Dict[str, str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert column types based on schema.
        
        Args:
            df: Input DataFrame
            schema: Target schema
            column_mapping: Column name mapping
            
        Returns:
            Tuple of (converted_df, list of changes)
        """
        changes = []
        result = df.copy()
        
        for col_def in schema.columns:
            if col_def.name not in column_mapping:
                continue
            
            source_col = column_mapping[col_def.name]
            if source_col not in result.columns:
                continue
            
            original_dtype = result[source_col].dtype
            
            try:
                if col_def.type == ColumnType.FLOAT:
                    result[source_col] = pd.to_numeric(result[source_col], errors='coerce')
                    if result[source_col].dtype != original_dtype:
                        changes.append(f"Converted '{source_col}' to float")
                        
                elif col_def.type == ColumnType.INT:
                    numeric = pd.to_numeric(result[source_col], errors='coerce')
                    result[source_col] = numeric.astype('Int64')
                    if result[source_col].dtype != original_dtype:
                        changes.append(f"Converted '{source_col}' to integer")
                        
                elif col_def.type == ColumnType.STRING:
                    if original_dtype != 'object':
                        result[source_col] = result[source_col].astype(str)
                        changes.append(f"Converted '{source_col}' to string")
                        
                elif col_def.type == ColumnType.BOOL:
                    # Convert common boolean representations
                    bool_map = {
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        '1': True, '0': False,
                        't': True, 'f': False,
                        'y': True, 'n': False,
                    }
                    result[source_col] = result[source_col].astype(str).str.lower().map(bool_map)
                    changes.append(f"Converted '{source_col}' to boolean")
                    
                elif col_def.type in (ColumnType.DATE, ColumnType.DATETIME):
                    result[source_col] = pd.to_datetime(result[source_col], errors='coerce')
                    if result[source_col].dtype != original_dtype:
                        changes.append(f"Converted '{source_col}' to datetime")
                        
            except Exception:
                pass  # Skip conversions that fail
        
        return result, changes
    
    def _fix_pattern_violations(
        self,
        df: pd.DataFrame,
        schema: Schema,
        column_mapping: Dict[str, str]
    ) -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
        """
        Fix regex pattern violations where possible.
        
        Args:
            df: Input DataFrame
            schema: Target schema
            column_mapping: Column name mapping
            
        Returns:
            Tuple of (fixed_df, changes, remaining_issues)
        """
        changes = []
        issues = []
        result = df.copy()
        
        for col_def in schema.columns:
            if not col_def.validation_regex:
                continue
            
            if col_def.name not in column_mapping:
                continue
            
            source_col = column_mapping[col_def.name]
            if source_col not in result.columns:
                continue
            
            pattern = re.compile(col_def.validation_regex)
            non_null = result[source_col].dropna()
            
            # Count violations before fix
            invalid_before = sum(1 for v in non_null if not pattern.match(str(v)))
            
            if invalid_before == 0:
                continue
            
            # Try common fixes based on pattern type
            fixed_count = 0
            
            # For gene symbols: remove invalid characters, keep alphanumeric and hyphen
            if 'gene' in col_def.name.lower() or 'symbol' in col_def.name.lower():
                original = result[source_col].copy()
                # Remove leading/trailing non-alpha characters
                result[source_col] = result[source_col].astype(str).str.strip()
                result[source_col] = result[source_col].str.replace(r'^[^A-Za-z]+', '', regex=True)
                result[source_col] = result[source_col].str.replace(r'[^A-Za-z0-9\-]+$', '', regex=True)
                
                # Count fixes
                non_null_after = result[source_col].dropna()
                invalid_after = sum(1 for v in non_null_after if not pattern.match(str(v)))
                fixed_count = invalid_before - invalid_after
            
            else:
                # Generic: try stripping whitespace and special characters
                original = result[source_col].copy()
                result[source_col] = result[source_col].astype(str).str.strip()
                
                non_null_after = result[source_col].dropna()
                invalid_after = sum(1 for v in non_null_after if not pattern.match(str(v)))
                fixed_count = invalid_before - invalid_after
            
            if fixed_count > 0:
                changes.append(f"Fixed {fixed_count} pattern violations in '{source_col}'")
            
            # Record remaining violations
            if invalid_after > 0:
                issues.append({
                    "column": source_col,
                    "issue_type": "pattern_violation",
                    "severity": "warning",
                    "description": f"{invalid_after} values don't match pattern {col_def.validation_regex}",
                    "affected_rows": invalid_after,
                    "suggested_fix": "Manual review required for remaining pattern violations"
                })
        
        return result, changes, issues
    
    def _fix_range_violations(
        self,
        df: pd.DataFrame,
        schema: Schema,
        column_mapping: Dict[str, str]
    ) -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
        """
        Fix range violations by clamping values.
        
        Args:
            df: Input DataFrame
            schema: Target schema
            column_mapping: Column name mapping
            
        Returns:
            Tuple of (fixed_df, changes, remaining_issues)
        """
        changes = []
        issues = []
        result = df.copy()
        
        for col_def in schema.columns:
            if col_def.min_value is None and col_def.max_value is None:
                continue
            
            if col_def.name not in column_mapping:
                continue
            
            source_col = column_mapping[col_def.name]
            if source_col not in result.columns:
                continue
            
            try:
                numeric = pd.to_numeric(result[source_col], errors='coerce')
                
                clamp_count = 0
                
                # Clamp to min value
                if col_def.min_value is not None:
                    below_min = (numeric < col_def.min_value).sum()
                    if below_min > 0:
                        numeric = numeric.clip(lower=col_def.min_value)
                        clamp_count += below_min
                
                # Clamp to max value
                if col_def.max_value is not None:
                    above_max = (numeric > col_def.max_value).sum()
                    if above_max > 0:
                        numeric = numeric.clip(upper=col_def.max_value)
                        clamp_count += above_max
                
                if clamp_count > 0:
                    result[source_col] = numeric
                    changes.append(
                        f"Clamped {clamp_count} values in '{source_col}' to range "
                        f"[{col_def.min_value}, {col_def.max_value}]"
                    )
                    
            except Exception as e:
                issues.append({
                    "column": source_col,
                    "issue_type": "range_violation",
                    "severity": "warning",
                    "description": f"Could not fix range violations: {str(e)}",
                    "affected_rows": "unknown",
                    "suggested_fix": "Convert to numeric before range check"
                })
        
        return result, changes, issues
    
    def _handle_null_values(
        self,
        df: pd.DataFrame,
        schema: Schema,
        column_mapping: Dict[str, str]
    ) -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
        """
        Handle null values based on schema requirements.
        
        Args:
            df: Input DataFrame
            schema: Target schema
            column_mapping: Column name mapping
            
        Returns:
            Tuple of (fixed_df, changes, remaining_issues)
        """
        changes = []
        issues = []
        result = df.copy()
        
        for col_def in schema.columns:
            if col_def.name not in column_mapping:
                continue
            
            source_col = column_mapping[col_def.name]
            if source_col not in result.columns:
                continue
            
            null_count = result[source_col].isnull().sum()
            
            if null_count == 0:
                continue
            
            # If there's a default value, fill nulls
            if col_def.default_value is not None:
                result[source_col] = result[source_col].fillna(col_def.default_value)
                changes.append(
                    f"Filled {null_count} null values in '{source_col}' with default: {col_def.default_value}"
                )
            elif col_def.required:
                # Required column with nulls - record as issue
                issues.append({
                    "column": source_col,
                    "issue_type": "null_values",
                    "severity": "critical" if col_def.required else "warning",
                    "description": f"Column '{source_col}' has {null_count} null values",
                    "affected_rows": null_count,
                    "suggested_fix": "Fill with appropriate values or remove rows"
                })
            else:
                # Optional column - just note it
                issues.append({
                    "column": source_col,
                    "issue_type": "null_values",
                    "severity": "info",
                    "description": f"Optional column '{source_col}' has {null_count} null values",
                    "affected_rows": null_count,
                    "suggested_fix": "Consider filling with appropriate values"
                })
        
        return result, changes, issues
    
    def _generate_report(
        self,
        changes_made: List[str],
        issues_remaining: List[Dict[str, Any]],
        schema: Schema
    ) -> str:
        """
        Generate a human-readable preprocessing report.
        
        Args:
            changes_made: List of changes applied
            issues_remaining: List of remaining issues
            schema: Target schema
            
        Returns:
            Report string
        """
        lines = [
            "=" * 50,
            "RULES-BASED PREPROCESSING REPORT",
            "=" * 50,
            f"Target Schema: {schema.name}",
            "",
        ]
        
        if changes_made:
            lines.append(f"Changes Applied ({len(changes_made)}):")
            lines.append("-" * 30)
            for i, change in enumerate(changes_made, 1):
                lines.append(f"  {i}. {change}")
            lines.append("")
        else:
            lines.append("No changes were needed.")
            lines.append("")
        
        if issues_remaining:
            critical = [i for i in issues_remaining if i.get("severity") == "critical"]
            warnings = [i for i in issues_remaining if i.get("severity") == "warning"]
            info = [i for i in issues_remaining if i.get("severity") == "info"]
            
            lines.append(f"Remaining Issues ({len(issues_remaining)}):")
            lines.append("-" * 30)
            
            if critical:
                lines.append(f"  Critical ({len(critical)}):")
                for issue in critical:
                    lines.append(f"    - {issue['column']}: {issue['description']}")
            
            if warnings:
                lines.append(f"  Warnings ({len(warnings)}):")
                for issue in warnings:
                    lines.append(f"    - {issue['column']}: {issue['description']}")
            
            if info:
                lines.append(f"  Info ({len(info)}):")
                for issue in info:
                    lines.append(f"    - {issue['column']}: {issue['description']}")
            
            lines.append("")
        else:
            lines.append("All issues resolved by preprocessing!")
            lines.append("")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


def get_preprocessor(schema_manager: Optional[SchemaManager] = None) -> RulesBasedPreprocessor:
    """Factory function to get a RulesBasedPreprocessor instance."""
    return RulesBasedPreprocessor(schema_manager)

