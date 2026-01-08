"""
Schema Manager for BIDS

Handles output schema definitions and validation:
- IPA Standard format
- Custom user-defined schemas
- Schema validation and transformation
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np


class ColumnType(Enum):
    """Supported column data types."""
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"


class TransformType(Enum):
    """Available column transformations."""
    NONE = "none"
    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    TRIM = "trim"
    LOG2 = "log2"
    LOG10 = "log10"
    ABS = "abs"
    ROUND = "round"
    NEGATE = "negate"
    # Statistical test transformations
    FDR_BENJAMINI_HOCHBERG = "fdr_benjamini_hochberg"
    FDR_BENJAMINI_YEKUTIELI = "fdr_benjamini_yekutieli"
    BONFERRONI = "bonferroni"
    HOLM_BONFERRONI = "holm_bonferroni"


@dataclass
class ColumnDefinition:
    """Definition of an output column."""
    name: str
    type: ColumnType
    source: str = "auto"  # Column name to map from, or "auto" for automatic detection
    required: bool = True
    description: str = ""
    transform: TransformType = TransformType.NONE
    default_value: Optional[Any] = None
    validation_regex: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    # Statistical test parameters
    stat_test_source_column: Optional[str] = None  # Source column containing p-values to adjust
    stat_test_method: Optional[str] = None  # Method name (e.g., "fdr_bh", "bonferroni")
    stat_test_alpha: Optional[float] = None  # Significance level (default 0.05)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.type.value,
            "source": self.source,
            "required": self.required,
            "description": self.description,
            "transform": self.transform.value,
            "default_value": self.default_value,
            "validation_regex": self.validation_regex,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "stat_test_source_column": self.stat_test_source_column,
            "stat_test_method": self.stat_test_method,
            "stat_test_alpha": self.stat_test_alpha
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnDefinition':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type=ColumnType(data.get("type", "string")),
            source=data.get("source", "auto"),
            required=data.get("required", True),
            description=data.get("description", ""),
            transform=TransformType(data.get("transform", "none")),
            default_value=data.get("default_value"),
            validation_regex=data.get("validation_regex"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            stat_test_source_column=data.get("stat_test_source_column"),
            stat_test_method=data.get("stat_test_method"),
            stat_test_alpha=data.get("stat_test_alpha")
        )


@dataclass
class Schema:
    """Output schema definition."""
    name: str
    version: str = "1.0"
    description: str = ""
    builtin: bool = False
    columns: List[ColumnDefinition] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "builtin": self.builtin,
            "columns": [col.to_dict() for col in self.columns]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Schema':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            builtin=data.get("builtin", False),
            columns=[ColumnDefinition.from_dict(c) for c in data.get("columns", [])]
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Schema':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_column_names(self) -> List[str]:
        """Get list of output column names."""
        return [col.name for col in self.columns]
    
    def get_required_columns(self) -> List[str]:
        """Get list of required column names."""
        return [col.name for col in self.columns if col.required]
    
    def get_validation_rules(self) -> str:
        """Get a string representation of validation rules."""
        rules = []
        for col in self.columns:
            rule = f"- {col.name} ({col.type.value})"
            if col.required:
                rule += " [REQUIRED]"
            if col.validation_regex:
                rule += f" matches: {col.validation_regex}"
            if col.min_value is not None or col.max_value is not None:
                rule += f" range: [{col.min_value or '-inf'}, {col.max_value or 'inf'}]"
            rules.append(rule)
        return "\n".join(rules)


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    column_mapping: Dict[str, str]  # target_col -> source_col


class SchemaManager:
    """
    Manages output schemas for BIDS.
    
    Handles:
    - Loading/saving schemas
    - IPA standard schema
    - Custom schema creation
    - Schema validation
    - Data transformation
    """
    
    # IPA Standard Schema
    IPA_STANDARD = Schema(
        name="IPA Standard",
        version="1.0",
        description="Standard Ingenuity Pathway Analysis format",
        builtin=True,
        columns=[
            ColumnDefinition(
                name="GeneSymbol",
                type=ColumnType.STRING,
                source="auto",
                required=True,
                description="Gene symbol identifier (e.g., BRCA1, TP53)",
                validation_regex=r"^[A-Za-z][A-Za-z0-9\-]*$"
            ),
            ColumnDefinition(
                name="FoldChange",
                type=ColumnType.FLOAT,
                source="auto",
                required=True,
                description="Expression fold change value"
            ),
            ColumnDefinition(
                name="PValue",
                type=ColumnType.FLOAT,
                source="auto",
                required=True,
                description="Statistical significance p-value",
                min_value=0.0,
                max_value=1.0
            )
        ]
    )
    
    def __init__(self, schemas_dir: Optional[Union[str, Path]] = None):
        """
        Initialize SchemaManager.
        
        Args:
            schemas_dir: Directory for storing custom schemas
        """
        self.schemas_dir = Path(schemas_dir) if schemas_dir else Path("schemas")
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache of loaded schemas
        self._schemas: Dict[str, Schema] = {
            "IPA Standard": self.IPA_STANDARD
        }
        
        # Load custom schemas from directory
        self._load_schemas()
    
    def _load_schemas(self) -> None:
        """Load all schemas from the schemas directory."""
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    data = json.load(f)
                schema = Schema.from_dict(data)
                self._schemas[schema.name] = schema
            except Exception as e:
                print(f"Error loading schema {schema_file}: {e}")
    
    def get_schema(self, name: str) -> Optional[Schema]:
        """
        Get a schema by name.
        
        Args:
            name: Schema name
            
        Returns:
            Schema or None if not found
        """
        return self._schemas.get(name)
    
    def list_schemas(self) -> List[str]:
        """
        List all available schema names.
        
        Returns:
            List of schema names
        """
        return list(self._schemas.keys())
    
    def save_schema(self, schema: Schema) -> bool:
        """
        Save a schema to disk.
        
        Args:
            schema: Schema to save
            
        Returns:
            True if successful
        """
        try:
            # Generate filename from schema name
            filename = re.sub(r'[^a-zA-Z0-9_]', '_', schema.name.lower()) + ".json"
            filepath = self.schemas_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(schema.to_dict(), f, indent=2)
            
            self._schemas[schema.name] = schema
            return True
            
        except Exception as e:
            print(f"Error saving schema: {e}")
            return False
    
    def delete_schema(self, name: str) -> bool:
        """
        Delete a custom schema.
        
        Args:
            name: Schema name
            
        Returns:
            True if successful
        """
        if name not in self._schemas:
            return False
        
        schema = self._schemas[name]
        if schema.builtin:
            return False  # Can't delete built-in schemas
        
        try:
            # Find and delete the file
            filename = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower()) + ".json"
            filepath = self.schemas_dir / filename
            
            if filepath.exists():
                filepath.unlink()
            
            del self._schemas[name]
            return True
            
        except Exception:
            return False
    
    def create_schema(
        self,
        name: str,
        columns: List[Dict[str, Any]],
        description: str = ""
    ) -> Schema:
        """
        Create a new custom schema.
        
        Args:
            name: Schema name
            columns: List of column definition dicts
            description: Schema description
            
        Returns:
            Created Schema object
        """
        col_definitions = [ColumnDefinition.from_dict(c) for c in columns]
        
        schema = Schema(
            name=name,
            version="1.0",
            description=description,
            builtin=False,
            columns=col_definitions
        )
        
        return schema
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        schema: Schema
    ) -> ValidationResult:
        """
        Validate a DataFrame against a schema.
        
        Args:
            df: DataFrame to validate
            schema: Target schema
            
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        column_mapping = {}
        
        # Try to map columns
        for target_col in schema.columns:
            source_col = self._find_source_column(df, target_col)
            
            if source_col:
                column_mapping[target_col.name] = source_col
                
                # Validate column values
                col_errors, col_warnings = self._validate_column(
                    df[source_col], 
                    target_col
                )
                errors.extend(col_errors)
                warnings.extend(col_warnings)
            else:
                if target_col.required:
                    errors.append(f"Required column '{target_col.name}' not found in data")
                else:
                    warnings.append(f"Optional column '{target_col.name}' not found")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            column_mapping=column_mapping
        )
    
    def _find_source_column(
        self,
        df: pd.DataFrame,
        target_col: ColumnDefinition
    ) -> Optional[str]:
        """Find the source column for a target column definition."""
        # If source is specified (not auto), use it
        if target_col.source != "auto":
            if target_col.source in df.columns:
                return target_col.source
            return None
        
        # Try exact match first
        if target_col.name in df.columns:
            return target_col.name
        
        # Try case-insensitive match
        lower_name = target_col.name.lower()
        for col in df.columns:
            if col.lower() == lower_name:
                return col
        
        # Try pattern matching based on column type
        patterns = self._get_column_patterns(target_col)
        for col in df.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    return col
        
        return None
    
    def _get_column_patterns(self, col_def: ColumnDefinition) -> List[str]:
        """Get regex patterns for auto-detecting columns."""
        name_lower = col_def.name.lower()
        
        # Common bioinformatics patterns
        patterns_map = {
            "genesymbol": [r"gene", r"symbol", r"name", r"id"],
            "foldchange": [r"fold", r"fc", r"log2?fc", r"ratio", r"change"],
            "pvalue": [r"p[-_]?val", r"pvalue", r"significance", r"adj[-_]?p"],
        }
        
        return patterns_map.get(name_lower, [name_lower])
    
    def _validate_column(
        self,
        series: pd.Series,
        col_def: ColumnDefinition
    ) -> tuple[List[str], List[str]]:
        """Validate a single column against its definition."""
        errors = []
        warnings = []
        col_name = col_def.name
        
        # Check nulls
        null_count = series.isnull().sum()
        if null_count > 0:
            if col_def.required:
                errors.append(f"Column '{col_name}' has {null_count} null values")
            else:
                warnings.append(f"Column '{col_name}' has {null_count} null values")
        
        # Check type compatibility
        non_null = series.dropna()
        if len(non_null) > 0:
            if col_def.type == ColumnType.FLOAT:
                try:
                    pd.to_numeric(non_null, errors='raise')
                except (ValueError, TypeError):
                    errors.append(f"Column '{col_name}' has non-numeric values")
            
            elif col_def.type == ColumnType.INT:
                try:
                    values = pd.to_numeric(non_null, errors='raise')
                    if not all(values == values.astype(int)):
                        warnings.append(f"Column '{col_name}' has non-integer values")
                except (ValueError, TypeError):
                    errors.append(f"Column '{col_name}' has non-numeric values")
        
        # Check regex pattern
        if col_def.validation_regex and col_def.type == ColumnType.STRING:
            pattern = re.compile(col_def.validation_regex)
            invalid_count = sum(1 for v in non_null if not pattern.match(str(v)))
            if invalid_count > 0:
                errors.append(f"Column '{col_name}' has {invalid_count} values not matching pattern")
        
        # Check range
        if col_def.type in (ColumnType.FLOAT, ColumnType.INT):
            try:
                numeric = pd.to_numeric(non_null, errors='coerce')
                if col_def.min_value is not None:
                    below_min = (numeric < col_def.min_value).sum()
                    if below_min > 0:
                        errors.append(f"Column '{col_name}' has {below_min} values below minimum {col_def.min_value}")
                if col_def.max_value is not None:
                    above_max = (numeric > col_def.max_value).sum()
                    if above_max > 0:
                        errors.append(f"Column '{col_name}' has {above_max} values above maximum {col_def.max_value}")
            except Exception:
                pass
        
        return errors, warnings
    
    def transform_dataframe(
        self,
        df: pd.DataFrame,
        schema: Schema,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Transform a DataFrame to match a schema.
        
        Args:
            df: Input DataFrame
            schema: Target schema
            column_mapping: Optional manual column mapping
            
        Returns:
            Transformed DataFrame
        """
        if column_mapping is None:
            validation = self.validate_dataframe(df, schema)
            column_mapping = validation.column_mapping
        
        result = pd.DataFrame()
        
        for col_def in schema.columns:
            if col_def.name in column_mapping:
                source_col = column_mapping[col_def.name]
                series = df[source_col].copy()
                
                # Check if this is a statistical test transform
                if col_def.transform in (TransformType.FDR_BENJAMINI_HOCHBERG, 
                                        TransformType.FDR_BENJAMINI_YEKUTIELI,
                                        TransformType.BONFERRONI,
                                        TransformType.HOLM_BONFERRONI):
                    # Apply statistical test transformation
                    series = self._apply_statistical_test(df, col_def, column_mapping)
                else:
                    # Apply regular transformation
                    series = self._apply_transform(series, col_def.transform)
                
                # Convert type
                series = self._convert_type(series, col_def.type)
                
                result[col_def.name] = series
            else:
                # Use default value or NaN
                if col_def.default_value is not None:
                    result[col_def.name] = col_def.default_value
                else:
                    result[col_def.name] = np.nan
        
        return result
    
    def _apply_statistical_test(
        self,
        df: pd.DataFrame,
        col_def: ColumnDefinition,
        column_mapping: Dict[str, str]
    ) -> pd.Series:
        """
        Apply a statistical test transformation (FDR, Bonferroni, etc.).
        
        Args:
            df: Input DataFrame
            col_def: Column definition with statistical test parameters
            column_mapping: Column mapping dictionary
            
        Returns:
            Series with adjusted p-values
        """
        try:
            # Get source column for p-values
            source_col = col_def.stat_test_source_column
            if not source_col or source_col not in df.columns:
                # Try to find it in column mapping or by name
                if source_col in column_mapping.values():
                    # Find the key that maps to this value
                    for key, val in column_mapping.items():
                        if val == source_col:
                            source_col = val
                            break
                else:
                    # Try to find column by name pattern
                    pvalue_patterns = [r"p[-_]?val", r"pvalue", r"significance"]
                    import re
                    for col in df.columns:
                        col_lower = col.lower()
                        for pattern in pvalue_patterns:
                            if re.search(pattern, col_lower):
                                source_col = col
                                break
                        if source_col and source_col in df.columns:
                            break
                
                if not source_col or source_col not in df.columns:
                    # Return NaN series if source column not found
                    return pd.Series([np.nan] * len(df), index=df.index)
            
            # Get p-values from source column
            pvalues = pd.to_numeric(df[source_col], errors='coerce')
            
            # Remove NaN values for the test
            valid_mask = pvalues.notna()
            if not valid_mask.any():
                return pd.Series([np.nan] * len(df), index=df.index)
            
            # Validate p-values are in [0, 1] range
            valid_pvalues = pvalues[valid_mask]
            if (valid_pvalues < 0).any() or (valid_pvalues > 1).any():
                # Clamp to [0, 1] range
                valid_pvalues = valid_pvalues.clip(0.0, 1.0)
            
            # Determine method based on transform type
            method_map = {
                TransformType.FDR_BENJAMINI_HOCHBERG: 'fdr_bh',
                TransformType.FDR_BENJAMINI_YEKUTIELI: 'fdr_by',
                TransformType.BONFERRONI: 'bonferroni',
                TransformType.HOLM_BONFERRONI: 'holm'
            }
            method = method_map.get(col_def.transform, 'fdr_bh')
            
            # Get alpha value
            alpha = col_def.stat_test_alpha if col_def.stat_test_alpha is not None else 0.05
            
            # Apply multiple testing correction
            try:
                from statsmodels.stats.multitest import multipletests
                adjusted_pvalues, _, _, _ = multipletests(
                    valid_pvalues.values,
                    alpha=alpha,
                    method=method
                )
            except ImportError:
                # Fallback if statsmodels not available
                raise ImportError("statsmodels is required for statistical test transformations. Install with: pip install statsmodels")
            
            # Create result series with same index as original
            result = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
            result[valid_mask] = adjusted_pvalues
            
            return result
            
        except Exception as e:
            # Return NaN series on error
            import warnings
            warnings.warn(f"Statistical test transformation failed for {col_def.name}: {str(e)}")
            return pd.Series([np.nan] * len(df), index=df.index)
    
    def _apply_transform(
        self,
        series: pd.Series,
        transform: TransformType
    ) -> pd.Series:
        """Apply a transformation to a series."""
        if transform == TransformType.NONE:
            return series
        elif transform == TransformType.UPPERCASE:
            return series.astype(str).str.upper()
        elif transform == TransformType.LOWERCASE:
            return series.astype(str).str.lower()
        elif transform == TransformType.TRIM:
            return series.astype(str).str.strip()
        elif transform == TransformType.LOG2:
            return np.log2(pd.to_numeric(series, errors='coerce'))
        elif transform == TransformType.LOG10:
            return np.log10(pd.to_numeric(series, errors='coerce'))
        elif transform == TransformType.ABS:
            return pd.to_numeric(series, errors='coerce').abs()
        elif transform == TransformType.ROUND:
            return pd.to_numeric(series, errors='coerce').round()
        elif transform == TransformType.NEGATE:
            return -pd.to_numeric(series, errors='coerce')
        else:
            return series
    
    def _convert_type(
        self,
        series: pd.Series,
        target_type: ColumnType
    ) -> pd.Series:
        """Convert series to target type."""
        try:
            if target_type == ColumnType.STRING:
                return series.astype(str)
            elif target_type == ColumnType.INT:
                return pd.to_numeric(series, errors='coerce').astype('Int64')
            elif target_type == ColumnType.FLOAT:
                return pd.to_numeric(series, errors='coerce')
            elif target_type == ColumnType.BOOL:
                return series.astype(bool)
            elif target_type in (ColumnType.DATE, ColumnType.DATETIME):
                return pd.to_datetime(series, errors='coerce')
            else:
                return series
        except Exception:
            return series
    
    def attempt_auto_fix(
        self,
        df: pd.DataFrame,
        schema: Schema,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> tuple[pd.DataFrame, List[str], List[str]]:
        """
        Attempt automatic fixes for common schema violations.
        
        This is used by the rules-based preprocessor to fix simple issues
        without needing the LLM.
        
        Args:
            df: DataFrame to fix
            schema: Target schema
            column_mapping: Optional column mapping
            
        Returns:
            Tuple of (fixed_df, changes_made, remaining_issues)
        """
        if column_mapping is None:
            validation = self.validate_dataframe(df, schema)
            column_mapping = validation.column_mapping
        
        result = df.copy()
        changes = []
        remaining = []
        
        for col_def in schema.columns:
            if col_def.name not in column_mapping:
                if col_def.required:
                    remaining.append(f"Required column '{col_def.name}' not found")
                continue
            
            source_col = column_mapping[col_def.name]
            if source_col not in result.columns:
                continue
            
            # Try to fix type violations
            fix_result = self._fix_type_violations(result[source_col], col_def)
            if fix_result["changed"]:
                result[source_col] = fix_result["series"]
                changes.append(f"Fixed type for '{source_col}': {fix_result['description']}")
            if fix_result["remaining"]:
                remaining.append(fix_result["remaining"])
            
            # Try to fix pattern violations
            fix_result = self._fix_pattern_violations(result[source_col], col_def)
            if fix_result["changed"]:
                result[source_col] = fix_result["series"]
                changes.append(f"Fixed pattern for '{source_col}': {fix_result['description']}")
            if fix_result["remaining"]:
                remaining.append(fix_result["remaining"])
            
            # Try to fix range violations
            fix_result = self._fix_range_violations(result[source_col], col_def)
            if fix_result["changed"]:
                result[source_col] = fix_result["series"]
                changes.append(f"Fixed range for '{source_col}': {fix_result['description']}")
            if fix_result["remaining"]:
                remaining.append(fix_result["remaining"])
            
            # Handle null values
            fix_result = self._fix_null_values(result[source_col], col_def)
            if fix_result["changed"]:
                result[source_col] = fix_result["series"]
                changes.append(f"Fixed nulls for '{source_col}': {fix_result['description']}")
            if fix_result["remaining"]:
                remaining.append(fix_result["remaining"])
        
        return result, changes, remaining
    
    def _fix_type_violations(
        self,
        series: pd.Series,
        col_def: ColumnDefinition
    ) -> Dict[str, Any]:
        """Attempt to fix type violations."""
        result = {"changed": False, "series": series, "description": "", "remaining": None}
        
        try:
            if col_def.type == ColumnType.FLOAT:
                # Try to convert to numeric
                original = series.copy()
                converted = pd.to_numeric(series, errors='coerce')
                coerced_count = original.notna().sum() - converted.notna().sum()
                if coerced_count > 0:
                    result["series"] = converted
                    result["changed"] = True
                    result["description"] = f"Coerced {coerced_count} values to numeric"
                    if coerced_count > len(series) * 0.1:
                        result["remaining"] = f"'{series.name}' has {coerced_count} non-numeric values"
                        
            elif col_def.type == ColumnType.INT:
                original = series.copy()
                converted = pd.to_numeric(series, errors='coerce').round().astype('Int64')
                result["series"] = converted
                result["changed"] = True
                result["description"] = "Converted to integer"
                
            elif col_def.type == ColumnType.BOOL:
                bool_map = {
                    'true': True, 'false': False,
                    'yes': True, 'no': False,
                    '1': True, '0': False,
                    't': True, 'f': False,
                }
                converted = series.astype(str).str.lower().map(bool_map)
                if converted.notna().any():
                    result["series"] = converted
                    result["changed"] = True
                    result["description"] = "Converted to boolean"
                    
        except Exception:
            pass
        
        return result
    
    def _fix_pattern_violations(
        self,
        series: pd.Series,
        col_def: ColumnDefinition
    ) -> Dict[str, Any]:
        """Attempt to fix pattern violations."""
        result = {"changed": False, "series": series, "description": "", "remaining": None}
        
        if not col_def.validation_regex or col_def.type != ColumnType.STRING:
            return result
        
        pattern = re.compile(col_def.validation_regex)
        
        try:
            original = series.copy()
            fixed = series.astype(str).str.strip()
            
            # For gene symbols: remove invalid leading/trailing characters
            if 'gene' in col_def.name.lower() or 'symbol' in col_def.name.lower():
                fixed = fixed.str.replace(r'^[^A-Za-z]+', '', regex=True)
                fixed = fixed.str.replace(r'[^A-Za-z0-9\-]+$', '', regex=True)
            
            # Count fixes
            non_null = fixed.dropna()
            invalid_after = sum(1 for v in non_null if not pattern.match(str(v)))
            original_invalid = sum(1 for v in original.dropna() if not pattern.match(str(v)))
            
            if invalid_after < original_invalid:
                result["series"] = fixed
                result["changed"] = True
                result["description"] = f"Fixed {original_invalid - invalid_after} pattern violations"
            
            if invalid_after > 0:
                result["remaining"] = f"'{col_def.name}' still has {invalid_after} pattern violations"
                
        except Exception:
            pass
        
        return result
    
    def _fix_range_violations(
        self,
        series: pd.Series,
        col_def: ColumnDefinition
    ) -> Dict[str, Any]:
        """Attempt to fix range violations by clamping."""
        result = {"changed": False, "series": series, "description": "", "remaining": None}
        
        if col_def.min_value is None and col_def.max_value is None:
            return result
        
        if col_def.type not in (ColumnType.FLOAT, ColumnType.INT):
            return result
        
        try:
            numeric = pd.to_numeric(series, errors='coerce')
            clamp_count = 0
            
            if col_def.min_value is not None:
                below = (numeric < col_def.min_value).sum()
                if below > 0:
                    numeric = numeric.clip(lower=col_def.min_value)
                    clamp_count += below
            
            if col_def.max_value is not None:
                above = (numeric > col_def.max_value).sum()
                if above > 0:
                    numeric = numeric.clip(upper=col_def.max_value)
                    clamp_count += above
            
            if clamp_count > 0:
                result["series"] = numeric
                result["changed"] = True
                result["description"] = f"Clamped {clamp_count} values to [{col_def.min_value}, {col_def.max_value}]"
                
        except Exception:
            pass
        
        return result
    
    def _fix_null_values(
        self,
        series: pd.Series,
        col_def: ColumnDefinition
    ) -> Dict[str, Any]:
        """Attempt to fix null values."""
        result = {"changed": False, "series": series, "description": "", "remaining": None}
        
        null_count = series.isnull().sum()
        if null_count == 0:
            return result
        
        # If there's a default value, fill nulls
        if col_def.default_value is not None:
            result["series"] = series.fillna(col_def.default_value)
            result["changed"] = True
            result["description"] = f"Filled {null_count} nulls with default: {col_def.default_value}"
        elif col_def.required:
            result["remaining"] = f"'{col_def.name}' has {null_count} null values (required)"
        
        return result


def get_schema_manager(schemas_dir: Optional[str] = None) -> SchemaManager:
    """Factory function to get a SchemaManager instance."""
    return SchemaManager(schemas_dir)
