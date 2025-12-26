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
            "max_value": self.max_value
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
            max_value=data.get("max_value")
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
                
                # Apply transformation
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


def get_schema_manager(schemas_dir: Optional[str] = None) -> SchemaManager:
    """Factory function to get a SchemaManager instance."""
    return SchemaManager(schemas_dir)
