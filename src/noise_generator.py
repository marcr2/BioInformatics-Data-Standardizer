"""
Noise Generator for BIDS (Pre-Made Standard Error Generator)

Generates synthetic noise in DataFrames for testing and training
the error-matching RAG system.

Noise Types:
- Typos: Character-level corruptions
- Semantic: Value-level corruptions (similar but wrong values)
- Structural: Schema-level corruptions (type changes, missing columns)
"""

import random
import string
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd


class NoiseType(Enum):
    """Types of noise that can be injected."""
    TYPO = "typo"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"


class DistributionMode(Enum):
    """How noise is distributed across columns."""
    UNIFORM = "uniform"  # Equal probability for all columns
    WEIGHTED = "weighted"  # Custom weights per column
    TARGETED = "targeted"  # Only specific columns


@dataclass
class NoiseConfig:
    """Configuration for noise generation."""
    noise_ratio: float = 0.1  # 0.0-1.0, percentage of cells to corrupt
    distribution_mode: DistributionMode = DistributionMode.UNIFORM
    column_weights: Dict[str, float] = field(default_factory=dict)
    target_columns: List[str] = field(default_factory=list)
    noise_types: List[NoiseType] = field(default_factory=lambda: [
        NoiseType.TYPO, 
        NoiseType.SEMANTIC, 
        NoiseType.STRUCTURAL
    ])
    seed: Optional[int] = None


@dataclass
class NoiseReport:
    """Report of noise applied to a DataFrame."""
    original_shape: tuple
    cells_modified: int
    modifications: List[Dict[str, Any]]
    noise_ratio_actual: float
    columns_affected: List[str]


class NoiseGenerator:
    """
    Generates synthetic noise in DataFrames.
    
    Supports three types of noise:
    1. Typo: Character swaps, deletions, insertions
    2. Semantic: Replace with similar but wrong values
    3. Structural: Type changes, NaN injection, column drops
    """
    
    # Common typos - adjacent keys on QWERTY keyboard
    ADJACENT_KEYS = {
        'a': ['s', 'q', 'w', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 's', 'd', 'r'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'j', 'k', 'o'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'k', 'l', 'p'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 'd', 'f', 't'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'f', 'g', 'y'],
        'u': ['y', 'h', 'j', 'i'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'a', 's', 'e'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'g', 'h', 'u'],
        'z': ['a', 's', 'x'],
    }
    
    # Common gene symbol confusions (for bioinformatics context)
    GENE_CONFUSIONS = {
        'BRCA1': ['BRCA2', 'BRAC1', 'BRCA'],
        'BRCA2': ['BRCA1', 'BRAC2', 'BRCA'],
        'TP53': ['TP52', 'TP54', 'P53', 'TRP53'],
        'EGFR': ['EFGR', 'EGF', 'EGRF'],
        'KRAS': ['KARS', 'HRAS', 'NRAS'],
        'MYC': ['MYCC', 'C-MYC', 'MIC'],
        'AKT1': ['AKT2', 'AKT3', 'ATK1'],
    }
    
    def __init__(self, config: Optional[NoiseConfig] = None):
        """
        Initialize NoiseGenerator.
        
        Args:
            config: NoiseConfig object (uses defaults if not provided)
        """
        self.config = config or NoiseConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
    
    def inject_noise(self, df: pd.DataFrame) -> tuple[pd.DataFrame, NoiseReport]:
        """
        Inject noise into a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (noisy DataFrame, NoiseReport)
        """
        df_noisy = df.copy()
        modifications: List[Dict[str, Any]] = []
        
        # Calculate number of cells to modify
        total_cells = df.size
        cells_to_modify = int(total_cells * self.config.noise_ratio)
        
        # Get eligible columns based on distribution mode
        eligible_columns = self._get_eligible_columns(df)
        
        if not eligible_columns:
            return df_noisy, NoiseReport(
                original_shape=df.shape,
                cells_modified=0,
                modifications=[],
                noise_ratio_actual=0.0,
                columns_affected=[]
            )
        
        # Generate random cell positions
        positions = self._generate_positions(df, eligible_columns, cells_to_modify)
        
        # Apply noise to each position
        for row_idx, col_name in positions:
            original_value = df_noisy.at[row_idx, col_name]
            noise_type = random.choice(self.config.noise_types)
            
            new_value, modification = self._apply_noise(
                original_value, 
                col_name, 
                noise_type,
                df_noisy[col_name].dtype
            )
            
            df_noisy.at[row_idx, col_name] = new_value
            modifications.append({
                "row": row_idx,
                "column": col_name,
                "original": original_value,
                "modified": new_value,
                "noise_type": noise_type.value,
                "modification": modification
            })
        
        return df_noisy, NoiseReport(
            original_shape=df.shape,
            cells_modified=len(modifications),
            modifications=modifications,
            noise_ratio_actual=len(modifications) / total_cells if total_cells > 0 else 0,
            columns_affected=list(set(m["column"] for m in modifications))
        )
    
    def _get_eligible_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns eligible for noise injection based on config."""
        if self.config.distribution_mode == DistributionMode.TARGETED:
            return [c for c in self.config.target_columns if c in df.columns]
        else:
            return list(df.columns)
    
    def _generate_positions(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        count: int
    ) -> List[tuple[int, str]]:
        """Generate random positions for noise injection."""
        positions = []
        
        if self.config.distribution_mode == DistributionMode.WEIGHTED:
            # Use weights for column selection
            weights = [self.config.column_weights.get(c, 1.0) for c in columns]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            for _ in range(count):
                col = np.random.choice(columns, p=weights)
                row = random.randint(0, len(df) - 1)
                positions.append((row, col))
        else:
            # Uniform distribution
            for _ in range(count):
                col = random.choice(columns)
                row = random.randint(0, len(df) - 1)
                positions.append((row, col))
        
        return positions
    
    def _apply_noise(
        self, 
        value: Any, 
        column: str, 
        noise_type: NoiseType,
        dtype: np.dtype
    ) -> tuple[Any, str]:
        """
        Apply noise to a single value.
        
        Returns:
            Tuple of (new_value, modification_description)
        """
        if noise_type == NoiseType.TYPO:
            return self._apply_typo(value, dtype)
        elif noise_type == NoiseType.SEMANTIC:
            return self._apply_semantic(value, column, dtype)
        else:  # STRUCTURAL
            return self._apply_structural(value, dtype)
    
    def _apply_typo(self, value: Any, dtype: np.dtype) -> tuple[Any, str]:
        """Apply typo-style noise."""
        if pd.isna(value):
            return value, "skipped_nan"
        
        str_value = str(value)
        if len(str_value) < 2:
            return value, "too_short"
        
        # Choose typo type
        typo_type = random.choice(['swap', 'delete', 'insert', 'adjacent'])
        
        if typo_type == 'swap' and len(str_value) >= 2:
            # Swap two adjacent characters
            idx = random.randint(0, len(str_value) - 2)
            chars = list(str_value)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            new_value = ''.join(chars)
            desc = f"swapped chars at {idx}"
            
        elif typo_type == 'delete':
            # Delete a character
            idx = random.randint(0, len(str_value) - 1)
            new_value = str_value[:idx] + str_value[idx + 1:]
            desc = f"deleted char at {idx}"
            
        elif typo_type == 'insert':
            # Insert a random character
            idx = random.randint(0, len(str_value))
            char = random.choice(string.ascii_letters)
            new_value = str_value[:idx] + char + str_value[idx:]
            desc = f"inserted '{char}' at {idx}"
            
        else:  # adjacent
            # Replace with adjacent key
            idx = random.randint(0, len(str_value) - 1)
            char = str_value[idx].lower()
            if char in self.ADJACENT_KEYS:
                replacement = random.choice(self.ADJACENT_KEYS[char])
                if str_value[idx].isupper():
                    replacement = replacement.upper()
                new_value = str_value[:idx] + replacement + str_value[idx + 1:]
                desc = f"adjacent key '{char}' -> '{replacement}'"
            else:
                new_value = str_value
                desc = "no_adjacent_found"
        
        # Try to maintain original type for numeric
        if np.issubdtype(dtype, np.number):
            try:
                new_value = type(value)(new_value)
            except (ValueError, TypeError):
                pass  # Keep as string
        
        return new_value, desc
    
    def _apply_semantic(
        self, 
        value: Any, 
        column: str, 
        dtype: np.dtype
    ) -> tuple[Any, str]:
        """Apply semantic-style noise (similar but wrong values)."""
        if pd.isna(value):
            return value, "skipped_nan"
        
        str_value = str(value).upper()
        
        # Check for gene symbol confusions
        if str_value in self.GENE_CONFUSIONS:
            new_value = random.choice(self.GENE_CONFUSIONS[str_value])
            return new_value, f"gene_confusion: {str_value} -> {new_value}"
        
        # For numeric values, add small perturbation
        if np.issubdtype(dtype, np.number) and not pd.isna(value):
            if np.issubdtype(dtype, np.floating):
                # Small float perturbation
                perturbation = random.uniform(-0.1, 0.1) * abs(value) if value != 0 else random.uniform(-0.1, 0.1)
                new_value = value + perturbation
                return new_value, f"float_drift: {value} -> {new_value:.4f}"
            else:
                # Integer off-by-one
                offset = random.choice([-2, -1, 1, 2])
                new_value = value + offset
                return new_value, f"int_drift: {value} -> {new_value}"
        
        # For strings, apply case changes or truncation
        if isinstance(value, str):
            semantic_type = random.choice(['case', 'truncate', 'prefix'])
            if semantic_type == 'case':
                new_value = value.swapcase()
                return new_value, "case_swap"
            elif semantic_type == 'truncate' and len(value) > 3:
                new_value = value[:-random.randint(1, 2)]
                return new_value, "truncated"
            else:
                prefix = random.choice(['_', '-', ' ', 'X'])
                new_value = prefix + value
                return new_value, f"added_prefix: '{prefix}'"
        
        return value, "no_semantic_change"
    
    def _apply_structural(
        self, 
        value: Any, 
        dtype: np.dtype
    ) -> tuple[Any, str]:
        """Apply structural-style noise (type changes, NaN injection)."""
        structural_type = random.choice(['nan', 'type_change', 'empty', 'placeholder'])
        
        if structural_type == 'nan':
            return np.nan, "injected_nan"
        
        elif structural_type == 'type_change':
            if np.issubdtype(dtype, np.number):
                # Convert number to string representation
                return f"VALUE_{value}", "num_to_str"
            else:
                # Try to convert string to number or vice versa
                try:
                    return float(str(value)[:5].replace(',', '.')), "str_to_num"
                except (ValueError, TypeError):
                    return 99999, "forced_num"
        
        elif structural_type == 'empty':
            return '', "empty_string"
        
        else:  # placeholder
            placeholders = ['N/A', 'NULL', '-', '#N/A', '???', 'MISSING', 'NA']
            placeholder = random.choice(placeholders)
            return placeholder, f"placeholder: {placeholder}"
    
    def create_noisy_dataset(
        self, 
        df: pd.DataFrame, 
        variations: int = 5
    ) -> List[tuple[pd.DataFrame, NoiseReport]]:
        """
        Create multiple noisy variations of a DataFrame.
        
        Args:
            df: Input DataFrame
            variations: Number of noisy versions to create
            
        Returns:
            List of (noisy_df, report) tuples
        """
        results = []
        for i in range(variations):
            # Vary the noise ratio slightly
            original_ratio = self.config.noise_ratio
            self.config.noise_ratio = original_ratio * random.uniform(0.5, 1.5)
            self.config.noise_ratio = min(1.0, max(0.01, self.config.noise_ratio))
            
            noisy_df, report = self.inject_noise(df)
            results.append((noisy_df, report))
            
            self.config.noise_ratio = original_ratio
        
        return results


def generate_test_noise(
    df: pd.DataFrame,
    noise_ratio: float = 0.1,
    noise_types: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> tuple[pd.DataFrame, NoiseReport]:
    """
    Convenience function to generate noisy data.
    
    Args:
        df: Input DataFrame
        noise_ratio: Fraction of cells to corrupt
        noise_types: List of noise type names ('typo', 'semantic', 'structural')
        target_columns: Specific columns to target (None = all)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (noisy DataFrame, NoiseReport)
    """
    config = NoiseConfig(
        noise_ratio=noise_ratio,
        distribution_mode=DistributionMode.TARGETED if target_columns else DistributionMode.UNIFORM,
        target_columns=target_columns or [],
        noise_types=[NoiseType(nt) for nt in (noise_types or ['typo', 'semantic', 'structural'])],
        seed=seed
    )
    
    generator = NoiseGenerator(config)
    return generator.inject_noise(df)
