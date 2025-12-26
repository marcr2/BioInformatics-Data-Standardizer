"""
Noise Vectorizer for BIDS

Creates structural fingerprints of DataFrames for similarity matching.
Embeds the STRUCTURE of data (not raw content) using metadata tokens.

Strategy:
- Convert columns into descriptive tokens
- Use TF-IDF to create vectors
- Enable similarity search for matching error patterns
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DataFingerprint:
    """Structural fingerprint of a DataFrame."""
    tokens: List[str]
    token_string: str
    vector: Optional[np.ndarray]
    metadata: Dict[str, Any]


class NoiseVectorizer:
    """
    Creates structural fingerprints of DataFrames using TF-IDF.
    
    Instead of embedding raw content, we embed metadata tokens that
    describe the structure of each column:
    - Data type
    - Null pattern
    - Cardinality
    - Value range
    - Format patterns
    """
    
    def __init__(
        self,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize the vectorizer.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            token_pattern=r'[A-Za-z0-9_]+',
        )
        self._is_fitted = False
        self._fingerprints: List[str] = []
    
    def create_fingerprint(self, df: pd.DataFrame) -> DataFingerprint:
        """
        Create a structural fingerprint of a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFingerprint object
        """
        tokens = []
        column_metadata = {}
        
        # Generate tokens for each column
        for idx, col in enumerate(df.columns):
            col_tokens, col_meta = self._analyze_column(df[col], idx)
            tokens.extend(col_tokens)
            column_metadata[col] = col_meta
        
        # Add DataFrame-level tokens
        df_tokens = self._analyze_dataframe(df)
        tokens.extend(df_tokens)
        
        token_string = ' '.join(tokens)
        
        return DataFingerprint(
            tokens=tokens,
            token_string=token_string,
            vector=None,  # Will be set by vectorize()
            metadata={
                'shape': df.shape,
                'columns': list(df.columns),
                'column_details': column_metadata
            }
        )
    
    def _analyze_column(
        self, 
        series: pd.Series, 
        index: int
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Analyze a column and generate descriptive tokens.
        
        Args:
            series: Column data
            index: Column index
            
        Returns:
            Tuple of (tokens list, metadata dict)
        """
        tokens = []
        metadata = {}
        
        # Basic info
        col_name = str(series.name) if series.name else f"col_{index}"
        dtype = str(series.dtype)
        
        # Token: Column index and type
        dtype_token = self._dtype_to_token(dtype)
        tokens.append(f"COL_{index}_{dtype_token}")
        metadata['dtype'] = dtype
        metadata['dtype_token'] = dtype_token
        
        # Token: Null pattern
        null_ratio = series.isnull().mean()
        if null_ratio == 0:
            null_token = "NO_NULLS"
        elif null_ratio < 0.1:
            null_token = "FEW_NULLS"
        elif null_ratio < 0.5:
            null_token = "SOME_NULLS"
        else:
            null_token = "MANY_NULLS"
        tokens.append(f"COL_{index}_{null_token}")
        metadata['null_ratio'] = null_ratio
        
        # Token: Cardinality
        non_null = series.dropna()
        if len(non_null) > 0:
            unique_ratio = non_null.nunique() / len(non_null)
            if unique_ratio < 0.01:
                card_token = "CONSTANT"
            elif unique_ratio < 0.1:
                card_token = "LOW_CARD"
            elif unique_ratio < 0.5:
                card_token = "MED_CARD"
            elif unique_ratio < 0.95:
                card_token = "HIGH_CARD"
            else:
                card_token = "UNIQUE"
            tokens.append(f"COL_{index}_{card_token}")
            metadata['cardinality'] = card_token
        
        # Type-specific tokens
        if pd.api.types.is_numeric_dtype(series):
            num_tokens = self._analyze_numeric(non_null, index)
            tokens.extend(num_tokens)
            metadata['numeric_analysis'] = True
        elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            str_tokens = self._analyze_string(non_null, index)
            tokens.extend(str_tokens)
            metadata['string_analysis'] = True
        
        # Column name pattern tokens
        name_tokens = self._analyze_column_name(col_name, index)
        tokens.extend(name_tokens)
        
        return tokens, metadata
    
    def _dtype_to_token(self, dtype: str) -> str:
        """Convert pandas dtype to token."""
        dtype = dtype.lower()
        if 'int' in dtype:
            return 'INT'
        elif 'float' in dtype:
            return 'FLOAT'
        elif 'bool' in dtype:
            return 'BOOL'
        elif 'datetime' in dtype:
            return 'DATETIME'
        elif 'timedelta' in dtype:
            return 'TIMEDELTA'
        elif 'category' in dtype:
            return 'CATEGORY'
        elif 'object' in dtype or 'str' in dtype:
            return 'STRING'
        else:
            return 'UNKNOWN'
    
    def _analyze_numeric(self, series: pd.Series, index: int) -> List[str]:
        """Generate tokens for numeric columns."""
        tokens = []
        
        if len(series) == 0:
            return tokens
        
        # Range tokens
        min_val, max_val = series.min(), series.max()
        
        if min_val >= 0 and max_val <= 1:
            tokens.append(f"COL_{index}_RANGE_0_1")
        elif min_val >= 0:
            tokens.append(f"COL_{index}_POSITIVE")
        elif max_val <= 0:
            tokens.append(f"COL_{index}_NEGATIVE")
        else:
            tokens.append(f"COL_{index}_MIXED_SIGN")
        
        # Scale tokens
        if max_val - min_val > 1000000:
            tokens.append(f"COL_{index}_LARGE_RANGE")
        elif max_val - min_val < 1:
            tokens.append(f"COL_{index}_SMALL_RANGE")
        
        # Distribution tokens
        mean = series.mean()
        std = series.std()
        if std > 0:
            skew = series.skew()
            if abs(skew) < 0.5:
                tokens.append(f"COL_{index}_SYMMETRIC")
            elif skew > 0:
                tokens.append(f"COL_{index}_RIGHT_SKEW")
            else:
                tokens.append(f"COL_{index}_LEFT_SKEW")
        
        # P-value pattern (common in bioinformatics)
        if min_val >= 0 and max_val <= 1:
            small_values = (series < 0.05).mean()
            if small_values > 0.1:
                tokens.append(f"COL_{index}_PVALUE_PATTERN")
        
        return tokens
    
    def _analyze_string(self, series: pd.Series, index: int) -> List[str]:
        """Generate tokens for string columns."""
        tokens = []
        
        if len(series) == 0:
            return tokens
        
        # Convert to string
        str_series = series.astype(str)
        
        # Length patterns
        lengths = str_series.str.len()
        avg_len = lengths.mean()
        
        if avg_len < 5:
            tokens.append(f"COL_{index}_SHORT_STR")
        elif avg_len < 20:
            tokens.append(f"COL_{index}_MED_STR")
        else:
            tokens.append(f"COL_{index}_LONG_STR")
        
        # Case patterns
        sample = str_series.head(100)
        upper_ratio = sample.str.isupper().mean()
        if upper_ratio > 0.8:
            tokens.append(f"COL_{index}_UPPERCASE")
        elif upper_ratio < 0.2:
            tokens.append(f"COL_{index}_LOWERCASE")
        else:
            tokens.append(f"COL_{index}_MIXEDCASE")
        
        # Content patterns
        has_numbers = sample.str.contains(r'\d', regex=True, na=False).mean()
        if has_numbers > 0.5:
            tokens.append(f"COL_{index}_HAS_NUMBERS")
        
        has_special = sample.str.contains(r'[^A-Za-z0-9\s]', regex=True, na=False).mean()
        if has_special > 0.3:
            tokens.append(f"COL_{index}_HAS_SPECIAL")
        
        # Gene symbol pattern (common in bioinformatics)
        gene_pattern = sample.str.match(r'^[A-Z][A-Z0-9]{1,10}$', na=False).mean()
        if gene_pattern > 0.5:
            tokens.append(f"COL_{index}_GENE_PATTERN")
        
        return tokens
    
    def _analyze_column_name(self, name: str, index: int) -> List[str]:
        """Generate tokens based on column name patterns."""
        tokens = []
        name_lower = name.lower()
        
        # Common bioinformatics column patterns
        patterns = {
            'gene': ['gene', 'symbol', 'name', 'id'],
            'pvalue': ['pval', 'p_val', 'p-val', 'pvalue', 'p_value', 'significance'],
            'foldchange': ['fold', 'fc', 'log2fc', 'logfc', 'ratio', 'change'],
            'expression': ['expr', 'expression', 'level', 'count', 'fpkm', 'tpm', 'rpkm'],
            'sample': ['sample', 'patient', 'subject', 'case'],
        }
        
        for pattern_name, keywords in patterns.items():
            for kw in keywords:
                if kw in name_lower:
                    tokens.append(f"COL_{index}_NAME_{pattern_name.upper()}")
                    break
        
        return tokens
    
    def _analyze_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Generate DataFrame-level tokens."""
        tokens = []
        
        # Shape tokens
        rows, cols = df.shape
        
        if rows < 100:
            tokens.append("DF_SMALL_ROWS")
        elif rows < 10000:
            tokens.append("DF_MED_ROWS")
        else:
            tokens.append("DF_LARGE_ROWS")
        
        if cols < 5:
            tokens.append("DF_FEW_COLS")
        elif cols < 20:
            tokens.append("DF_MED_COLS")
        else:
            tokens.append("DF_MANY_COLS")
        
        # Type distribution
        dtypes = df.dtypes.astype(str)
        numeric_ratio = sum(1 for d in dtypes if 'int' in d or 'float' in d) / len(dtypes)
        
        if numeric_ratio > 0.8:
            tokens.append("DF_MOSTLY_NUMERIC")
        elif numeric_ratio < 0.2:
            tokens.append("DF_MOSTLY_STRING")
        else:
            tokens.append("DF_MIXED_TYPES")
        
        # Overall null pattern
        total_nulls = df.isnull().sum().sum() / df.size
        if total_nulls < 0.01:
            tokens.append("DF_CLEAN")
        elif total_nulls < 0.1:
            tokens.append("DF_SOME_MISSING")
        else:
            tokens.append("DF_MANY_MISSING")
        
        return tokens
    
    def fit(self, fingerprints: List[DataFingerprint]) -> 'NoiseVectorizer':
        """
        Fit the TF-IDF vectorizer on a list of fingerprints.
        
        Args:
            fingerprints: List of DataFingerprint objects
            
        Returns:
            self
        """
        token_strings = [fp.token_string for fp in fingerprints]
        self.vectorizer.fit(token_strings)
        self._is_fitted = True
        self._fingerprints = token_strings
        return self
    
    def fit_transform(
        self, 
        fingerprints: List[DataFingerprint]
    ) -> np.ndarray:
        """
        Fit and transform fingerprints to vectors.
        
        Args:
            fingerprints: List of DataFingerprint objects
            
        Returns:
            Array of TF-IDF vectors
        """
        token_strings = [fp.token_string for fp in fingerprints]
        vectors = self.vectorizer.fit_transform(token_strings).toarray()
        self._is_fitted = True
        self._fingerprints = token_strings
        
        # Update fingerprints with vectors
        for fp, vec in zip(fingerprints, vectors):
            fp.vector = vec
        
        return vectors
    
    def transform(self, fingerprint: DataFingerprint) -> np.ndarray:
        """
        Transform a single fingerprint to a vector.
        
        Args:
            fingerprint: DataFingerprint object
            
        Returns:
            TF-IDF vector
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        vector = self.vectorizer.transform([fingerprint.token_string]).toarray()[0]
        fingerprint.vector = vector
        return vector
    
    def vectorize_dataframe(self, df: pd.DataFrame) -> DataFingerprint:
        """
        Create and vectorize a fingerprint for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFingerprint with vector
        """
        fingerprint = self.create_fingerprint(df)
        
        if self._is_fitted:
            self.transform(fingerprint)
        
        return fingerprint
    
    def find_similar(
        self, 
        query_fingerprint: DataFingerprint,
        corpus_fingerprints: List[DataFingerprint],
        top_k: int = 5
    ) -> List[Tuple[int, float, DataFingerprint]]:
        """
        Find similar fingerprints in a corpus.
        
        Args:
            query_fingerprint: Query fingerprint
            corpus_fingerprints: List of fingerprints to search
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score, fingerprint) tuples
        """
        if query_fingerprint.vector is None:
            raise ValueError("Query fingerprint not vectorized")
        
        # Ensure all corpus fingerprints have vectors
        corpus_vectors = []
        for fp in corpus_fingerprints:
            if fp.vector is None:
                raise ValueError("Corpus fingerprint not vectorized")
            corpus_vectors.append(fp.vector)
        
        corpus_matrix = np.array(corpus_vectors)
        query_vector = query_fingerprint.vector.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, corpus_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                corpus_fingerprints[idx]
            ))
        
        return results
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names from the fitted vectorizer."""
        if not self._is_fitted:
            return []
        return list(self.vectorizer.get_feature_names_out())


def fingerprint_dataframe(df: pd.DataFrame) -> DataFingerprint:
    """
    Convenience function to create a fingerprint.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFingerprint object (not vectorized)
    """
    vectorizer = NoiseVectorizer()
    return vectorizer.create_fingerprint(df)
