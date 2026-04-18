import pandas as pd
import numpy as np
from typing import List, Optional

class AutoCleaner:
    """Automates common data cleaning tasks for ML readiness."""
    
    def __init__(self, target_column: Optional[str] = None):
        self.target_column = target_column
        self.stats = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the standard cleaning suite."""
        df = df.copy()
        
        # 1. Deduplication
        df = self._remove_duplicates(df)
        
        # 2. Handle missing values
        df = self._impute_missing(df)
        
        # 3. Outlier Handling (optional, simple clip)
        df = self._clip_outliers(df)
        
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        df = df.drop_duplicates()
        self.stats['duplicates_removed'] = initial_len - len(df)
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple imputation strategy: median for numeric, mode for categorical."""
        for col in df.columns:
            if col == self.target_column:
                continue
                
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                else:
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
                    df[col] = df[col].fillna(fill_value)
        return df

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clips numeric values to 1st and 99th percentiles."""
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == self.target_column:
                continue
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
        return df
