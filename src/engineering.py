import pandas as pd
import numpy as np
from typing import List

class FeatureOptimizer:
    """Automates feature generation for ML workflows."""
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Applies automated feature engineering transformations."""
        df = df.copy()
        
        # 1. Date Features
        df = FeatureOptimizer._expand_dates(df)
        
        # 2. Categorical frequency encoding
        df = FeatureOptimizer._frequency_encode(df)
        
        return df

    @staticmethod
    def _expand_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Extracts day, month, year, dayofweek from date columns."""
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or \
               (df[col].dtype == 'object' and FeatureOptimizer._is_date_string(df[col])):
                
                date_series = pd.to_datetime(df[col])
                prefix = col + "_"
                df[prefix + 'year'] = date_series.dt.year
                df[prefix + 'month'] = date_series.dt.month
                df[prefix + 'day'] = date_series.dt.day
                df[prefix + 'dow'] = date_series.dt.dayofweek
                df = df.drop(columns=[col])
        return df

    @staticmethod
    def _frequency_encode(df: pd.DataFrame) -> pd.DataFrame:
        """Encodes high-cardinality categorical features with frequencies."""
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() > 10:
                freq = df[col].value_counts(normalize=True)
                df[col + '_freq'] = df[col].map(freq)
        return df

    @staticmethod
    def _is_date_string(series: pd.Series) -> bool:
        """Heuristic to check if a string column is actually a date."""
        try:
            # Check a sample
            sample = series.dropna().head(5)
            if sample.empty: return False
            pd.to_datetime(sample)
            return True
        except:
            return False
