import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any

class DataHealthScout:
    """Advanced data quality and health analysis."""
    
    @staticmethod
    def calculate_health_score(df: pd.DataFrame) -> Dict[str, Any]:
        """Runs a comprehensive suite of health checks and returns a score."""
        n_rows, n_cols = df.shape
        
        # 1. Completeness Score
        null_ratio = df.isnull().mean().mean()
        completeness = 1.0 - null_ratio
        
        # 2. Uniqueness Score
        duplicate_ratio = df.duplicated().mean()
        uniqueness = 1.0 - duplicate_ratio
        
        # 3. Information Density (Detect constant/near-constant columns)
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        info_density = 1.0 - (len(constant_cols) / n_cols if n_cols > 0 else 0)
        
        # 4. Statistical Health (Detect high skewness)
        numeric_df = df.select_dtypes(include=[np.number])
        skewed_cols = []
        if not numeric_df.empty:
            for col in numeric_df.columns:
                try:
                    skew = numeric_df[col].skew()
                    if abs(skew) > 2:
                        skewed_cols.append(col)
                except:
                    pass
        skew_score = 1.0 - (len(skewed_cols) / len(numeric_df.columns) if not numeric_df.empty else 0)

        # Final Weighted Score
        final_score = (completeness * 0.4) + (uniqueness * 0.2) + (info_density * 0.2) + (skew_score * 0.2)
        
        return {
            "overall_health_score": round(final_score * 100, 2),
            "metrics": {
                "completeness": round(completeness, 4),
                "uniqueness": round(uniqueness, 4),
                "info_density": round(info_density, 4),
                "skew_score": round(skew_score, 4)
            },
            "anomalies": {
                "constant_columns": constant_cols,
                "highly_skewed_columns": skewed_cols,
                "null_count_total": int(df.isnull().sum().sum())
            }
        }

    @staticmethod
    def detect_target_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.95) -> List[str]:
        """Identifies features that are too highly correlated with the target."""
        if target_col not in df.columns:
            return []
            
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col not in numeric_df.columns:
            # Handle categorical target correlation? (Maybe later)
            return []
            
        correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        leaky_cols = correlations[(correlations > threshold) & (correlations < 1.0)].index.tolist()
        
        return leaky_cols
