import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger("AnomalyDetector")

class AnomalyDetector:
    """
    Institutional-grade anomaly detection using statistical heuristics.
    Detects outliers using Z-score and Interquartile Range (IQR) methods.
    """
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, List[int]]:
        """
        Identifies outliers in numeric columns using Z-scores.
        Returns a dictionary of column names and indices of detected anomalies.
        """
        anomalies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_indices = df.index[z_scores > threshold].tolist()
            if outlier_indices:
                anomalies[col] = outlier_indices
                
        return anomalies

    @staticmethod
    def calculate_iqr_bounds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculates Upper and Lower bounds for all numeric columns using IQR.
        """
        bounds = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            bounds[col] = {
                "lower": q1 - 1.5 * iqr,
                "upper": q3 + 1.5 * iqr
            }
            
        return bounds

    @classmethod
    def analyze(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs a full anomaly analysis suite.
        """
        z_anomalies = cls.detect_outliers(df)
        iqr_bounds = cls.calculate_iqr_bounds(df)
        
        total_anomalies = sum(len(indices) for indices in z_anomalies.values())
        
        return {
            "z_score_anomalies": z_anomalies,
            "iqr_bounds": iqr_bounds,
            "anomaly_count": total_anomalies,
            "anomaly_percentage": (total_anomalies / (df.shape[0] * df.shape[1])) * 100 if df.size > 0 else 0
        }
