import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("DriftMonitor")

class DriftMonitor:
    """
    Monitors data drift between a reference dataset and current data
    using statistical tests (Kolmogorov-Smirnov).
    """
    
    @staticmethod
    def calculate_drift(reference_df: pd.DataFrame, 
                        current_df: pd.DataFrame, 
                        threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detects drift in numeric columns using the KS test.
        A p-value < threshold indicates a high probability that the distributions differ.
        """
        drift_report = {}
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in current_df.columns:
                stat, p_val = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
                drift_report[col] = {
                    "p_value": float(p_val),
                    "statistic": float(stat),
                    "drifted": p_val < threshold
                }
                
        drifted_cols = [col for col, res in drift_report.items() if res['drifted']]
        
        return {
            "drift_detected": len(drifted_cols) > 0,
            "drifted_columns": drifted_cols,
            "detailed_report": drift_report,
            "drift_percentage": (len(drifted_cols) / len(numeric_cols)) * 100 if len(numeric_cols) > 0 else 0
        }
