import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("AnomalyDetector")


class AnomalyDetector:
    """Robust statistical anomaly analysis for numeric features."""

    @staticmethod
    def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, List[Any]]:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        anomalies: Dict[str, List[Any]] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].replace([np.inf, -np.inf], np.nan)
            std = series.std()
            if pd.isna(std) or std == 0:
                continue
            z_scores = ((series - series.mean()) / std).abs()
            indices = df.index[z_scores > threshold].tolist()
            if indices:
                anomalies[col] = indices
        return anomalies

    @staticmethod
    def detect_iqr_outliers(df: pd.DataFrame) -> Dict[str, List[Any]]:
        anomalies: Dict[str, List[Any]] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].replace([np.inf, -np.inf], np.nan)
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
            indices = df.index[mask.fillna(False)].tolist()
            if indices:
                anomalies[col] = indices
        return anomalies

    @staticmethod
    def calculate_iqr_bounds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        bounds = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].replace([np.inf, -np.inf], np.nan)
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            bounds[col] = {"lower": float(q1 - 1.5 * iqr), "upper": float(q3 + 1.5 * iqr)}
        return bounds

    @classmethod
    def analyze(cls, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        z = cls.detect_outliers(df, threshold)
        iqr = cls.detect_iqr_outliers(df)
        z_rows = {i for indices in z.values() for i in indices}
        iqr_rows = {i for indices in iqr.values() for i in indices}
        return {
            "z_score_anomalies": z,
            "iqr_anomalies": iqr,
            "iqr_bounds": cls.calculate_iqr_bounds(df),
            "anomaly_count": len(z_rows),
            "affected_cells_zscore": sum(len(v) for v in z.values()),
            "affected_rows_zscore": len(z_rows),
            "anomaly_percentage": (len(z_rows) / len(df)) * 100 if len(df) else 0.0,
            "union_affected_rows": len(z_rows | iqr_rows),
        }
