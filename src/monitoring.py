import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger("DriftMonitor")


class DriftMonitor:
    """Detects numeric and categorical distribution drift with effect-size context."""

    @staticmethod
    def calculate_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")

        report: Dict[str, Dict[str, Any]] = {}
        all_columns = sorted(set(reference_df.columns) | set(current_df.columns))
        missing_in_current = sorted(set(reference_df.columns) - set(current_df.columns))
        new_columns = sorted(set(current_df.columns) - set(reference_df.columns))

        for col in all_columns:
            if col not in reference_df or col not in current_df:
                continue
            ref, cur = reference_df[col], current_df[col]
            if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
                r = ref.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                c = cur.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                if len(r) == 0 or len(c) == 0:
                    continue
                stat, p = ks_2samp(r, c)
                scale = np.std(r)
                effect = float(wasserstein_distance(r, c) / scale) if scale > 0 else float(wasserstein_distance(r, c))
                report[col] = {"method": "KS+normalized_Wasserstein", "p_value": float(p), "statistic": float(stat), "effect_size": effect, "drifted": bool(p < threshold)}
            else:
                r = ref.astype("string").fillna("<NA>").value_counts(normalize=True)
                c = cur.astype("string").fillna("<NA>").value_counts(normalize=True)
                categories = sorted(set(r.index) | set(c.index))
                rp = np.array([r.get(k, 0.0) for k in categories], dtype=float)
                cp = np.array([c.get(k, 0.0) for k in categories], dtype=float)
                js = float(jensenshannon(rp, cp, base=2.0)) if categories else 0.0
                report[col] = {"method": "Jensen-Shannon", "distance": js, "drifted": js > threshold}

        drifted = [c for c, v in report.items() if v["drifted"]]
        return {
            "drift_detected": bool(drifted or missing_in_current or new_columns),
            "drifted_columns": drifted,
            "detailed_report": report,
            "missing_columns": missing_in_current,
            "new_columns": new_columns,
            "drift_percentage": (len(drifted) / len(report)) * 100 if report else 0.0,
        }
