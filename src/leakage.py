"""Additional leakage checks that complement correlation-based detection."""
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

class LeakageDetector:
    @staticmethod
    def inspect(df: pd.DataFrame, target: str) -> Dict[str, Any]:
        if target not in df:
            raise KeyError(f"Target column not found: {target}")
        findings: List[Dict[str, Any]] = []
        target_values = df[target]
        target_set = set(target_values.dropna().tolist())
        for col in df.columns:
            if col == target:
                continue
            s = df[col]
            overlap = set(s.dropna().tolist()) & target_set
            if s.nunique(dropna=True) == target_values.nunique(dropna=True) and s.nunique(dropna=True) > 1:
                findings.append({"column": col, "type": "matching_cardinality", "severity": "review"})
            name = str(col).lower()
            if any(token in name for token in ("target", "label", "outcome", "future", "post_", "post-")):
                findings.append({"column": col, "type": "suspicious_name", "severity": "review"})
            if len(overlap) == 0 and s.notna().any():
                continue
        return {"target": target, "findings": findings, "count": len(findings)}
