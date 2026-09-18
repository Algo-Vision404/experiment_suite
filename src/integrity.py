import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp
from rich.console import Console
from rich.table import Table

console = Console()


class DataSchema(BaseModel):
    columns: Dict[str, str]
    required_columns: List[str]
    value_ranges: Optional[Dict[str, tuple]] = None


class IntegrityReport(BaseModel):
    data_hash: str
    row_count: int
    missing_values: Dict[str, int]
    schema_valid: bool
    warnings: List[str] = Field(default_factory=list)


class DriftReport(BaseModel):
    column: str
    p_value: float
    drifted: bool
    method: str = "Kolmogorov-Smirnov"


class IntegrityGuard:
    """Schema validation, deterministic hashing, and statistical integrity checks."""

    @staticmethod
    def get_data_hash(df: pd.DataFrame, chunk_size: int = 100_000) -> str:
        """Hash all rows deterministically, including columns and dtypes."""
        hasher = hashlib.sha256()
        hasher.update("|".join(map(str, df.columns)).encode("utf-8"))
        hasher.update("|".join(map(str, df.dtypes)).encode("utf-8"))
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            hashed = pd.util.hash_pandas_object(chunk, index=True).values
            hasher.update(hashed.tobytes())
        return hasher.hexdigest()

    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> List[DriftReport]:
        reports = []
        for col in reference_df.select_dtypes(include=[np.number]).columns:
            if col not in current_df.columns:
                continue
            r, c = reference_df[col].dropna(), current_df[col].dropna()
            if r.empty or c.empty:
                continue
            _, p_val = ks_2samp(r, c)
            reports.append(DriftReport(column=col, p_value=float(p_val), drifted=p_val < threshold))
        return reports

    def validate_schema(self, df: pd.DataFrame, schema: DataSchema) -> IntegrityReport:
        errors, warnings = [], []
        for col in schema.required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        for col, expected_type in schema.columns.items():
            if col in df.columns and str(df[col].dtype) != expected_type:
                errors.append(f"Type mismatch for {col}: expected {expected_type}, got {df[col].dtype}")
        if schema.value_ranges:
            for col, (vmin, vmax) in schema.value_ranges.items():
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    actual_min, actual_max = df[col].min(), df[col].max()
                    if actual_min < vmin or actual_max > vmax:
                        warnings.append(f"Range violation in {col}: got ({actual_min}, {actual_max}), expected ({vmin}, {vmax})")
        report = IntegrityReport(
            data_hash=self.get_data_hash(df),
            row_count=len(df),
            missing_values={k: int(v) for k, v in df.isnull().sum().items()},
            schema_valid=not errors,
            warnings=warnings + errors,
        )
        table = Table(title="Data Integrity Report", show_header=True)
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Data Hash", report.data_hash[:12] + "...")
        table.add_row("Rows", str(report.row_count))
        table.add_row("Schema Status", "PASS" if report.schema_valid else "FAIL")
        console.print(table)
        for warning in report.warnings:
            console.print(f"[yellow]Warning:[/] {warning}")
        return report
