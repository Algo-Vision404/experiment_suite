import pandas as pd
import numpy as np
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp
from rich.console import Console
from rich.table import Table

console = Console()

class DataSchema(BaseModel):
    """Defines the expected structure and constraints for a dataset."""
    columns: Dict[str, str]  # name: type (e.g., "int64", "float64", "object")
    required_columns: List[str]
    value_ranges: Optional[Dict[str, tuple]] = None  # name: (min, max)

class IntegrityReport(BaseModel):
    """Summary of data integrity checks."""
    data_hash: str
    row_count: int
    missing_values: Dict[str, int]
    schema_valid: bool
    warnings: List[str] = []

class DriftReport(BaseModel):
    """Report on statistical drift between two datasets."""
    column: str
    p_value: float
    drifted: bool
    method: str = "Kolmogorov-Smirnov"

class IntegrityGuard:
    """Handles high-performance data validation and drift detection."""
    
    @staticmethod
    def get_data_hash(df: pd.DataFrame) -> str:
        """Memory-efficient hashing for heavy-duty datasets."""
        # Use only a sample if dataset is multi-million row to maintain speed
        if len(df) > 1_000_000:
            sample = df.sample(100_000, random_state=42)
            hash_val = hashlib.sha256(pd.util.hash_pandas_object(sample).values).hexdigest()
            return f"sample_{hash_val}"
        
        hash_val = hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
        return hash_val

    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> List[DriftReport]:
        """Detects statistical drift in numeric columns using KS test."""
        drift_reports = []
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in current_df.columns:
                stat, p_val = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
                drift_reports.append(DriftReport(
                    column=col,
                    p_value=float(p_val),
                    drifted=p_val < threshold
                ))
        return drift_reports

    def validate_schema(self, df: pd.DataFrame, schema: DataSchema) -> IntegrityReport:
        """Validates the dataframe with a premium terminal report."""
        errors = []
        warnings = []
        
        # Check required columns
        for col in schema.required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check types
        for col, expected_type in schema.columns.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    warnings.append(f"Type mismatch for [bold cyan]{col}[/]: expected {expected_type}, got {actual_type}")

        # Check ranges
        if schema.value_ranges is not None:
            for col, range_vals in schema.value_ranges.items():
                vmin, vmax = range_vals
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    col_min, col_max = df[col].min(), df[col].max()
                    if col_min < vmin or col_max > vmax:
                        warnings.append(f"Range violation in [bold cyan]{col}[/]: got ({col_min:.2f}, {col_max:.2f}), expected ({vmin}, {vmax})")

        report = IntegrityReport(
            data_hash=self.get_data_hash(df),
            row_count=len(df),
            missing_values=df.isnull().sum().to_dict(),
            schema_valid=len(errors) == 0,
            warnings=warnings
        )

        # Premium Rich Output
        table = Table(title="Data Integrity Report", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value")
        
        table.add_row("Data Hash", report.data_hash[:12] + "...")
        table.add_row("Rows", str(report.row_count))
        table.add_row("Schema Status", "[bold green]PASS[/]" if report.schema_valid else "[bold red]FAIL[/]")
        
        console.print(table)
        
        if warnings:
            for w in warnings:
                console.print(f"[yellow]Warning:[/] {w}")
        
        return report
