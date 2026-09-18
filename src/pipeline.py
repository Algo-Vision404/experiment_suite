import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from sklearn.pipeline import Pipeline

from src.integrity import DataSchema, IntegrityGuard
from src.provenance import capture_environment

console = Console()


class StandardPipeline:
    """Scikit-learn pipeline wrapper with schema, drift, metadata, and reproducibility checks."""

    def __init__(self, name: str, pipeline_steps: list, schema: Optional[DataSchema] = None):
        if not name.strip():
            raise ValueError("Pipeline name cannot be empty")
        self.name = name
        self.pipeline = Pipeline(steps=pipeline_steps)
        self.schema = schema
        self.guard = IntegrityGuard()
        self.reference_df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runs": [],
            "environment": capture_environment(),
        }

    def fit(self, X, y, **fit_params):
        X = X.copy() if hasattr(X, "copy") else X
        if self.schema:
            report = self.guard.validate_schema(X, self.schema)
            if not report.schema_valid:
                raise ValueError(f"Schema validation failed: {report.warnings}")
            if self.reference_df is not None:
                drifts = self.guard.detect_drift(self.reference_df, X)
                for d in drifts:
                    if d.drifted:
                        console.print(Panel(
                            f"[bold red]DRIFT DETECTED[/] in column: {d.column}\n"
                            f"(p-value: {d.p_value:.4f})",
                            border_style="red",
                        ))
            else:
                self.reference_df = X.copy()
            self.metadata["runs"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_hash": report.data_hash,
                "row_count": len(X),
            })
        self.pipeline.fit(X, y, **fit_params)
        return self

    def save(self, directory: str = "artifacts"):
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        base_path = os.path.join(directory, f"{self.name}_{timestamp}")
        joblib.dump(self.pipeline, f"{base_path}_model.joblib")
        with open(f"{base_path}_meta.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.metadata, f, sort_keys=False)
        return base_path

    def predict(self, X):
        return self.pipeline.predict(X)
