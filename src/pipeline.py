import yaml
import time
import joblib
import os
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional
from sklearn.pipeline import Pipeline
from src.integrity import IntegrityGuard, DataSchema, DriftReport
from rich.console import Console
from rich.panel import Panel

console = Console()

class StandardPipeline:
    """A standardized ML pipeline wrapper with auto-logging and integrity checks."""
    
    def __init__(self, name: str, pipeline_steps: list, schema: Optional[DataSchema] = None):
        self.name = name
        self.pipeline = Pipeline(steps=pipeline_steps)
        self.schema = schema
        self.guard = IntegrityGuard()
        self.config: Dict[str, Any] = {}
        self.reference_df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "runs": []
        }

    def fit(self, X, y, **fit_params):
        """Fit with automatic integrity checks and drift detection."""
        # 1. Integrity & Drift
        if self.schema:
            report = self.guard.validate_schema(X, self.schema)
            
            # Drift Check if we have reference data
            if self.reference_df is not None:
                drifts = self.guard.detect_drift(self.reference_df, X)
                for d in drifts:
                    if d.drifted:
                        console.print(Panel(f"[bold red]DRIFT DETECTED[/] in column: {d.column}\n(p-value: {d.p_value:.4f})", border_style="red"))
            else:
                self.reference_df = X.copy() # Set first run as reference
            
            run_info = {
                "timestamp": datetime.now().isoformat(),
                "data_hash": report.data_hash,
                "row_count": report.row_count
            }
            self.metadata["runs"].append(run_info)

        # 2. Fit Pipeline
        with console.status(f"[bold green]Fitting {self.name} pipeline..."):
            self.pipeline.fit(X, y, **fit_params)
        
        console.print(f"[bold green]{self.name}[/] model trained successfully!\n")
        return self

    def save(self, directory: str = "artifacts"):
        """Save the pipeline, config, and metadata for reproducibility."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(directory, f"{self.name}_{timestamp}")
        
        # Save Model
        joblib.dump(self.pipeline, f"{base_path}_model.joblib")
        
        # Save Metadata/Config
        with open(f"{base_path}_meta.yaml", 'w') as f:
            yaml.dump(self.metadata, f)
            
        print(f"[Repository] Artifacts saved to {base_path}*")
        return base_path

    def predict(self, X):
        return self.pipeline.predict(X)
