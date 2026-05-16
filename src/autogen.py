import pandas as pd
import numpy as np
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from src.integrity import DataSchema

console = Console()

class AutoGenerator:
    """Analyzes data and generates high-performance pipeline recipes."""
    
    @staticmethod
    def infer_schema(df: pd.DataFrame) -> DataSchema:
        """Infers a heavy-duty schema from a dataframe."""
        columns = {col: str(df[col].dtype) for col in df.columns}
        required = list(df.columns)
        
        ranges = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                ranges[col] = (float(df[col].min()), float(df[col].max()))
        
        return DataSchema(
            columns=columns,
            required_columns=required,
            value_ranges=ranges
        )

    @staticmethod
    def suggest_pipeline_steps(df: pd.DataFrame):
        """Advanced heuristics for preprocessing steps."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        steps = []
        
        # Missing Value Strategy
        null_pct = df.isnull().mean()
        high_null_cols = null_pct[null_pct > 0.05].index.tolist()
        
        if not high_null_cols:
            steps.append({"type": "SimpleImputer", "target": "all", "strategy": "median", "reason": "Basic cleanup"})
        else:
            steps.append({"type": "IterativeImputer", "target": high_null_cols, "reason": "High missingness detected"})

        if numeric_cols:
            steps.append({
                "type": "StandardScaler",
                "target": numeric_cols,
                "reason": f"Normalization for {len(numeric_cols)} features"
            })
        
        if categorical_cols:
            steps.append({
                "type": "OneHotEncoder",
                "target": categorical_cols,
                "reason": "Categorical cardinality management"
            })
            
        return steps

    @classmethod
    def generate_report(cls, df: pd.DataFrame):
        """Generates a premium diagnostic report."""
        schema = cls.infer_schema(df)
        steps = cls.suggest_pipeline_steps(df)
        
        table = Table(title="ML Pipeline Recipe", show_header=True, header_style="bold yellow")
        table.add_column("Proposed Step", style="cyan")
        table.add_column("Configuration")
        table.add_column("Heuristic Reason")
        
        for step in steps:
            config = step.get('strategy', 'Default')
            table.add_row(step['type'], config, step['reason'])
        
        console.print(Panel(f"Analyzed [bold green]{len(df)}[/] samples across [bold green]{len(df.columns)}[/] columns.", title="AutoGenerator v2.0"))
        console.print(table)
        
        return {
            "schema": schema.dict(),
            "suggested_steps": steps
        }
