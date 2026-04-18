import pandas as pd
import os
from typing import Union, Optional
from pathlib import Path

class DataIngestor:
    """Handles data loading from various formats with initial integrity checks."""
    
    @staticmethod
    def load(file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads data from CSV, JSON, or Parquet based on extension."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {file_path}")
            
        extension = path.suffix.lower()
        
        try:
            if extension == '.csv':
                return pd.read_csv(file_path)
            elif extension == '.json':
                return pd.read_json(file_path)
            elif extension == '.parquet':
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {str(e)}")

    @staticmethod
    def inspect(df: pd.DataFrame) -> dict:
        """Returns high-level statistics of the ingested data."""
        return {
            "size": len(df),
            "columns": list(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "dtypes": df.dtypes.to_dict()
        }
