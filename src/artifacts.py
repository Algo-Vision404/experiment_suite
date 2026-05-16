import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger("ArtifactManager")

class ArtifactManager:
    """
    Manages the persistence of processed data and metadata.
    Ensures that every run is versioned and reproducible.
    """
    
    def __init__(self, base_dir: str = "./artifacts"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, 
                 df: pd.DataFrame, 
                 context_history: Dict[str, Any], 
                 metrics: Dict[str, Any],
                 run_id: Optional[str] = None) -> str:
        """
        Saves the processed dataframe and its metadata as a versioned artifact.
        """
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        run_path = self.base_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save Data (using Parquet for performance and type preservation)
        data_path = run_path / "processed_data.parquet"
        df.to_parquet(data_path)
        
        # 2. Save Metadata (Ensure JSON serializability)
        def custom_serializer(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if hasattr(obj, 'dict'):
                return obj.dict()
            return str(obj)

        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "history": context_history,
            "metrics": metrics,
            "columns": list(df.columns),
            "shape": df.shape
        }
        
        meta_path = run_path / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=custom_serializer)
            
        logger.info(f"Artifacts saved to {run_path}")
        return str(run_path)

    def load_run(self, run_id: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads a previous run's data and metadata.
        """
        run_path = self.base_dir / run_id
        if not run_path.exists():
            raise FileNotFoundError(f"Run ID {run_id} not found in {self.base_dir}")
            
        df = pd.read_parquet(run_path / "processed_data.parquet")
        with open(run_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            
        return df, metadata
