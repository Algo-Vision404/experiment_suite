import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger("ArtifactManager")


class ArtifactManager:
    """Persists processed data and machine-readable run metadata."""

    def __init__(self, base_dir: str = "./artifacts"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, df: pd.DataFrame, context_history: Dict[str, Any], metrics: Dict[str, Any], run_id: Optional[str] = None) -> str:
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        run_path = self.base_dir / run_id
        run_path.mkdir(parents=True, exist_ok=False)
        df.to_parquet(run_path / "processed_data.parquet", index=True)
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "history": context_history,
            "metrics": metrics,
            "columns": list(df.columns),
            "shape": list(df.shape),
        }
        with (run_path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=self._serialize)
        return str(run_path)

    @staticmethod
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)

    def load_run(self, run_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        run_path = self.base_dir / run_id
        if not run_path.is_dir():
            raise FileNotFoundError(f"Run ID {run_id} not found in {self.base_dir}")
        data_file, meta_file = run_path / "processed_data.parquet", run_path / "metadata.json"
        if not data_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Run {run_id} is incomplete")
        df = pd.read_parquet(data_file)
        with meta_file.open("r", encoding="utf-8") as f:
            return df, json.load(f)
