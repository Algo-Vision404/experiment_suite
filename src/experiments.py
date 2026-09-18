"""Lightweight local experiment registry backed by JSON."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

class ExperimentRegistry:
    def __init__(self, path: str = "./artifacts/experiments.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def record(self, experiment: str, run_id: str, metrics: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        records = self._read()
        record = {
            "experiment": experiment,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "metadata": metadata or {},
        }
        records.append(record)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
        tmp.replace(self.path)
        return record

    def list_runs(self, experiment: Optional[str] = None) -> List[Dict[str, Any]]:
        records = self._read()
        return [r for r in records if experiment is None or r["experiment"] == experiment]

    def compare(self, run_a: str, run_b: str) -> Dict[str, Any]:
        records = {r["run_id"]: r for r in self._read()}
        if run_a not in records or run_b not in records:
            raise KeyError("Both run IDs must exist in the registry")
        a, b = records[run_a]["metrics"], records[run_b]["metrics"]
        keys = sorted(set(a) | set(b))
        return {"run_a": run_a, "run_b": run_b, "metrics": {
            k: {"a": a.get(k), "b": b.get(k)} for k in keys
        }}
