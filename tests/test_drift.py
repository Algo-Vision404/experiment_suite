import pandas as pd
import numpy as np
from src.monitoring import DriftMonitor

def test_drift_detects_distribution_shift():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": rng.normal(5, 1, 500)})
    report = DriftMonitor.calculate_drift(ref, cur)
    assert report["drift_detected"] is True
    assert "x" in report["drifted_columns"]

def test_drift_handles_categorical_columns_without_crashing():
    ref = pd.DataFrame({"category": ["a", "b", "a", "b"]})
    cur = pd.DataFrame({"category": ["a", "a", "a", "a"]})
    report = DriftMonitor.calculate_drift(ref, cur)
    assert report["drift_detected"] is False
