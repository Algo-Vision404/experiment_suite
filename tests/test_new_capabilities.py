import pandas as pd
from src.evaluation import classification_metrics, regression_metrics
from src.experiments import ExperimentRegistry
from src.splitting import split_dataset

def test_evaluation_metrics():
    assert classification_metrics([0, 1, 1], [0, 1, 0])["accuracy"] > 0
    assert regression_metrics([1, 2, 3], [1, 2, 4])["rmse"] > 0

def test_split_has_no_overlap():
    df = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
    train, test = split_dataset(df, "target", stratify=True)
    assert set(train.index).isdisjoint(test.index)

def test_registry_round_trip(tmp_path):
    registry = ExperimentRegistry(str(tmp_path / "experiments.json"))
    registry.record("demo", "r1", {"accuracy": 0.8})
    registry.record("demo", "r2", {"accuracy": 0.9})
    assert len(registry.list_runs("demo")) == 2
    assert registry.compare("r1", "r2")["metrics"]["accuracy"]["b"] == 0.9
