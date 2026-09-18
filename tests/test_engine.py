import pandas as pd
from src import MLDataEngine, EngineConfig

def test_engine_end_to_end(tmp_path):
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [100, 200, 300, 400],
        "category": ["a", "b", "c", "d"],
        "target": [0, 1, 0, 1],
    })
    source = tmp_path / "data.csv"
    df.to_csv(source, index=False)
    engine = MLDataEngine(EngineConfig(output_dir=str(tmp_path / "artifacts")))
    result = engine.run_pipeline(str(source), target_column="target")
    assert len(result) == len(df)
    assert (tmp_path / "artifacts").exists()
