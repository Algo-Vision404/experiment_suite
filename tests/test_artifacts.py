import pandas as pd
from src.artifacts import ArtifactManager

def test_artifact_round_trip(tmp_path):
    df = pd.DataFrame({"x": [1,2,3], "y": ["a","b","c"]})
    manager = ArtifactManager(tmp_path)
    path = manager.save_run(df, {"step": {"ok": True}}, {"score": 0.9}, run_id="test-run")
    loaded_df, metadata = manager.load_run("test-run")
    pd.testing.assert_frame_equal(df, loaded_df)
    assert metadata["run_id"] == "test-run"
    assert path.endswith("test-run")
