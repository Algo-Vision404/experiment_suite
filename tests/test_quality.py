import pandas as pd
from src.quality import DataHealthScout

def test_health_score_is_bounded(sample_df):
    score = DataHealthScout.calculate_health_score(sample_df)["overall_health_score"]
    assert 0 <= score <= 100

def test_leakage_detector_flags_high_correlation():
    df = pd.DataFrame({"target": [0,1,0,1,0,1], "leak": [0,1,0,1,0,1], "noise": [1,4,2,8,3,9]})
    assert "leak" in DataHealthScout.detect_target_leakage(df, "target")
