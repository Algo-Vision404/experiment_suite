from src.cleaning import AutoCleaner

def test_cleaner_removes_duplicates_and_imputes(dirty_df):
    result = AutoCleaner().clean(dirty_df)
    assert len(result) == 4
    assert result.isna().sum().sum() == 0

def test_target_is_not_imputed():
    import pandas as pd
    df = pd.DataFrame({"x": [1.0, None, 3.0], "target": [0, None, 1]})
    result = AutoCleaner(target_column="target").clean(df)
    assert result["target"].isna().sum() == 1
