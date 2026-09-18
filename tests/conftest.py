import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age": rng.integers(18, 70, 100),
        "income": rng.normal(50000, 10000, 100),
        "category": rng.choice(["A", "B", "C"], 100),
    })

@pytest.fixture
def dirty_df():
    df = pd.DataFrame({
        "a": [1, 2, 2, np.nan, 100],
        "b": ["x", "y", "y", None, "x"],
    })
    return df
