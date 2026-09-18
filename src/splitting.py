"""Leakage-aware dataset splitting helpers."""
from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split

def split_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    group_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if target not in df:
        raise KeyError(f"Target column not found: {target}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if group_column:
        if group_column not in df:
            raise KeyError(f"Group column not found: {group_column}")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df[group_column]))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    stratify_values = df[target] if stratify else None
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_values)
    return train.copy(), test.copy()
