import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import MLDataEngine

def create_raw_dirty_data(file_path: str):
    """Creates a raw dataset with issues (missing values, duplicates, mixed types, leakage)."""
    np.random.seed(42)
    n_samples = 200
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='D').strftime('%Y-%m-%d'),
        'user_id': np.random.randint(1000, 1050, n_samples),
        'score': np.random.normal(100, 20, n_samples).tolist(),
        'useless_filler': ['CONSTANT'] * n_samples, # Constant column
        'category': np.random.choice(['A', 'B', 'C', None], n_samples),
        'growth': np.random.uniform(0, 1, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Introduce missing values in score
    df.loc[np.random.choice(df.index, 10), 'score'] = np.nan
    
    # Introduce target leakage (feature highly correlated with growth)
    df['leakage_proxy'] = df['growth'] * 1.05 + np.random.normal(0, 0.01, n_samples)
    
    # Introduce duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    
    df.to_csv(file_path, index=False)

def main():
    # 1. Setup
    raw_path = "raw_data.csv"
    create_raw_dirty_data(raw_path)
    
    engine = MLDataEngine()
    
    # 2. Run End-to-End Pipeline
    # The engine now handles all the visual feedback internally!
    processed_df = engine.run_pipeline(
        input_path=raw_path,
        target_column='growth'
    )
    
    # Cleanup
    if os.path.exists(raw_path):
        os.remove(raw_path)

if __name__ == "__main__":
    main()
