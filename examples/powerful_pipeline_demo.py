import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import MLDataEngine, EngineConfig

def create_data(file_path: str, drift: bool = False):
    """Creates a dataset, optionally with distribution drift."""
    np.random.seed(42 if not drift else 43)
    n_samples = 500
    
    # Baseline features
    score_mean = 100 if not drift else 120 # Drifting the mean
    score_std = 20 if not drift else 30
    
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='h').strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': np.random.randint(1000, 2000, n_samples),
        'score': np.random.normal(score_mean, score_std, n_samples).tolist(),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'conversion': np.random.uniform(0, 1, n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df

def main():
    # 1. Setup paths
    ref_path = "reference_data.csv"
    cur_path = "current_data.csv"
    
    print("Creating Reference and Current datasets...")
    create_data(ref_path, drift=False)
    create_data(cur_path, drift=True) # Current data has drifted!
    
    # 2. Configure the Engine
    config = EngineConfig(
        output_dir="./experiment_artifacts",
        drift_threshold=0.01,
        enable_persistence=True
    )
    
    engine = MLDataEngine(config=config)
    
    # 3. Run Pipeline with Drift Detection and Persistence
    print("\nExecuting Powerful Pipeline...")
    processed_df = engine.run_pipeline(
        input_path=cur_path,
        target_column='conversion',
        reference_path=ref_path # Enable Drift Monitoring
    )
    
    # 4. Cleanup
    for p in [ref_path, cur_path]:
        if os.path.exists(p):
            os.remove(p)

if __name__ == "__main__":
    main()
