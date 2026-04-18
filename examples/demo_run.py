import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.integrity import DataSchema
from src.pipeline import StandardPipeline
from src.autogen import AutoGenerator

def main():
    # 1. Create a synthetic dataset
    print("[Demo] Creating reference dataset...")
    data = {
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'target': np.random.randint(0, 2, 100)
    }
    df_ref = pd.DataFrame(data)

    # 2. Automated Data Profiling & Schema Inference
    analysis = AutoGenerator.generate_report(df_ref.drop('target', axis=1))
    schema = DataSchema(**analysis['schema'])

    # 3. Setup Standardized Pipeline
    pipeline_steps = [
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ]
    
    std_pipe = StandardPipeline(
        name="HeavyDutyPredictor",
        pipeline_steps=pipeline_steps,
        schema=schema
    )

    # 4. Training Run (Reference)
    X_ref = df_ref[['age', 'income']]
    y_ref = df_ref['target']
    std_pipe.fit(X_ref, y_ref)

    # 5. Simulate Statistical Drift
    print("\n[Demo] Simulating massive income drift for Run 2...")
    data_drifted = {
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(90000, 5000, 100), # Mean shifted from 50k to 90k
        'target': np.random.randint(0, 2, 100)
    }
    df_new = pd.DataFrame(data_drifted)
    X_new = df_new[['age', 'income']]
    y_new = df_new['target']

    # 6. Run 2 (Should trigger Drift Alert)
    std_pipe.fit(X_new, y_new)

    # 7. Reproducibility Check
    artifact_path = std_pipe.save(directory="C:/Users/kop/.gemini/antigravity/scratch/ml-experimental-pipeline/artifacts")
    print(f"\n[Artifacts] Tracked at: {artifact_path}")

if __name__ == "__main__":
    main()
