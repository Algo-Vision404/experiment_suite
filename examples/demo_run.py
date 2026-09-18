import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.integrity import DataSchema
from src.pipeline import StandardPipeline
from src.autogen import AutoGenerator


def main():
    rng = np.random.default_rng(42)
    data = {
        "age": rng.integers(18, 80, 100),
        "income": rng.normal(50000, 15000, 100),
        "target": rng.integers(0, 2, 100),
    }
    df_ref = pd.DataFrame(data)

    analysis = AutoGenerator.generate_report(df_ref.drop("target", axis=1))
    schema = DataSchema(**analysis["schema"])

    std_pipe = StandardPipeline(
        name="HeavyDutyPredictor",
        pipeline_steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ],
        schema=schema,
    )

    X_ref, y_ref = df_ref[["age", "income"]], df_ref["target"]
    std_pipe.fit(X_ref, y_ref)

    drifted = pd.DataFrame({
        "age": rng.integers(18, 80, 100),
        "income": rng.normal(90000, 5000, 100),
    })
    std_pipe.fit(drifted, rng.integers(0, 2, 100))

    artifact_path = std_pipe.save(directory="./artifacts")
    print(f"Artifacts tracked at: {artifact_path}")


if __name__ == "__main__":
    main()
