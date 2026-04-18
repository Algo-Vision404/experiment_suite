# ML Experimental Standardization Suite

Experimental tools to reduce manual toil and improve reproducibility in machine learning workflows.

## Components

### 1. MLDataEngine (`src/engine.py`) - **NEW**
- **Orchestration**: Automatically runs Ingestion -> Validation -> Cleaning -> Feature Engineering -> Final Validation.
- **Audit Trail**: Generates a detailed history of every step, including hashes and statistics for reproducibility.

### 2. AutoCleaner (`src/cleaning.py`) - **NEW**
- **Deduplication**: Automatically identifies and removes duplicate rows.
- **Smart Imputation**: Uses median for numeric and mode for categorical missing values.
- **Clipping**: Automatically clips outliers at 1st and 99th percentiles.

### 3. FeatureOptimizer (`src/engineering.py`) - **NEW**
- **Date Expansion**: Automatically converts date strings/columns into year, month, day, and day-of-week features.
- **Frequency Encoding**: Encodes high-cardinality categorical features using frequency distributions.

### 4. IntegrityGuard (`src/integrity.py`)
- **Schema Validation**: Ensures incoming data matches expected types and required columns.
- **Data Hashing**: Generates SHA-256 signatures of datasets to detect data contamination or drift.

### 5. AutoGenerator (`src/autogen.py`)
- **Recipe Engine**: Suggests preprocessing steps (Scaling, Encoding) based on data distribution.
- **Schema Inference**: Automatically generates schemas from raw dataframes.

## Getting Started

1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Run the End-to-End Demo**:
   ```powershell
   python examples/auto_pipeline_demo.py
   ```

3. **Run the Standard Pipeline Demo**:
   ```powershell
   python examples/demo_run.py
   ```

## Goals
- **Standardization**: Reduce "spaghetti" preprocessing scripts.
- **Reproducibility**: If a model fails, the data hash and metadata tell you exactly what state caused it.
- **Speed**: Automate the repetitive "Check nulls -> Check types -> Scaler" workflow.
