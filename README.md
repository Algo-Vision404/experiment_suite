# ML Experimental Standardization Suite (EXSS)

[![Architecture](https://img.shields.io/badge/Architecture-Institutional--Grade-blueviolet)](#system-architecture)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

> **Autonomous Data Intelligence for High-Performance Machine Learning.**

The **Experimental Standardization Suite (EXSS)** is a production-ready orchestration layer designed to eliminate manual data preparation toil. It transforms raw, chaotic datasets into deterministic, model-ready features using advanced statistical heuristics and autonomous cleaning agents.

---

## Core Value Proposition

-   **Deterministic Lineage**: Every dataset is cryptographically signed (SHA-256) and saved as a versioned artifact for 100% reproducibility.
-   **Autonomous Optimization**: Intelligent agents analyze data distributions to dynamically inject cleaning and engineering steps.
-   **Statistical Monitoring**: Integrated drift detection (KS-test) flags distribution shifts before they impact production models.
-   **High-Fidelity Observability**: Real-time terminal dashboards provide institutional-grade telemetry on data health and pipeline performance.

## Quick Start

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Execute Autonomous Pipeline
```python
from src import MLDataEngine, EngineConfig

# Configure the institutional engine
config = EngineConfig(
    output_dir="./experiment_artifacts",
    drift_threshold=0.01,
    enable_persistence=True
)

engine = MLDataEngine(config=config)

# Run orchestration with drift detection
df = engine.run_pipeline(
    input_path="current_data.csv",
    reference_path="reference_data.csv", # Enable drift monitoring
    target_column="conversion"
)
```

## System Architecture

EXSS is built on a modular, decoupled architecture that separates **Intelligence**, **Execution**, and **Reporting**. It follows a "Stateful Pipeline" pattern where every processing node is independent but governed by a central orchestration engine.

### High-Level Design (System Topology)

```mermaid
graph TD
    A[Raw Data Source] --> B[Data Ingestor]
    B --> C[Integrity Guard]
    C --> D[Data Health Scout]
    D --> E[Auto-Generator]
    E --> F[Automated Pipeline]
    
    subgraph "Intelligent Pipeline"
        F --> F1[Auto-Cleaner]
        F1 --> F2[Feature Optimizer]
        F2 --> F3[Anomaly Detector]
        F3 --> F4[Drift Monitor]
    end
    
    F4 --> G[Final Validation]
    G --> H[Artifact Manager]
    H --> I[Processed Data Artifact]
    H --> J[Audit Trail & Metrics]
    
    subgraph "Observability Layer"
        K[Spectacular Reporter]
        L[CLI Live Dashboard]
    end
    
    F -.-> K
    G -.-> K
    J -.-> L
```

### Core Components & Sub-Systems

#### 1. The Intelligence Layer
- **Auto-Generator**: Utilizes statistical distribution analysis to infer optimal data schemas and compression types.
- **Data Health Scout**: Performs high-fidelity checks for target leakage, sparse features, and information density.
- **Anomaly Detector**: Implements multi-strategy outlier detection (Z-Score & IQR) to filter noise.
- **Drift Monitor**: Integrated Kolmogorov-Smirnov (KS) statistical testing to detect feature distribution shifts against reference benchmarks.

#### 2. The Execution Engine
- **MLDataEngine**: The primary system orchestrator. It manages the `PipelineContext` and ensures atomic execution of all nodes.
- **EngineConfig**: A Pydantic-driven configuration layer providing strictly typed, environment-aware parameter management.
- **Context Management**: A stateful container that tracks telemetry, hashes, and transformation history throughout the run.

#### 3. The Artifact & Integrity Layer
- **Integrity Guard**: Enforces SHA-256 cryptographic signatures on every input/output state to guarantee data lineage.
- **Artifact Manager**: A production-grade persistence engine that saves versioned datasets as **Parquet** (preserving schema integrity) and metadata as **JSON**.

#### 4. Observability Suite
- **Spectacular Reporter**: A `Rich`-native telemetry dashboard providing sub-second feedback on pipeline performance and data health.
- **Audit Trails**: Fully serialized transformation logs for institutional compliance and experiment tracking integration.

## Data Flow Pattern

1.  **Ingestion**: Load data from CSV/Parquet and generate an initial integrity hash.
2.  **Analysis**: Perform a comprehensive health check. If quality is below the "Golden Threshold", the pipeline halts.
3.  **Dynamic Transformation**: The system applies cleaning and engineering steps based on the inferred schema and detected issues.
4.  **Verification**: A final hash and schema check are performed to ensure the transformation was deterministic and safe.
5.  **Artifact Generation**: The processed dataset and its corresponding metadata (audit trail) are saved as versioned artifacts.

## Visual Intelligence

The suite features a **Spectacular Reporter** powered by `Rich`, providing:
-   **Live Progress Tracking**: Real-time status of multi-stage transformations.
-   **Health Dashboards**: Visual breakdown of completeness, uniqueness, and information density.
-   **Audit Trails**: Comprehensive JSON summaries of every transformation applied.

---

"Standardizing the chaos of experimental data."
