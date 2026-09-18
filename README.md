# ML Experimental Standardization Suite (EXSS)

[![CI](https://github.com/Algo-Vision404/experiment_suite/actions/workflows/ci.yml/badge.svg)](https://github.com/Algo-Vision404/experiment_suite/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

EXSS is a reproducible ML data-engineering toolkit for turning experimental datasets into auditable, model-ready artifacts. It combines ingestion, schema validation, data-health analysis, leakage checks, anomaly detection, drift monitoring, transformations, evaluation, provenance, and experiment tracking.

## v2.1 upgrades

- Deterministic full-data SHA-256 hashing with chunked processing.
- Strict schema validation and explicit integrity warnings.
- Numeric drift with KS plus normalized Wasserstein effect size.
- Categorical drift with Jensen-Shannon divergence.
- Row-aware Z-score and IQR anomaly reporting.
- Configurable anomaly thresholds and reproducibility metadata.
- Leakage-aware random, stratified, and group splitting helpers.
- Standard classification and regression evaluation metrics.
- Local experiment registry with run comparison.
- Portable `exss` CLI.
- Automated tests across Python 3.9, 3.11, and 3.12 plus Ruff linting.
- Modern `pyproject.toml` packaging metadata and an MIT license.

## Quick start

Install:
```bash
pip install -e ".[dev]"
```

Profile a dataset:
```bash
exss profile data.csv
```

Compare reference/current distributions:
```bash
exss drift reference.csv current.csv --threshold 0.05
```

Python API:
```python
from src import MLDataEngine, EngineConfig

engine = MLDataEngine(EngineConfig(
    output_dir="./artifacts",
    drift_threshold=0.05,
    anomaly_threshold=3.0,
    enable_persistence=True,
) )

processed = engine.run_pipeline(
    input_path="current_data.csv",
    reference_path="reference_data.csv",
    target_column="conversion",
)
```

## Architecture

```text
Data source
   |
   v
Ingestion -> deterministic hash
   |
   +--> schema + integrity
   +--> health + leakage + anomalies
   +--> reference drift
   v
Cleaning -> feature engineering
   v
Final validation -> provenance -> artifact
   +--> experiment registry
   +--> evaluation
   +--> run comparison
   +--> CLI/reporting
```

## Design principles

**Reproducibility.** Persisted runs record input/output state, configuration, timestamps, and execution-environment metadata.

**Train/test safety.** Split before fitting transformations that learn statistics from data. The core cleaner is intentionally simple and should generally be applied to a training partition for model development.

**Statistical humility.** A p-value describes evidence under a statistical test, not business impact. EXSS reports effect-size context for numeric drift.

**Failure visibility.** Schema violations, missing artifacts, invalid configuration, and malformed inputs fail explicitly.

## Repository layout

```text
src/
  engine.py          orchestration
  ingestion.py       CSV/JSON/Parquet loading
  integrity.py       hashing + schema validation
  quality.py         data health + correlation leakage
  leakage.py         additional leakage inspection
  cleaning.py        deduplication + imputation + clipping
  engineering.py     date + frequency features
  anomaly.py         Z-score + IQR analysis
  monitoring.py      numeric/categorical drift
  splitting.py       leakage-aware data splitting
  evaluation.py      standard model metrics
  experiments.py     local run registry
  provenance.py      environment capture
  artifacts.py       persisted run artifacts
  cli.py             command-line interface

tests/               unit + integration coverage
examples/            runnable demonstrations
.github/workflows/    CI
```

## Scope

EXSS is a toolkit, not a hosted ML platform or a replacement for domain-specific validation. Automated cleaning and statistical thresholds should be reviewed against the actual modeling and business context.

"Standardizing the chaos of experimental data."