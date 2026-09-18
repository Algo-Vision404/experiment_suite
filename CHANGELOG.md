# Changelog

## 2.1.0 - 2026-09-18

### Added
- Modern project metadata and CLI entry point.
- Deterministic chunked SHA-256 hashing.
- Numeric and categorical drift analysis.
- IQR anomaly detection and row-aware anomaly metrics.
- Leakage-aware dataset splitting and leakage inspection.
- Standard classification/regression evaluation helpers.
- Local experiment registry and run comparison.
- Environment and Git provenance capture.
- CI, coverage, linting, tests, security policy, and contribution guide.

### Fixed
- Configured anomaly thresholds are now respected.
- Simulated progress delays were removed.
- Artifact run IDs include microseconds to reduce collisions.
- StandardPipeline validates schemas before fitting.
- Demo paths are portable across machines.
