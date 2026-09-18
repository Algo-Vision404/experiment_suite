import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel, Field

from .anomaly import AnomalyDetector
from .artifacts import ArtifactManager
from .autogen import AutoGenerator
from .cleaning import AutoCleaner
from .engineering import FeatureOptimizer
from .ingestion import DataIngestor
from .integrity import DataSchema, IntegrityGuard
from .monitoring import DriftMonitor
from .provenance import capture_environment
from .quality import DataHealthScout
from .visualizer import SpectacularReporter

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("MLDataEngine")


class EngineConfig(BaseModel):
    output_dir: str = "./artifacts"
    drift_threshold: float = Field(default=0.05, gt=0, lt=1)
    anomaly_threshold: float = Field(default=3.0, gt=0)
    enable_persistence: bool = True
    stop_on_leakage: bool = False
    random_seed: int = 42


@dataclass
class PipelineContext:
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    input_path: str = ""
    config: EngineConfig = field(default_factory=EngineConfig)
    target_column: Optional[str] = None
    history: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def record_step(self, step_name: str, metadata: Any) -> None:
        self.history[step_name] = metadata


class MLDataEngine:
    """Orchestrates ingestion, quality, integrity, cleaning, engineering, and artifacts."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.guard = IntegrityGuard()
        self.reporter = SpectacularReporter()
        self.scout = DataHealthScout()
        self.detector = AnomalyDetector()
        self.artifact_manager = ArtifactManager(base_dir=self.config.output_dir)
        self.monitor = DriftMonitor()

    def run_pipeline(
        self,
        input_path: str,
        target_column: Optional[str] = None,
        schema: Optional[DataSchema] = None,
        reference_path: Optional[str] = None,
    ) -> pd.DataFrame:
        ctx = PipelineContext(input_path=input_path, config=self.config, target_column=target_column)
        self.reporter.welcome_banner()
        df = DataIngestor.load(input_path)
        ctx.record_step("ingestion", {
            "hash": self.guard.get_data_hash(df),
            "shape": list(df.shape),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        health_report = self.scout.calculate_health_score(df)
        anomaly_report = self.detector.analyze(df, threshold=self.config.anomaly_threshold)
        ctx.record_step("intelligence", {"health": health_report, "anomalies": anomaly_report})

        if reference_path:
            ref_df = DataIngestor.load(reference_path)
            ctx.record_step("drift", self.monitor.calculate_drift(ref_df, df, self.config.drift_threshold))

        self.reporter.print_health_dashboard(health_report)

        if target_column:
            leaky_cols = self.scout.detect_target_leakage(df, target_column)
            if leaky_cols:
                ctx.record_step("alerts", {"target_leakage": leaky_cols})
                if self.config.stop_on_leakage:
                    raise ValueError(f"Pipeline halted due to target leakage in: {leaky_cols}")

        if schema is None:
            schema = AutoGenerator.infer_schema(df)
        validation_report = self.guard.validate_schema(df, schema)
        if not validation_report.schema_valid:
            raise ValueError(f"Input schema validation failed: {validation_report.warnings}")
        ctx.record_step("validation_initial", validation_report.model_dump())

        cleaner = AutoCleaner(target_column=target_column)
        df = cleaner.clean(df)
        ctx.record_step("cleaning", cleaner.stats)

        df = FeatureOptimizer.engineer_features(df)
        ctx.record_step("engineering", {"engineered_columns": list(df.columns)})

        final_schema = AutoGenerator.infer_schema(df)
        final_report = self.guard.validate_schema(df, final_schema)
        ctx.record_step("validation_final", final_report.model_dump())
        ctx.metrics.update({
            "processed_hash": final_report.data_hash,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": (datetime.now(timezone.utc) - ctx.start_time).total_seconds(),
            "environment": capture_environment(),
        })

        if self.config.enable_persistence:
            ctx.metrics["artifact_path"] = self.artifact_manager.save_run(df, ctx.history, ctx.metrics)
        self.reporter.finish_summary(ctx.history | ctx.metrics)
        return df

    def get_summary(self, ctx: PipelineContext) -> Dict[str, Any]:
        return ctx.history | ctx.metrics
