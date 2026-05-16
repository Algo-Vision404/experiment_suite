import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from .ingestion import DataIngestor
from .cleaning import AutoCleaner
from .engineering import FeatureOptimizer
from .integrity import IntegrityGuard, DataSchema
from .autogen import AutoGenerator
from .quality import DataHealthScout
from .visualizer import SpectacularReporter
from .anomaly import AnomalyDetector
from .artifacts import ArtifactManager
from .monitoring import DriftMonitor

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("MLDataEngine")

class EngineConfig(BaseModel):
    """Institutional configuration for the ML Data Engine."""
    output_dir: str = "./artifacts"
    drift_threshold: float = 0.05
    anomaly_threshold: float = 3.0
    enable_persistence: bool = True
    stop_on_leakage: bool = False

@dataclass
class PipelineContext:
    """Institutional state container for pipeline telemetry and metadata."""
    start_time: datetime = field(default_factory=datetime.now)
    input_path: str = ""
    config: EngineConfig = field(default_factory=EngineConfig)
    target_column: Optional[str] = None
    history: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def record_step(self, step_name: str, metadata: Any):
        self.history[step_name] = metadata

class MLDataEngine:
    """
    Autonomous, institutional-grade data orchestration engine.
    Manages the lifecycle of experimental data from ingestion to model-ready artifacts.
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.guard = IntegrityGuard()
        self.reporter = SpectacularReporter()
        self.scout = DataHealthScout()
        self.detector = AnomalyDetector()
        self.artifact_manager = ArtifactManager(base_dir=self.config.output_dir)
        self.monitor = DriftMonitor()

    def run_pipeline(self, 
                     input_path: str, 
                     target_column: Optional[str] = None,
                     schema: Optional[DataSchema] = None,
                     reference_path: Optional[str] = None) -> pd.DataFrame:
        """
        Executes the autonomous pipeline with real-time telemetry, validation, and drift detection.
        """
        ctx = PipelineContext(input_path=input_path, config=self.config, target_column=target_column)
        self.reporter.welcome_banner()
        
        # 1. Ingestion & Cryptographic Signing
        self.reporter.task_progress(["Ingesting Data", "Generating Cryptographic Hash"])
        df = DataIngestor.load(input_path)
        ctx.record_step("ingestion", {
            "hash": self.guard.get_data_hash(df),
            "shape": df.shape,
            "timestamp": datetime.now().isoformat()
        })
        
        # 2. Intelligence Layer: Health, Anomalies & Drift
        tasks = ["Performing Health Audit", "Scanning for Anomalies"]
        if reference_path:
            tasks.append("Calculating Data Drift")
            
        self.reporter.task_progress(tasks)
        health_report = self.scout.calculate_health_score(df)
        anomaly_report = self.detector.analyze(df)
        
        ctx.record_step("intelligence", {
            "health": health_report,
            "anomalies": anomaly_report
        })
        
        if reference_path:
            ref_df = DataIngestor.load(reference_path)
            drift_report = self.monitor.calculate_drift(ref_df, df, threshold=self.config.drift_threshold)
            ctx.record_step("drift", drift_report)
            if drift_report['drift_detected']:
                self.reporter.console.print(f"[bold yellow]Warning:[/bold yellow] Significant Drift detected in: {drift_report['drifted_columns']}")

        self.reporter.print_health_dashboard(health_report)
        
        if target_column:
            leaky_cols = self.scout.detect_target_leakage(df, target_column)
            if leaky_cols:
                self.reporter.console.print(f"[bold red]SECURITY ALERT:[/bold red] Potential Target Leakage in: {leaky_cols}")
                ctx.record_step("alerts", {"target_leakage": leaky_cols})
                if self.config.stop_on_leakage:
                    raise ValueError(f"Pipeline halted due to target leakage in: {leaky_cols}")

        # 3. Dynamic Schema Inference & Validation
        if not schema:
            schema = AutoGenerator.infer_schema(df)
        
        validation_report = self.guard.validate_schema(df, schema)
        ctx.record_step("validation_initial", validation_report.dict())

        # 4. Autonomous Cleaning & Denoising
        self.reporter.task_progress(["Executing Deduplication", "Adaptive Imputation", "Outlier Clipping"])
        cleaner = AutoCleaner(target_column=target_column)
        df = cleaner.clean(df)
        ctx.record_step("cleaning", cleaner.stats)
        
        # 5. Advanced Feature Engineering
        self.reporter.task_progress(["Temporal Decomposition", "Frequency Encoding", "Structural Optimization"])
        df = FeatureOptimizer.engineer_features(df)
        ctx.record_step("engineering", {"engineered_columns": list(df.columns)})
        
        # 6. Final Integrity Handshake & Persistence
        self.reporter.task_progress(["Final Integrity Verification", "Saving Versioned Artifacts"])
        final_schema = AutoGenerator.infer_schema(df)
        final_report = self.guard.validate_schema(df, final_schema)
        
        ctx.record_step("validation_final", final_report.dict())
        ctx.metrics['processed_hash'] = final_report.data_hash
        ctx.metrics['end_time'] = datetime.now().isoformat()
        
        if self.config.enable_persistence:
            artifact_path = self.artifact_manager.save_run(df, ctx.history, ctx.metrics)
            ctx.metrics['artifact_path'] = artifact_path
        
        self.reporter.finish_summary(ctx.history | ctx.metrics)
        return df

    def get_summary(self, ctx: PipelineContext) -> Dict[str, Any]:
        """Returns the full audit trail for the pipeline run."""
        return ctx.history | ctx.metrics

