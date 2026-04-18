import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from .ingestion import DataIngestor
from .cleaning import AutoCleaner
from .engineering import FeatureOptimizer
from .integrity import IntegrityGuard, DataSchema
from .autogen import AutoGenerator
from .quality import DataHealthScout
from .visualizer import SpectacularReporter

logging.basicConfig(level=logging.ERROR) # Mute standard logging for spectacular output
logger = logging.getLogger("MLDataEngine")

class MLDataEngine:
    """End-to-end automated data engine for ML workflows with premium reporting."""
    
    def __init__(self, output_dir: str = "./processed_data"):
        self.output_dir = output_dir
        self.guard = IntegrityGuard()
        self.reporter = SpectacularReporter()
        self.scout = DataHealthScout()
        self.history: Dict[str, Any] = {}

    def run_pipeline(self, 
                     input_path: str, 
                     target_column: Optional[str] = None,
                     schema: Optional[DataSchema] = None) -> pd.DataFrame:
        """
        Executes full end-to-end pipeline with spectacular visual feedback.
        """
        self.reporter.welcome_banner()
        self.history['start_time'] = datetime.now().isoformat()
        
        # 1. Ingestion
        self.reporter.task_progress(["Ingesting Data", "Calculating Initial Hash"])
        df = DataIngestor.load(input_path)
        self.history['input_hash'] = self.guard.get_data_hash(df)
        self.history['raw_shape'] = df.shape
        
        # 2. Advanced Health Check
        self.reporter.task_progress(["Analyzing Data Health", "Checking for Target Leakage"])
        health_report = self.scout.calculate_health_score(df)
        self.history['initial_health'] = health_report
        self.reporter.print_health_dashboard(health_report)
        
        if target_column:
            leaky_cols = self.scout.detect_target_leakage(df, target_column)
            if leaky_cols:
                self.reporter.console.print(f"[bold red]⚠ CRITICAL:[/bold red] Detected potential target leakage in: {leaky_cols}")
                self.history['target_leakage'] = leaky_cols

        # 3. Schema Validation
        if not schema:
            schema = AutoGenerator.infer_schema(df)
        
        report = self.guard.validate_schema(df, schema)
        self.history['initial_report'] = report.dict()

        # 4. Automated Cleaning
        self.reporter.task_progress(["Applying Deduplication", "Standardizing Missing Values", "Clipping Outliers"])
        cleaner = AutoCleaner(target_column=target_column)
        df = cleaner.clean(df)
        self.history['cleaning_stats'] = cleaner.stats
        
        # 5. Feature Engineering
        self.reporter.task_progress(["Decomposing Temporal Features", "Frequency Encoding Categoricals"])
        df = FeatureOptimizer.engineer_features(df)
        self.history['engineered_columns'] = list(df.columns)
        
        # 6. Final Validation & Summary
        self.reporter.task_progress(["Final Integrity Verification", "Generating Audit Trail"])
        final_schema = AutoGenerator.infer_schema(df)
        final_report = self.guard.validate_schema(df, final_schema)
        self.history['final_report'] = final_report.dict()
        self.history['processed_hash'] = final_report.data_hash
        self.history['end_time'] = datetime.now().isoformat()
        
        self.reporter.finish_summary(self.history)
        return df

    def get_summary(self) -> Dict[str, Any]:
        """Returns the processing history for audit/reproducibility."""
        return self.history
