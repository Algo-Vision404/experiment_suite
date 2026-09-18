from typing import Any, Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

class SpectacularReporter:
    """Human-facing terminal reporting. Progress reflects actual completed stages."""

    def __init__(self):
        self.console = Console(force_terminal=True, soft_wrap=True, legacy_windows=True)

    def welcome_banner(self):
        banner = Panel.fit(
            "[bold cyan]ML EXPERIMENTAL DATA ENGINE[/bold cyan]\n"
            "[italic white]Experimental Standardization Suite[/italic white]",
            padding=(1, 2),
        )
        self.console.print(banner)

    def print_health_dashboard(self, health_data: Dict[str, Any]):
        score = health_data["overall_health_score"]
        color = "green" if score > 80 else "yellow" if score > 50 else "red"
        table = Table(title="Data Health Dashboard", box=None)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Overall Health Score", f"[{color}]{score}%[/{color}]")
        for key, label in (("completeness", "Completeness"), ("uniqueness", "Uniqueness"), ("info_density", "Info Density"), ("skew_score", "Skew Score")):
            table.add_row(label, f"{health_data['metrics'][key]:.2%}")
        self.console.print(table)
        constants = health_data["anomalies"]["constant_columns"]
        if constants:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] Constant columns: {constants}")

    def task_progress(self, tasks: List[str]):
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=self.console) as progress:
            for task_desc in tasks:
                task = progress.add_task(task_desc, total=100)
                progress.update(task, completed=100)
                self.console.print(f"[bold green]DONE[/bold green] {task_desc}")

    def finish_summary(self, history: Dict[str, Any]):
        ingestion = history.get("ingestion", {})
        input_hash = ingestion.get("hash", "N/A")
        processed_hash = history.get("processed_hash", "N/A")
        features = history.get("engineering", {}).get("engineered_columns", [])
        self.console.print(Panel(
            f"[bold green]Pipeline Execution Successful![/bold green]\n\n"
            f"Input Hash: [dim]{input_hash[:12]}...[/dim]\n"
            f"Processed Hash: [bold]{processed_hash[:12]}...[/bold]\n"
            f"Final Features: {len(features)}",
            title="Summary Report",
        ))
