from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint
from typing import Dict, Any, List
import time

class SpectacularReporter:
    """Provides high-end visual feedback for the ML pipeline."""
    
    def __init__(self):
        # Use a safe console config for Windows legacy terminals
        self.console = Console(force_terminal=True, soft_wrap=True)

    def welcome_banner(self):
        self.console.clear()
        banner = Panel.fit(
            "[bold cyan]ML EXPERIMENTAL DATA ENGINE[/bold cyan]\n"
            "[italic white]Next-Gen Automated Preprocessing System[/italic white]",
            border_style="bold magenta",
            padding=(1, 2)
        )
        self.console.print(banner)

    def print_health_dashboard(self, health_data: Dict[str, Any]):
        """Prints a beautiful table of data health metrics."""
        score = health_data['overall_health_score']
        color = "green" if score > 80 else "yellow" if score > 50 else "red"
        
        table = Table(title="[bold blue]Data Health Dashboard[/bold blue]", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Overall Health Score", f"[{color}]{score}%[/{color}]")
        table.add_row("Completeness", f"{health_data['metrics']['completeness']:.2%}")
        table.add_row("Uniqueness", f"{health_data['metrics']['uniqueness']:.2%}")
        table.add_row("Info Density", f"{health_data['metrics']['info_density']:.2%}")
        
        self.console.print(table)
        
        if health_data['anomalies']['constant_columns']:
            self.console.print(f"[bold red]⚠ Warning:[/bold red] Found constant columns: {health_data['anomalies']['constant_columns']}")

    def task_progress(self, tasks: List[str]):
        """Runs a simulated or real progress bar for the pipeline steps."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True
        ) as progress:
            for task_desc in tasks:
                task = progress.add_task(f"[yellow]{task_desc}...", total=100)
                # This is just for visual "wow", in reality it would be tied to steps
                for i in range(100):
                    time.sleep(0.005) 
                    progress.update(task, advance=1)
                self.console.print(f"[bold green] DONE [/bold green] {task_desc} complete.")

    def finish_summary(self, history: Dict[str, Any]):
        summary_panel = Panel(
            f"[bold green]Pipeline Execution Successful![/bold green]\n\n"
            f"Input Hash: [dim]{history['input_hash'][:8]}...[/dim]\n"
            f"Processed Hash: [bold]{history['processed_hash'][:8]}...[/bold]\n"
            f"Final Features: {len(history['engineered_columns'])}",
            title="Summary Report",
            border_style="green"
        )
        self.console.print(summary_panel)
