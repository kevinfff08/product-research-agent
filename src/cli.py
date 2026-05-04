"""CLI entry point using Typer + Rich."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.models.input import ResearchRequest
from src.orchestrator import Orchestrator

app = typer.Typer(
    name="product-research",
    help="AI-powered product & technology landscape research agent.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def research(
    idea: str = typer.Argument(help="Product idea or concept to research"),
    depth: str = typer.Option(
        "comprehensive", "--depth", "-d",
        help="Research depth: quick, comprehensive, deep",
    ),
    max_paths: int = typer.Option(
        5, "--max-paths", "-p",
        help="Maximum research paths to explore (1-10)",
        min=1, max=10,
    ),
    output_format: str = typer.Option(
        "markdown", "--format", "-f",
        help="Output format: markdown, docx, both",
    ),
    config: str = typer.Option(
        "config/default.yaml", "--config", "-c",
        help="Path to configuration file",
    ),
    output_dir: str = typer.Option(
        "output", "--output-dir", "-o",
        help="Output directory for reports",
    ),
) -> None:
    """Run a full technology landscape research for a product idea."""
    console.print(Panel(
        f"[bold]Researching:[/bold] {idea}",
        title="Product Research Agent",
        border_style="blue",
    ))

    request = ResearchRequest(
        raw_input=idea,
        depth=depth,
        max_paths=max_paths,
        output_format=output_format,
    )

    orchestrator = Orchestrator(
        config_path=config,
        output_dir=output_dir,
    )

    with console.status("[bold green]Running research pipeline..."):
        report = asyncio.run(orchestrator.run(request))

    # Display summary
    console.print()
    console.print(Panel(
        f"[bold green]Research Complete![/bold green]\n\n"
        f"Session: {report.session_id}\n"
        f"Technologies found: {len(report.technology_landscape)}\n"
        f"Workflows generated: {len(report.workflows)}\n"
        f"Sources used: {len(report.all_sources)}",
        title="Results",
        border_style="green",
    ))

    output_paths = orchestrator.output_paths or [Path(output_dir) / f"{report.session_id}.md"]
    console.print("\nReports saved to:")
    for output_path in output_paths:
        console.print(f"  [bold]{output_path}[/bold]")


@app.command()
def list_sessions(
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
) -> None:
    """List all research sessions."""
    from src.storage.local_store import LocalStore
    store = LocalStore(data_dir)
    sessions = store.list_sessions()

    if not sessions:
        console.print("[yellow]No research sessions found.[/yellow]")
        return

    table = Table(title="Research Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Created", style="dim")
    table.add_column("Description")

    for s in sessions:
        table.add_row(
            s.get("session_id", "?"),
            s.get("status", "unknown"),
            s.get("created_at", "")[:19],
            s.get("description", "")[:60],
        )

    console.print(table)


@app.command()
def show(
    session_id: str = typer.Argument(help="Session ID to display"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
) -> None:
    """Show details of a specific research session."""
    from src.storage.local_store import LocalStore
    store = LocalStore(data_dir)

    meta = store.load_json(f"research/{session_id}/meta.json")
    if not meta:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        json.dumps(meta, indent=2),
        title=f"Session: {session_id}",
        border_style="blue",
    ))

    # Check for report
    report_data = store.load_json(f"research/{session_id}/report.json")
    if report_data and isinstance(report_data, dict):
        console.print(f"\n[bold]Executive Summary:[/bold]")
        console.print(report_data.get("executive_summary", "N/A")[:500])

        techs = report_data.get("technology_landscape", [])
        if techs:
            table = Table(title="Technology Landscape")
            table.add_column("Technology")
            table.add_column("Category")
            table.add_column("Maturity")
            for t in techs[:15]:
                table.add_row(t.get("name", ""), t.get("category", ""), t.get("maturity", ""))
            console.print(table)


@app.command()
def status(
    session_id: str = typer.Argument(help="Session ID to check"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
) -> None:
    """Check the status of a research session."""
    from src.storage.local_store import LocalStore
    store = LocalStore(data_dir)
    meta = store.load_json(f"research/{session_id}/meta.json")
    if not meta:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        raise typer.Exit(1)
    status_val = meta.get("status", "unknown")
    color = {"completed": "green", "failed": "red", "started": "yellow"}.get(status_val, "white")
    console.print(f"Session [cyan]{session_id}[/cyan]: [{color}]{status_val}[/{color}]")
