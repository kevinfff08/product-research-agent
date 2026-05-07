"""CLI entry point using Typer + Rich."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Iterable

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.logging_config import current_log_path, setup_logging
from src.models.input import ResearchRequest
from src.orchestrator import Orchestrator
from src.utils.naming import build_run_name

app = typer.Typer(
    name="product-research",
    help="AI-powered product & technology landscape research agent.",
    no_args_is_help=True,
)
console = Console()


def _parse_focus_areas(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_research_request(
    request: ResearchRequest,
    *,
    config: str,
    output_dir: str,
    data_dir: str,
    log_dir: str,
) -> None:
    """Configure logging, run the pipeline, and print output locations."""
    if not request.run_name:
        request.run_name = build_run_name(request.title)
    log_path = setup_logging(run_name=request.run_name, log_dir=log_dir, force=True)

    console.print(Panel(
        f"[bold]Title:[/bold] {request.title}\n"
        f"[bold]Run name:[/bold] {request.run_name}",
        title="Product Research Agent",
        border_style="blue",
    ))

    orchestrator = Orchestrator(
        config_path=config,
        data_dir=data_dir,
        output_dir=output_dir,
    )

    with console.status("[bold green]Running research pipeline..."):
        report = asyncio.run(orchestrator.run(request))

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

    output_paths = orchestrator.output_paths or [
        Path(output_dir) / report.session_id / f"{report.session_id}.md"
    ]
    console.print("\nReports saved to:")
    for output_path in output_paths:
        console.print(f"  [bold]{output_path}[/bold]")
    console.print(f"\nLog saved to: [bold]{log_path or current_log_path()}[/bold]")


def _print_field_guide(
    step: str,
    *,
    required: bool,
    fmt: str,
    example: str,
) -> None:
    label = "必填" if required else "选填"
    console.print()
    console.print(Panel(
        f"[bold]{step}（{label}）[/bold]\n"
        f"格式：{fmt}\n"
        f"示例：{example}",
        border_style="cyan",
    ))


def _prompt_required_text(step: str, *, fmt: str, example: str) -> str:
    _print_field_guide(step, required=True, fmt=fmt, example=example)
    while True:
        value = typer.prompt("请输入").strip()
        if value:
            return value
        console.print("[red]该字段为必填，请重新输入。[/red]")


def _prompt_multiline_required(step: str, *, fmt: str, example: str) -> str:
    _print_field_guide(step, required=True, fmt=fmt, example=example)
    console.print("可以输入多段文字。输入单独一行 [bold]END[/bold] 结束。")
    while True:
        lines: list[str] = []
        while True:
            line = console.input("> ")
            if line.strip().upper() == "END":
                break
            lines.append(line)
        value = "\n".join(lines).strip()
        if value:
            return value
        console.print("[red]详细描述为必填，请至少输入一行内容，然后用 END 结束。[/red]")


def _prompt_optional_text(
    step: str,
    *,
    fmt: str,
    example: str,
    default: str = "",
) -> str:
    _print_field_guide(step, required=False, fmt=fmt, example=example)
    value = typer.prompt("请输入，留空使用默认值", default=default, show_default=True)
    return value.strip()


def _prompt_choice(
    step: str,
    *,
    choices: Iterable[str],
    default: str,
    example: str,
) -> str:
    valid = {choice.lower() for choice in choices}
    fmt = " / ".join(choices)
    while True:
        value = _prompt_optional_text(step, fmt=fmt, example=example, default=default).lower()
        if value in valid:
            return value
        console.print(f"[red]请输入以下选项之一：{fmt}[/red]")


def _prompt_int(
    step: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
    example: str,
) -> int:
    while True:
        raw = _prompt_optional_text(
            step,
            fmt=f"{minimum}-{maximum} 的整数",
            example=example,
            default=str(default),
        )
        try:
            value = int(raw)
        except ValueError:
            console.print("[red]请输入整数。[/red]")
            continue
        if minimum <= value <= maximum:
            return value
        console.print(f"[red]请输入 {minimum}-{maximum} 范围内的整数。[/red]")


@app.command()
def start(
    config: str = typer.Option(
        "config/default.yaml", "--config", "-c",
        help="Path to configuration file",
    ),
    output_dir: str = typer.Option(
        "output", "--output-dir", "-o",
        help="Output directory for reports",
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir",
        help="Data directory for intermediate session files",
    ),
    log_dir: str = typer.Option(
        "logs", "--log-dir",
        help="Log directory",
    ),
) -> None:
    """Start the guided multi-stage research input flow."""
    console.print(Panel(
        "已启动交互式调研向导。接下来会分阶段收集标题、详细描述和运行参数。",
        title="Start",
        border_style="green",
    ))

    title = _prompt_required_text(
        "1. 标题式输入",
        fmt="一句短标题，建议 5-30 个字；不要写成长段说明",
        example="实时视频翻译工具",
    )
    detailed_description = _prompt_multiline_required(
        "2. 二阶段细致描述",
        fmt="多段自然语言；写清目标用户、核心功能、约束、已有想法和你关心的问题",
        example="目标用户是跨国会议团队。希望支持实时字幕、语音克隆、低延迟和会议软件集成。END",
    )
    focus_raw = _prompt_optional_text(
        "3. 关注重点",
        fmt="逗号分隔；留空表示由系统自行判断",
        example="低延迟, 开源实现, 学术评测",
    )
    depth = _prompt_choice(
        "4. 调研深度",
        choices=("quick", "comprehensive", "deep"),
        default="comprehensive",
        example="comprehensive",
    )
    max_paths = _prompt_int(
        "5. 最大研究路径数",
        default=5,
        minimum=1,
        maximum=10,
        example="3",
    )
    output_format = _prompt_choice(
        "6. 输出格式",
        choices=("markdown", "docx", "both"),
        default="markdown",
        example="both",
    )

    run_name = build_run_name(title)
    request = ResearchRequest(
        title=title,
        detailed_description=detailed_description,
        focus_areas=_parse_focus_areas(focus_raw),
        depth=depth,
        max_paths=max_paths,
        output_format=output_format,
        run_name=run_name,
    )
    _run_research_request(
        request,
        config=config,
        output_dir=output_dir,
        data_dir=data_dir,
        log_dir=log_dir,
    )


@app.command()
def research(
    title: str = typer.Argument(help="Short title-style product idea or research topic"),
    detailed_description: str = typer.Option(
        "",
        "--description",
        "-D",
        help="Detailed second-stage description. If omitted, the title is used as compact context.",
    ),
    focus: str = typer.Option(
        "",
        "--focus",
        help="Optional comma-separated focus areas",
    ),
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
    data_dir: str = typer.Option(
        "data", "--data-dir",
        help="Data directory for intermediate session files",
    ),
    log_dir: str = typer.Option(
        "logs", "--log-dir",
        help="Log directory",
    ),
) -> None:
    """Run research non-interactively with a title and optional detailed description."""
    request = ResearchRequest(
        title=title,
        detailed_description=detailed_description or title,
        focus_areas=_parse_focus_areas(focus),
        depth=depth,
        max_paths=max_paths,
        output_format=output_format,
        run_name=build_run_name(title),
    )
    _run_research_request(
        request,
        config=config,
        output_dir=output_dir,
        data_dir=data_dir,
        log_dir=log_dir,
    )


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
