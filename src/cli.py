"""CLI entry point using Typer + Rich."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Iterable

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.logging_config import current_log_path, setup_logging
from src.models.input import ResearchRequest
from src.models.plan import DecompositionResult
from src.models.report import ResearchReport
from src.orchestrator import Orchestrator
from src.utils.naming import build_run_name

app = typer.Typer(
    name="product-research",
    help="AI-powered product & technology landscape research agent.",
    no_args_is_help=True,
)
console = Console()


class PipelineTerminalProgress:
    """User-facing terminal progress for one pipeline run."""

    def __init__(self) -> None:
        self._task_id = None
        self._seen_stages: set[str] = set()
        self._path_statuses: dict[str, str] = {}
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )

    def __enter__(self) -> "PipelineTerminalProgress":
        self._progress.start()
        self._task_id = self._progress.add_task("准备启动调研流水线...", total=8, completed=0)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                description="[red]调研流水线异常退出[/red]",
            )
        self._progress.stop()

    def __call__(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "pipeline_started":
            self._update_total(int(event.get("total_steps") or 8))
            self._print_info(event.get("message") or "开始调研流水线")
        elif kind == "pipeline_stage":
            self._handle_stage(event)
        elif kind == "path_status":
            self._handle_path_status(event)
        elif kind == "path_subagent_status":
            self._handle_path_subagent_status(event)
        elif kind == "pipeline_completed":
            self._handle_completed(event)
        elif kind == "pipeline_failed":
            message = event.get("message") or "调研流水线失败"
            self._print_error(str(message))

    def _update_total(self, total: int) -> None:
        if self._task_id is not None:
            self._progress.update(self._task_id, total=total)

    def _handle_stage(self, event: dict[str, Any]) -> None:
        status = str(event.get("status") or "")
        message = str(event.get("message") or status)
        step = int(event.get("step") or 0)
        total = int(event.get("total_steps") or 8)
        description = f"[{step}/{total}] {message}" if step else message

        if self._task_id is not None:
            completed = max(step - 1, 0) if step else 0
            self._progress.update(
                self._task_id,
                total=total,
                completed=completed,
                description=description,
            )

        if status not in self._seen_stages:
            prefix = f"[cyan][{step}/{total}][/cyan] " if step else ""
            self._progress.console.print(f"{prefix}{message}")
            self._seen_stages.add(status)

    def _handle_path_status(self, event: dict[str, Any]) -> None:
        path_id = str(event.get("path_id") or "?")
        status = str(event.get("status") or "unknown")
        previous = self._path_statuses.get(path_id)
        if previous == status:
            return
        self._path_statuses[path_id] = status

        title = str(event.get("path_title") or path_id)
        status_label = {
            "running": "开始",
            "completed": "完成",
            "partial_failed": "部分失败",
            "failed": "失败",
        }.get(status, status)
        counts = event.get("query_counts") or {}
        count_text = ""
        if isinstance(counts, dict) and counts:
            count_text = "；查询数 " + ", ".join(
                f"{source}:{count}" for source, count in sorted(counts.items())
            )
        style = "green" if status == "completed" else "yellow"
        if status in {"failed", "partial_failed"}:
            style = "red"
        self._progress.console.print(
            f"  [{style}]Path {path_id} {status_label}[/{style}] - {title}{count_text}"
        )

    def _handle_path_subagent_status(self, event: dict[str, Any]) -> None:
        path_id = str(event.get("path_id") or "?")
        agent_name = str(event.get("agent_name") or event.get("agent") or "子任务")
        status = str(event.get("status") or "unknown")
        status_label = {
            "running": "开始",
            "completed": "完成",
            "failed": "失败",
        }.get(status, status)
        style = "green" if status == "completed" else "dim"
        if status == "failed":
            style = "red"
        error = event.get("error")
        suffix = f"；{error}" if error else ""
        self._progress.console.print(
            f"    [{style}]Path {path_id} / {agent_name} {status_label}[/{style}]{suffix}"
        )

    def _handle_completed(self, event: dict[str, Any]) -> None:
        total = int(event.get("total_steps") or 8)
        if self._task_id is not None:
            self._progress.update(
                self._task_id,
                total=total,
                completed=total,
                description="[green]调研流水线完成[/green]",
            )
        self._print_info(event.get("message") or "调研流水线完成")

    def _print_info(self, message: str) -> None:
        self._progress.console.print(f"[dim]{message}[/dim]")

    def _print_error(self, message: str) -> None:
        self._progress.console.print(f"[red]{message}[/red]")


def _parse_focus_areas(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_run_header(request: ResearchRequest, *, title: str = "Product Research Agent") -> None:
    console.print(Panel(
        f"[bold]Title:[/bold] {request.title}\n"
        f"[bold]Run name:[/bold] {request.run_name}\n"
        f"[bold]Research paths:[/bold] {request.max_paths}",
        title=title,
        border_style="blue",
    ))


def _run_orchestrator_once(
    request: ResearchRequest,
    *,
    config: str,
    output_dir: str,
    data_dir: str,
) -> tuple[ResearchReport, Orchestrator]:
    """Run one pipeline execution and return the report plus orchestrator state."""
    with PipelineTerminalProgress() as progress:
        orchestrator = Orchestrator(
            config_path=config,
            data_dir=data_dir,
            output_dir=output_dir,
            progress_callback=progress,
        )
        report = asyncio.run(orchestrator.run(request))
    return report, orchestrator


def _print_run_results(
    report: ResearchReport,
    orchestrator: Orchestrator,
    *,
    output_dir: str,
    log_path: Path | str | None,
) -> None:
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


def _propose_paths(
    request: ResearchRequest,
    *,
    first_report: ResearchReport,
    config: str,
    output_dir: str,
    data_dir: str,
) -> DecompositionResult:
    """Build a candidate multi-path plan after an initial single-path run."""
    proposal_input = (
        f"{request.raw_input}\n\n"
        "Initial single-path research summary:\n"
        f"{first_report.executive_summary[:2000]}"
    )
    orchestrator = Orchestrator(
        config_path=config,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    try:
        return orchestrator.propose_paths(
            raw_input=proposal_input,
            max_paths=request.max_paths,
        )
    finally:
        asyncio.run(orchestrator._cleanup())


def _confirm_multi_path_plan(decomposition: DecompositionResult) -> bool:
    paths = decomposition.paths
    if len(paths) <= 1:
        console.print("[yellow]初轮调研后没有发现明确需要拆分的多条互斥路线，保持单路径结果。[/yellow]")
        return False

    table = Table(title="候选分路径调研计划")
    table.add_column("编号", style="cyan", no_wrap=True)
    table.add_column("技术路线")
    table.add_column("简要解释")
    for idx, path in enumerate(paths, start=1):
        table.add_row(
            str(idx),
            path.title,
            path.description[:180],
        )
    console.print()
    console.print(table)
    return typer.confirm("是否确认继续执行以上分路径调研？", default=False)


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
    base_run_name = request.run_name
    log_path = setup_logging(run_name=base_run_name, log_dir=log_dir, force=True)

    if request.max_paths <= 1:
        _print_run_header(request)
        report, orchestrator = _run_orchestrator_once(
            request,
            config=config,
            output_dir=output_dir,
            data_dir=data_dir,
        )
        _print_run_results(report, orchestrator, output_dir=output_dir, log_path=log_path)
        return

    first_request = request.model_copy(
        update={
            "max_paths": 1,
            "run_name": f"{base_run_name}_single",
        }
    )
    _print_run_header(first_request, title="Initial Single-Path Research")
    first_report, first_orchestrator = _run_orchestrator_once(
        first_request,
        config=config,
        output_dir=output_dir,
        data_dir=data_dir,
    )
    _print_run_results(
        first_report,
        first_orchestrator,
        output_dir=output_dir,
        log_path=log_path,
    )

    console.print()
    console.print("[bold]正在基于初轮调研生成候选分路径计划...[/bold]")
    decomposition = _propose_paths(
        request,
        first_report=first_report,
        config=config,
        output_dir=output_dir,
        data_dir=data_dir,
    )
    if not _confirm_multi_path_plan(decomposition):
        return

    multi_request = request.model_copy(update={"run_name": f"{base_run_name}_multipath"})
    _print_run_header(multi_request, title="Confirmed Multi-Path Research")
    report, orchestrator = _run_orchestrator_once(
        multi_request,
        config=config,
        output_dir=output_dir,
        data_dir=data_dir,
    )
    _print_run_results(report, orchestrator, output_dir=output_dir, log_path=log_path)


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
    console.print()
    wants_multi_path = typer.confirm(
        "5. 是否在初轮单路径调研后，再评估并确认分路径调研计划？默认否",
        default=False,
    )
    max_paths = 1
    if wants_multi_path:
        max_paths = _prompt_int(
            "5a. 候选分路径数量上限",
            default=3,
            minimum=2,
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
        1, "--max-paths", "-p",
        help=(
            "Maximum research paths to explore (1-10). Defaults to 1. "
            "Values >1 run a single-path pass first, then ask for confirmation."
        ),
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
