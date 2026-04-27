"""
命令行接口模块

提供以下命令：
- status: 显示项目状态和数据集信息
- inspect-task: 查看任务详情和上下文文件
- run-task: 运行单个任务
- run-benchmark: 批量运行任务

使用 typer 构建命令行界面，rich 提供美化的输出和进度条。
"""

from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.tools.filesystem import list_context_tree

from custom_agent.runner import run_single_task, run_benchmark, TaskRunArtifacts, create_run_output_dir

# 项目路径常量
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"

# typer 应用实例
app = typer.Typer(add_completion=False, no_args_is_help=False)
# rich 控制台实例
console = Console()


def _status_value(path: Path) -> str:
    """检查路径是否存在，返回状态字符串"""
    return "present" if path.exists() else "missing"


def _format_rate(count: int, elapsed: float) -> str:
    """
    格式化任务完成速率
    
    Args:
        count: 完成的任务数
        elapsed: 经过的秒数
    
    Returns:
        格式化的速率字符串，如 "rate=12.5 task/min"
    """
    if count <= 0 or elapsed <= 0:
        return "rate=0.0 task/min"
    return f"rate={(count / elapsed) * 60:.1f} task/min"


def _format_last(artifact: TaskRunArtifacts | None) -> str:
    """
    格式化最近完成的任务信息
    
    Args:
        artifact: 最近的任务产物，可选
    
    Returns:
        格式化的字符串，如 "last=task_11 (ok)"
    """
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_progress_fields(
    *,
    completed: int,
    succeeded: int,
    failed: int,
    total: int,
    workers: int,
    elapsed: float,
    last: TaskRunArtifacts | None,
) -> dict[str, str]:
    """
    构建进度条字段
    
    用于在进度条中显示详细统计信息。
    
    Args:
        completed: 已完成任务数
        succeeded: 成功任务数
        failed: 失败任务数
        total: 总任务数
        workers: 工作线程数
        elapsed: 经过的秒数
        last: 最近完成的任务产物
    
    Returns:
        包含各字段值的字典
    """
    remaining = max(total - completed, 0)
    running = min(workers, remaining)
    queued = max(remaining - running, 0)
    return {
        "ok": str(succeeded),
        "fail": str(failed),
        "run": str(running),
        "queue": str(queued),
        "speed": _format_rate(completed, elapsed),
        "last": _format_last(last),
    }


@app.callback()
def cli():
    """命令行入口回调"""
    pass


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
):
    """
    显示项目状态
    
    列出项目路径、数据集状态、任务数量等信息。
    
    Args:
        config: YAML 配置文件路径
    """
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    
    # 构建状态表格
    table = Table(title="Custom Agent Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")
    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row("dataset_root", str(app_config.dataset.root_path), _status_value(app_config.dataset.root_path))
    console.print(table)
    
    # 显示任务统计
    if dataset.exists:
        console.print(f"Public tasks: {len(dataset.list_task_ids())}")
        counts = dataset.task_counts()
        if counts:
            rendered = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            console.print(f"Task counts: {rendered}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
):
    """
    查看任务详情
    
    显示任务的问题、难度和上下文文件列表。
    
    Args:
        task_id: 任务 ID
        config: YAML 配置文件路径
    """
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    
    # 显示任务信息
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    
    # 显示上下文文件
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


@app.command("run-task")
def run_task_cmd(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
):
    """
    运行单个任务
    
    执行指定任务并输出结果文件。
    
    Args:
        task_id: 任务 ID
        config: YAML 配置文件路径
    """
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(app_config.run.output_dir, run_id=app_config.run.run_id)
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    
    artifact = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)
    
    # 显示结果
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifact.task_output_dir}")
    if artifact.prediction_csv_path:
        console.print(f"Prediction CSV: {artifact.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifact.failure_reason:
        console.print(f"Failure: {artifact.failure_reason}")


@app.command("run-benchmark")
def run_benchmark_cmd(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum tasks to run."),
):
    """
    批量运行任务
    
    执行多个任务并显示实时进度条。
    
    Args:
        config: YAML 配置文件路径
        limit: 最大任务数限制，可选
    """
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    total = len(dataset.iter_tasks())
    if limit is not None:
        total = min(total, limit)
    workers = app_config.run.max_workers

    # 配置进度条列
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    
    with Progress(*columns, console=console) as progress:
        pid = progress.add_task(
            "Benchmark",
            total=total,
            completed=0,
            **_build_progress_fields(completed=0, succeeded=0, failed=0, total=total, workers=workers, elapsed=0.0, last=None),
        )
        completed = succeeded = failed = 0
        start = perf_counter()

        def on_complete(artifact):
            """任务完成回调：更新进度条"""
            nonlocal completed, succeeded, failed
            completed += 1
            if artifact.succeeded:
                succeeded += 1
            else:
                failed += 1
            progress.update(
                pid,
                completed=completed,
                refresh=True,
                **_build_progress_fields(
                    completed=completed,
                    succeeded=succeeded,
                    failed=failed,
                    total=total,
                    workers=workers,
                    elapsed=perf_counter() - start,
                    last=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark(config=app_config, limit=limit, progress_callback=on_complete)
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    
    # 显示汇总结果
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded: {sum(1 for a in artifacts if a.succeeded)}")


def main():
    """命令行入口函数"""
    app()
