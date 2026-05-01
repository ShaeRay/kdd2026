"""
任务运行器模块

负责执行单个任务和批量任务，并输出结果文件：
- trace.json: 完整的执行步骤记录
- prediction.csv: 预测结果表格
- summary.json: 批量运行的汇总信息

支持：
- 单线程顺序执行（复用模型实例）
- 多线程并发执行（每个任务独立创建模型）
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig

from custom_agent.agent import CustomAgent, CustomAgentConfig


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    """
    任务运行产物
    
    记录单个任务的执行结果和输出文件路径。
    
    Attributes:
        task_id: 任务 ID
        task_output_dir: 任务输出目录
        prediction_csv_path: 预测 CSV 文件路径（如果成功）
        trace_path: 执行追踪 JSON 文件路径
        succeeded: 是否成功
        failure_reason: 失败原因（如果失败）
    """
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，用于 JSON 序列化"""
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    """
    创建运行 ID
    
    格式：YYYYMMDDTHHMMSSZ（UTC 时间）
    
    Returns:
        时间戳格式的运行 ID
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    """
    解析运行 ID
    
    如果未提供，自动生成时间戳 ID。
    验证 ID 不包含路径分隔符，防止路径注入。
    
    Args:
        run_id: 用户提供的运行 ID，可选
    
    Returns:
        有效的运行 ID
    
    Raises:
        ValueError: 运行 ID 无效
    """
    if run_id is None:
        return create_run_id()
    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    # 防止路径注入
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    """
    创建运行输出目录
    
    Args:
        output_root: 输出根目录
        run_id: 运行 ID，可选
    
    Returns:
        (运行 ID, 输出目录路径) 元组
    
    Raises:
        FileExistsError: 目录已存在
    """
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig) -> OpenAIModelAdapter:
    """
    构建模型适配器
    
    Args:
        config: 应用配置
    
    Returns:
        OpenAI 兼容的模型适配器
    """
    model_name = os.environ.get("MODEL_NAME") or config.agent.model
    api_base = os.environ.get("MODEL_API_URL") or config.agent.api_base
    api_key = os.environ.get("MODEL_API_KEY") or config.agent.api_key

    return OpenAIModelAdapter(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=config.agent.temperature,
    )


def _write_json(path: Path, payload: dict) -> None:
    """写入 JSON 文件"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """写入 CSV 文件"""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _run_single_task_core(*, task_id: str, config: AppConfig, model=None) -> dict:
    """
    执行单个任务的核心逻辑
    
    Args:
        task_id: 任务 ID
        config: 应用配置
        model: 可选的模型实例（用于复用）
    
    Returns:
        任务执行结果字典
    """
    dataset = DABenchPublicDataset(config.dataset.root_path)
    task = dataset.get_task(task_id)
    agent = CustomAgent(
        model=model or build_model_adapter(config),
        config=CustomAgentConfig(max_steps=config.agent.max_steps),
    )
    result = agent.run(task)
    return result.to_dict()


def _write_task_outputs(task_id: str, run_output_dir: Path, result: dict) -> TaskRunArtifacts:
    """
    写入任务输出文件
    
    输出文件：
    - trace.json: 完整执行记录
    - prediction.csv: 预测结果（如果成功）
    
    Args:
        task_id: 任务 ID
        run_output_dir: 运行输出目录
        result: 任务执行结果
    
    Returns:
        TaskRunArtifacts 对象
    """
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入执行追踪
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, result)
    
    # 写入预测结果
    prediction_csv_path: Path | None = None
    answer = result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(prediction_csv_path, list(answer.get("columns", [])), [list(r) for r in answer.get("rows", [])])
    
    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(result.get("succeeded")),
        failure_reason=result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
) -> TaskRunArtifacts:
    """
    运行单个任务
    
    执行任务并输出结果文件。
    
    Args:
        task_id: 任务 ID
        config: 应用配置
        run_output_dir: 运行输出目录
        model: 可选的模型实例（用于复用）
    
    Returns:
        TaskRunArtifacts 对象
    """
    started = perf_counter()
    result = _run_single_task_core(task_id=task_id, config=config, model=model)
    result["e2e_elapsed_seconds"] = round(perf_counter() - started, 3)
    return _write_task_outputs(task_id, run_output_dir, result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    limit: int | None = None,
    progress_callback=None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    """
    运行基准测试（批量任务）
    
    支持两种执行模式：
    1. 单线程：复用模型实例，减少 API 调用开销
    2. 多线程：每个任务独立执行，提高吞吐量
    
    输出文件：
    - summary.json: 汇总信息
    - 每个任务目录下的 trace.json 和 prediction.csv
    
    Args:
        config: 应用配置
        model: 可选的模型实例（用于复用）
        limit: 最大任务数限制，可选
        progress_callback: 进度回调函数，可选
    
    Returns:
        (输出目录路径, TaskRunArtifacts 列表) 元组
    """
    # 创建运行目录
    run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)
    
    # 加载任务列表
    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]
    task_ids = [t.task_id for t in tasks]
    
    # 确定并发模式
    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None:
        effective_workers = 1  # 如果提供了模型实例，强制单线程

    artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        # 单线程模式：复用模型实例
        shared_model = model or build_model_adapter(config)
        artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(task_id=task_id, config=config, run_output_dir=run_output_dir, model=shared_model)
            artifacts.append(artifact)
            if progress_callback:
                progress_callback(artifact)
    else:
        # 多线程模式：每个任务独立执行
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_idx = {
                executor.submit(run_single_task, task_id=tid, config=config, run_output_dir=run_output_dir): i
                for i, tid in enumerate(task_ids)
            }
            indexed: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_idx):
                artifact = future.result()
                indexed[future_to_idx[future]] = artifact
                if progress_callback:
                    progress_callback(artifact)
            artifacts = [a for a in indexed if a is not None]

    # 写入汇总文件
    summary_path = run_output_dir / "summary.json"
    _write_json(summary_path, {
        "run_id": run_id,
        "task_count": len(artifacts),
        "succeeded_task_count": sum(1 for a in artifacts if a.succeeded),
        "max_workers": effective_workers,
        "tasks": [a.to_dict() for a in artifacts],
    })
    return run_output_dir, artifacts
