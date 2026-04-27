from pathlib import Path

import typer
from rich.console import Console

from custom_agent.agent import CustomAgent, CustomAgentConfig
from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.runner import create_run_output_dir, run_single_task
from data_agent_baseline.tools.registry import create_default_tool_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


@app.callback()
def cli() -> None:
    """Custom Agent for DABench."""


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    app_config = load_app_config(config)
    
    _, run_output_dir = create_run_output_dir(
        app_config.run.output_dir, 
        run_id=app_config.run.run_id
    )
    
    model = OpenAIModelAdapter(
        model=app_config.agent.model,
        api_base=app_config.agent.api_base,
        api_key=app_config.agent.api_key,
        temperature=app_config.agent.temperature,
    )
    
    tools = create_default_tool_registry()
    
    agent = CustomAgent(
        model=model,
        tools=tools,
        config=CustomAgentConfig(max_steps=app_config.agent.max_steps),
    )
    
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    
    result = agent.run(task)
    
    from data_agent_baseline.run.runner import _write_task_outputs
    artifacts = _write_task_outputs(task_id, run_output_dir, result.to_dict())
    
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    if artifacts.failure_reason:
        console.print(f"Failure: {artifacts.failure_reason}")


if __name__ == "__main__":
    app()
