"""
Agent 核心模块

实现 ReAct 风格的 Agent 循环：
1. 构建消息（系统提示词 + 任务提示词 + 历史步骤）
2. 调用模型获取响应
3. 解析响应为 (thought, action, action_input)
4. 执行工具获取观察结果
5. 重复直到获得答案或达到最大步数

错误恢复机制：
- 连续 3 次格式错误才终止，单次错误继续循环
- 工具执行错误记录但不终止循环
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask

from custom_agent.prompt import build_system_prompt, build_task_prompt, build_observation_prompt
from custom_agent.tools import execute_tool


@dataclass(frozen=True, slots=True)
class CustomAgentConfig:
    """
    Agent 配置
    
    Attributes:
        max_steps: 最大执行步数，防止无限循环
    """
    max_steps: int = 25


def strip_json_fence(raw: str) -> str:
    """
    去除 JSON 代码围栏
    
    支持两种格式：
    - ```json ... ```（推荐）
    - ``` ... ```（通用）
    
    Args:
        raw: 原始响应文本
    
    Returns:
        去除围栏后的 JSON 文本
    """
    text = raw.strip()
    # 优先匹配 ```json 围栏
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # 其次匹配通用围栏
    m = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def parse_model_response(raw: str) -> ModelStep:
    """
    解析模型响应为结构化的步骤
    
    期望格式：
    ```json
    {"thought": "...", "action": "...", "action_input": {...}}
    ```
    
    Args:
        raw: 模型原始响应文本
    
    Returns:
        包含 thought、action、action_input 的 ModelStep 对象
    
    Raises:
        ValueError: JSON 无效或缺少必需字段
    """
    normalized = strip_json_fence(raw)
    try:
        payload, end = json.JSONDecoder().raw_decode(normalized)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    # 检查是否有额外内容
    remainder = normalized[end:].strip()
    if remainder:
        cleaned = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned:
            raise ValueError(f"Extra content after JSON: {cleaned[:100]}")
    # 验证必需字段
    if not isinstance(payload, dict):
        raise ValueError(f"Response must be a JSON object, got {type(payload).__name__}")
    if "thought" not in payload:
        raise ValueError("Missing required field: 'thought'")
    if "action" not in payload:
        raise ValueError("Missing required field: 'action'")
    if "action_input" not in payload:
        raise ValueError("Missing required field: 'action_input'")
    thought = payload["thought"]
    action = payload["action"]
    action_input = payload["action_input"]
    # 验证字段类型
    if not isinstance(thought, str):
        raise ValueError(f"'thought' must be a string, got {type(thought).__name__}")
    if not isinstance(action, str) or not action.strip():
        raise ValueError(f"'action' must be a non-empty string, got: {action!r}")
    if not isinstance(action_input, dict):
        raise ValueError(f"'action_input' must be an object, got {type(action_input).__name__}")
    return ModelStep(thought=thought, action=action, action_input=action_input, raw_response=raw)


class CustomAgent:
    """
    自定义 ReAct Agent
    
    实现 Thought-Action-Observation 循环：
    1. 根据当前状态构建消息
    2. 调用模型获取下一步行动
    3. 执行工具并记录观察结果
    4. 重复直到获得答案或达到最大步数
    """
    
    def __init__(
        self,
        *,
        model: ModelAdapter,
        config: CustomAgentConfig | None = None,
    ) -> None:
        """
        初始化 Agent
        
        Args:
            model: 模型适配器，用于调用 LLM API
            config: Agent 配置，默认使用 CustomAgentConfig()
        """
        self.model = model
        self.config = config or CustomAgentConfig()

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        """
        构建对话消息列表
        
        消息结构：
        1. system: 系统提示词（定义 Agent 行为规范）
        2. user: 任务提示词（包含具体问题）
        3. assistant/user 交替: 历史步骤的响应和观察
        
        Args:
            task: 任务对象
            state: 运行时状态，包含已执行的步骤
        
        Returns:
            完整的消息列表
        """
        system_content = build_system_prompt("")
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task.question)))
        # 添加历史步骤
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(ModelMessage(role="user", content=build_observation_prompt(step.observation)))
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        """
        运行 Agent 解决任务
        
        核心循环：
        1. 调用模型获取响应
        2. 解析响应为 (thought, action, action_input)
        3. 执行工具获取观察结果
        4. 如果是终止工具（answer），结束循环
        
        错误处理：
        - 格式错误：记录并继续，连续 3 次才终止
        - 工具错误：记录并继续
        
        Args:
            task: 任务对象
        
        Returns:
            包含答案、步骤记录、失败原因的 AgentRunResult
        """
        state = AgentRuntimeState()
        consecutive_errors = 0
        max_consecutive_errors = 3

        for step_index in range(1, self.config.max_steps + 1):
            # 步骤 1: 调用模型并解析响应
            try:
                raw_response = self.model.complete(self._build_messages(task, state))
                model_step = parse_model_response(raw_response)
                consecutive_errors = 0  # 重置错误计数
            except Exception as exc:
                consecutive_errors += 1
                error_msg = str(exc)
                # 记录格式错误
                observation = {
                    "ok": False,
                    "error": f"Format error: {error_msg}",
                    "hint": "You MUST respond with: {\"thought\": \"...\", \"action\": \"tool_name\", \"action_input\": {...}}",
                }
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="",
                    action="__format_error__",
                    action_input={},
                    raw_response=raw_response if "raw_response" in dir() else "",
                    observation=observation,
                    ok=False,
                ))
                # 连续多次格式错误，终止
                if consecutive_errors >= max_consecutive_errors:
                    state.failure_reason = f"Too many format errors. Last error: {error_msg}"
                    break
                continue

            # 步骤 2: 执行工具
            try:
                tool_result = execute_tool(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                ))
                # 如果是终止工具，保存答案并结束
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except KeyError as exc:
                # 未知工具
                observation = {
                    "ok": False,
                    "error": f"Unknown tool: {model_step.action}",
                    "available_tools": ["list_context", "read_csv", "read_json", "read_doc", "inspect_sqlite", "execute_sql", "execute_python", "answer"],
                }
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=False,
                ))
            except Exception as exc:
                # 工具执行错误
                observation = {
                    "ok": False,
                    "error": f"Tool execution error: {exc}",
                }
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=False,
                ))

        # 检查是否获得答案
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = f"Agent did not submit answer within {self.config.max_steps} steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
