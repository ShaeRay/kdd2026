from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class CustomAgentConfig:
    max_steps: int = 16


class CustomAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: CustomAgentConfig | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or CustomAgentConfig()

    def run(self, task: PublicTask) -> AgentRunResult:
        steps: list[StepRecord] = []
        answer: AnswerTable | None = None
        failure_reason: str | None = None

        messages = [
            ModelMessage(role="system", content="You are a data analysis agent."),
            ModelMessage(role="user", content=task.question),
        ]

        for step_index in range(1, self.config.max_steps + 1):
            raw_response = self.model.complete(messages)
            
            # TODO: 解析模型响应，执行工具，构建观察结果
            # 示例：
            # model_step = self._parse_response(raw_response)
            # tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
            # 
            # if tool_result.is_terminal:
            #     answer = tool_result.answer
            #     break
            
            step_record = StepRecord(
                step_index=step_index,
                thought="",
                action="",
                action_input={},
                raw_response=raw_response,
                observation={},
                ok=True,
            )
            steps.append(step_record)

        if answer is None:
            failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=answer,
            steps=steps,
            failure_reason=failure_reason,
        )
