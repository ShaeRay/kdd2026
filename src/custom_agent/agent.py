"""
v4 Agent 核心模块

采用 plan-and-execute 架构，并将控制流拆成三个角色：
1. planner: 基于任务画像生成执行计划
2. executor: 按当前子目标调用受限工具
3. verifier: 对最终答案进行一次轻量校验

设计目标：
- 用非 LLM 路由隔离简单任务与文档型复杂任务
- 避免单个超长 Prompt 同时承担规划、执行、纠错和验证
- 保持 CLI 与 runner 接口不变，降低迁移成本
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask

from custom_agent.prompt import (
    build_executor_system_prompt,
    build_executor_user_prompt,
    build_observation_prompt,
    build_planner_system_prompt,
    build_planner_user_prompt,
    build_verifier_system_prompt,
    build_verifier_user_prompt,
)
from custom_agent.tools import describe_tools, execute_tool

PLAN_ACTION = "__plan__"
STEP_DONE_ACTION = "__step_done__"
VERIFY_ACTION = "__verify__"
FORMAT_ERROR_ACTION = "__format_error__"


@dataclass(frozen=True, slots=True)
class CustomAgentConfig:
    """
    v4 配置

    max_steps 仅限制 executor 回合数。
    planner 和 verifier 属于固定编排开销，不占用该预算。
    """

    max_steps: int = 25
    max_plan_steps: int = 5
    max_repair_attempts: int = 1


@dataclass(frozen=True, slots=True)
class TaskProfile:
    mode: str
    available_tools: list[str]
    context_files: list[str]
    file_type_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "available_tools": list(self.available_tools),
            "context_files": list(self.context_files),
            "file_type_counts": dict(self.file_type_counts),
        }

    def summary_text(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.file_type_counts.items()))
        files_preview = ", ".join(self.context_files[:12]) if self.context_files else "none"
        if len(self.context_files) > 12:
            files_preview += ", ..."
        return (
            f"mode={self.mode}\n"
            f"file_type_counts={counts or 'none'}\n"
            f"context_files={files_preview}"
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlanStep:
    step_id: str
    title: str
    goal: str
    allowed_tools: list[str]
    exit_criteria: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.step_id,
            "title": self.title,
            "goal": self.goal,
            "allowed_tools": list(self.allowed_tools),
            "exit_criteria": self.exit_criteria,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    summary: str
    steps: list[ExecutionPlanStep]
    final_columns: list[str]
    verification_checks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
            "final_columns": list(self.final_columns),
            "verification_checks": list(self.verification_checks),
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    approved: bool
    issues: list[str]
    repair_goal: str
    recommended_tools: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "issues": list(self.issues),
            "repair_goal": self.repair_goal,
            "recommended_tools": list(self.recommended_tools),
        }


def infer_task_profile(task: PublicTask) -> TaskProfile:
    context_files: list[str] = []
    counts = {
        "csv": 0,
        "json": 0,
        "db": 0,
        "doc": 0,
        "other": 0,
    }
    context_root = task.context_dir
    for path in sorted(context_root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(context_root).as_posix()
        context_files.append(rel_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            counts["csv"] += 1
        elif suffix == ".json":
            counts["json"] += 1
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            counts["db"] += 1
        elif suffix in {".md", ".txt"}:
            counts["doc"] += 1
        else:
            counts["other"] += 1

    has_doc_dir = any(path.startswith("doc/") for path in context_files)
    non_knowledge_docs = [
        path
        for path in context_files
        if path.endswith((".md", ".txt")) and Path(path).name.lower() != "knowledge.md"
    ]
    has_structured = counts["db"] > 0 or counts["csv"] > 0 or counts["json"] > 0

    if has_doc_dir and not has_structured:
        mode = "doc_heavy"
    elif has_doc_dir and has_structured:
        mode = "mixed"
    elif counts["db"] > 0:
        mode = "structured_db"
    elif counts["csv"] > 0 or counts["json"] > 0:
        mode = "structured_file"
    elif non_knowledge_docs:
        mode = "doc_heavy"
    else:
        mode = "structured_file"

    tool_map = {
        "structured_db": [
            "list_context",
            "read_doc",
            "inspect_sqlite",
            "execute_sql",
            "execute_python",
            "answer",
        ],
        "structured_file": [
            "list_context",
            "read_doc",
            "read_csv",
            "read_json",
            "execute_python",
            "answer",
        ],
        "doc_heavy": [
            "list_context",
            "read_doc",
            "extract_patterns",
            "execute_python",
            "answer",
        ],
        "mixed": [
            "list_context",
            "read_doc",
            "read_csv",
            "read_json",
            "inspect_sqlite",
            "execute_sql",
            "extract_patterns",
            "execute_python",
            "answer",
        ],
    }
    return TaskProfile(
        mode=mode,
        available_tools=tool_map[mode],
        context_files=context_files,
        file_type_counts=counts,
    )


def strip_json_fence(raw: str) -> str:
    text = raw.strip()
    fenced = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    generic = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return text


def fix_json_code_field(text: str) -> str:
    match = re.search(r'"code"\s*:\s*"', text)
    if not match:
        return text

    start = match.end()
    i = start
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            i += 2
            continue
        if char == '"':
            code_content = text[start:i]
            fixed_code = code_content.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            return text[:start] + fixed_code + text[i:]
        i += 1
    return text


def load_single_json_payload(raw: str) -> dict[str, Any]:
    normalized = fix_json_code_field(strip_json_fence(raw))
    try:
        payload, end = json.JSONDecoder().raw_decode(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    remainder = normalized[end:].strip()
    if remainder:
        cleaned = remainder.replace("```", "").strip()
        cleaned = re.sub(r"(?:\\[nrt])+", "", cleaned).strip()
        cleaned = cleaned.strip("}")
        if cleaned.strip():
            raise ValueError(f"Extra content after JSON: {remainder[:100]}")

    if not isinstance(payload, dict):
        raise ValueError(f"Response must be a JSON object, got {type(payload).__name__}")
    return payload


def parse_model_response(raw: str) -> ModelStep:
    payload = load_single_json_payload(raw)
    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError(f"'thought' must be a string, got {type(thought).__name__}")
    if not isinstance(action, str) or not action.strip():
        raise ValueError(f"'action' must be a non-empty string, got: {action!r}")
    if not isinstance(action_input, dict):
        raise ValueError(f"'action_input' must be an object, got {type(action_input).__name__}")
    return ModelStep(thought=thought, action=action, action_input=action_input, raw_response=raw)


def _unique(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_execution_plan(raw: str, profile: TaskProfile, max_plan_steps: int) -> ExecutionPlan:
    payload = load_single_json_payload(raw)
    summary = str(payload.get("summary", "")).strip() or "Follow a routed execution plan."
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Planner response must include a non-empty steps list.")

    steps: list[ExecutionPlanStep] = []
    for idx, raw_step in enumerate(raw_steps[:max_plan_steps], start=1):
        if not isinstance(raw_step, dict):
            continue
        title = str(raw_step.get("title", "")).strip() or f"Step {idx}"
        goal = str(raw_step.get("goal", "")).strip() or title
        exit_criteria = str(raw_step.get("exit_criteria", "")).strip() or "Current step goal is satisfied."
        requested_tools = _coerce_string_list(raw_step.get("allowed_tools"))
        allowed_tools = [tool for tool in requested_tools if tool in profile.available_tools]
        if not allowed_tools:
            allowed_tools = _default_tools_for_mode(profile.mode, final_step=False)
        steps.append(
            ExecutionPlanStep(
                step_id=f"step_{idx}",
                title=title,
                goal=goal,
                allowed_tools=allowed_tools,
                exit_criteria=exit_criteria,
            )
        )

    if not steps:
        raise ValueError("Planner response did not contain any valid steps.")

    final_columns = _coerce_string_list(payload.get("final_columns"))
    verification_checks = _coerce_string_list(payload.get("verification_checks"))
    plan = ExecutionPlan(
        summary=summary,
        steps=steps,
        final_columns=final_columns,
        verification_checks=verification_checks,
    )
    return sanitize_execution_plan(plan, profile)


def parse_verification_result(raw: str, available_tools: list[str]) -> VerificationResult:
    payload = load_single_json_payload(raw)
    approved = bool(payload.get("approved", False))
    issues = _coerce_string_list(payload.get("issues"))
    repair_goal = str(payload.get("repair_goal", "")).strip()
    recommended_tools = [tool for tool in _coerce_string_list(payload.get("recommended_tools")) if tool in available_tools]
    return VerificationResult(
        approved=approved,
        issues=issues,
        repair_goal=repair_goal,
        recommended_tools=recommended_tools,
    )


def _default_tools_for_mode(mode: str, *, final_step: bool) -> list[str]:
    defaults = {
        "structured_db": ["inspect_sqlite", "execute_sql", "execute_python"],
        "structured_file": ["read_csv", "read_json", "execute_python"],
        "doc_heavy": ["read_doc", "extract_patterns", "execute_python"],
        "mixed": ["read_doc", "inspect_sqlite", "execute_sql", "read_csv", "read_json", "extract_patterns", "execute_python"],
    }
    tools = list(defaults.get(mode, ["read_doc", "execute_python"]))
    if final_step and "answer" not in tools:
        tools.append("answer")
    return tools


def build_default_plan(profile: TaskProfile) -> ExecutionPlan:
    if profile.mode == "structured_db":
        steps = [
            ExecutionPlanStep("step_1", "Inspect schema", "Identify the relevant database tables and columns.", ["list_context", "read_doc", "inspect_sqlite"], "Relevant schema is known."),
            ExecutionPlanStep("step_2", "Query data", "Use SQL to retrieve the candidate rows needed for the question.", ["execute_sql", "read_doc"], "The required rows or aggregates are available."),
            ExecutionPlanStep("step_3", "Finalize answer", "Verify the result and submit the answer.", ["execute_sql", "execute_python", "answer"], "Final answer is ready to submit."),
        ]
    elif profile.mode == "structured_file":
        steps = [
            ExecutionPlanStep("step_1", "Inspect files", "Identify the relevant CSV or JSON files and their columns.", ["list_context", "read_doc", "read_csv", "read_json"], "Relevant file structure is understood."),
            ExecutionPlanStep("step_2", "Compute result", "Filter and compute the result from the structured files.", ["read_csv", "read_json", "execute_python"], "The result rows are available."),
            ExecutionPlanStep("step_3", "Finalize answer", "Verify the output schema and submit the answer.", ["execute_python", "answer"], "Final answer is ready to submit."),
        ]
    elif profile.mode == "doc_heavy":
        steps = [
            ExecutionPlanStep("step_1", "Read documents", "Read the relevant documents and identify how the data is described.", ["list_context", "read_doc"], "The source documents and extraction strategy are clear."),
            ExecutionPlanStep("step_2", "Extract candidates", "Use document-aware tools to extract the candidate facts or records.", ["read_doc", "extract_patterns", "execute_python"], "Candidate records are extracted."),
            ExecutionPlanStep("step_3", "Finalize answer", "Verify the extracted result against the document context and submit the answer.", ["read_doc", "execute_python", "answer"], "Final answer is ready to submit."),
        ]
    else:
        steps = [
            ExecutionPlanStep("step_1", "Map sources", "Identify which structured and document sources are relevant.", ["list_context", "read_doc", "inspect_sqlite", "read_csv", "read_json"], "The source of truth is identified."),
            ExecutionPlanStep("step_2", "Extract data", "Retrieve the candidate data from the chosen sources.", ["execute_sql", "read_csv", "read_json", "extract_patterns", "execute_python"], "Candidate result data is available."),
            ExecutionPlanStep("step_3", "Finalize answer", "Merge any evidence, validate the result, and submit the answer.", ["read_doc", "execute_sql", "execute_python", "answer"], "Final answer is ready to submit."),
        ]

    verification_checks = [
        "The final rows must directly answer the question.",
        "The column names should be explicit and stable.",
        "Use only values verified from the task context.",
    ]
    return sanitize_execution_plan(
        ExecutionPlan(
            summary=f"Fallback {profile.mode} plan with routed tool isolation.",
            steps=steps,
            final_columns=[],
            verification_checks=verification_checks,
        ),
        profile,
    )


def sanitize_execution_plan(plan: ExecutionPlan, profile: TaskProfile) -> ExecutionPlan:
    sanitized_steps: list[ExecutionPlanStep] = []
    total_steps = len(plan.steps)
    for idx, step in enumerate(plan.steps, start=1):
        final_step = idx == total_steps
        allowed = [tool for tool in step.allowed_tools if tool in profile.available_tools]
        if not allowed:
            allowed = _default_tools_for_mode(profile.mode, final_step=final_step)
        if final_step:
            allowed = _unique(allowed + ["answer"])
        else:
            allowed = [tool for tool in allowed if tool != "answer"]
        sanitized_steps.append(
            ExecutionPlanStep(
                step_id=f"step_{idx}",
                title=step.title,
                goal=step.goal,
                allowed_tools=allowed,
                exit_criteria=step.exit_criteria,
            )
        )

    verification_checks = plan.verification_checks or [
        "The answer should satisfy the question with no missing filters.",
        "The answer schema should match the requested output.",
    ]
    return ExecutionPlan(
        summary=plan.summary,
        steps=sanitized_steps,
        final_columns=plan.final_columns,
        verification_checks=verification_checks,
    )


def build_repair_step(
    *,
    profile: TaskProfile,
    repair_attempt: int,
    verifier_result: VerificationResult,
) -> ExecutionPlanStep:
    recommended_tools = [
        tool for tool in verifier_result.recommended_tools if tool in profile.available_tools and tool != "answer"
    ]
    if not recommended_tools:
        recommended_tools = _default_tools_for_mode(profile.mode, final_step=False)
    allowed_tools = _unique(recommended_tools + ["answer"])
    goal = verifier_result.repair_goal or "Fix the draft answer using the verifier feedback."
    return ExecutionPlanStep(
        step_id=f"repair_{repair_attempt}",
        title="Repair answer",
        goal=goal,
        allowed_tools=allowed_tools,
        exit_criteria="Submit a corrected final answer.",
    )


class CustomAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        config: CustomAgentConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or CustomAgentConfig()

    def _plan_messages(self, task: PublicTask, profile: TaskProfile) -> list[ModelMessage]:
        tool_catalog = describe_tools(profile.available_tools)
        return [
            ModelMessage(role="system", content=build_planner_system_prompt()),
            ModelMessage(
                role="user",
                content=build_planner_user_prompt(
                    question=task.question,
                    profile_summary=profile.summary_text(),
                    available_tools=profile.available_tools,
                    context_files=profile.context_files,
                    tool_catalog=tool_catalog,
                ),
            ),
        ]

    def _executor_messages(
        self,
        *,
        task: PublicTask,
        profile: TaskProfile,
        plan: ExecutionPlan,
        plan_step: ExecutionPlanStep,
        current_step_index: int,
        completed_step_summaries: list[str],
        current_step_records: list[StepRecord],
        verifier_feedback: str | None,
    ) -> list[ModelMessage]:
        tool_catalog = describe_tools(plan_step.allowed_tools)
        messages = [
            ModelMessage(role="system", content=build_executor_system_prompt()),
            ModelMessage(
                role="user",
                content=build_executor_user_prompt(
                    question=task.question,
                    profile_summary=profile.summary_text(),
                    plan_summary=plan.summary,
                    current_step_index=current_step_index,
                    total_steps=len(plan.steps),
                    current_step_title=plan_step.title,
                    current_step_goal=plan_step.goal,
                    current_step_exit_criteria=plan_step.exit_criteria,
                    allowed_actions=plan_step.allowed_tools + [STEP_DONE_ACTION],
                    completed_step_summaries=completed_step_summaries,
                    context_files=profile.context_files,
                    tool_catalog=tool_catalog,
                    verifier_feedback=verifier_feedback,
                ),
            ),
        ]
        for record in current_step_records:
            messages.append(ModelMessage(role="assistant", content=record.raw_response))
            messages.append(ModelMessage(role="user", content=build_observation_prompt(record.observation)))
        return messages

    def _verifier_messages(
        self,
        *,
        task: PublicTask,
        profile: TaskProfile,
        plan: ExecutionPlan,
        completed_step_summaries: list[str],
        answer: AnswerTable,
    ) -> list[ModelMessage]:
        return [
            ModelMessage(role="system", content=build_verifier_system_prompt()),
            ModelMessage(
                role="user",
                content=build_verifier_user_prompt(
                    question=task.question,
                    profile_summary=profile.summary_text(),
                    plan_summary=plan.summary,
                    verification_checks=plan.verification_checks,
                    completed_step_summaries=completed_step_summaries,
                    answer_columns=answer.columns,
                    answer_rows=answer.rows,
                    available_tools=profile.available_tools,
                ),
            ),
        ]

    def _record_format_error(self, state: AgentRuntimeState, step_index: int, raw_response: str, error_msg: str) -> None:
        observation = {
            "ok": False,
            "error": f"Format error: {error_msg}",
            "hint": "Respond with one JSON object only and no markdown fences.",
        }
        state.steps.append(
            StepRecord(
                step_index=step_index,
                thought="",
                action=FORMAT_ERROR_ACTION,
                action_input={},
                raw_response=raw_response,
                observation=observation,
                ok=False,
            )
        )

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        profile = infer_task_profile(task)
        trace_step_index = 1

        plan_raw = ""
        plan_error: str | None = None
        try:
            plan_raw = self.model.complete(self._plan_messages(task, profile))
            plan = parse_execution_plan(plan_raw, profile, self.config.max_plan_steps)
            plan_source = "planner"
            plan_ok = True
        except Exception as exc:
            plan_error = str(exc)
            plan = build_default_plan(profile)
            plan_source = "fallback"
            plan_ok = False

        state.steps.append(
            StepRecord(
                step_index=trace_step_index,
                thought=f"Route task into {profile.mode} mode and prepare a bounded plan.",
                action=PLAN_ACTION,
                action_input={"mode": profile.mode, "available_tools": profile.available_tools},
                raw_response=plan_raw,
                observation={
                    "ok": plan_ok,
                    "source": plan_source,
                    "profile": profile.to_dict(),
                    "plan": plan.to_dict(),
                    "error": plan_error,
                },
                ok=plan_ok,
            )
        )
        trace_step_index += 1

        pending_steps = list(plan.steps)
        completed_step_summaries: list[str] = []
        executor_turns = 0
        repair_attempts = 0

        while pending_steps:
            plan_step = pending_steps.pop(0)
            verifier_feedback: str | None = None
            current_step_records: list[StepRecord] = []

            while executor_turns < self.config.max_steps:
                raw_response = self.model.complete(
                    self._executor_messages(
                        task=task,
                        profile=profile,
                        plan=plan,
                        plan_step=plan_step,
                        current_step_index=len(completed_step_summaries) + 1,
                        completed_step_summaries=completed_step_summaries,
                        current_step_records=current_step_records,
                        verifier_feedback=verifier_feedback,
                    )
                )
                executor_turns += 1

                try:
                    model_step = parse_model_response(raw_response)
                except Exception as exc:
                    self._record_format_error(state, trace_step_index, raw_response, str(exc))
                    current_step_records.append(state.steps[-1])
                    trace_step_index += 1
                    continue

                allowed_actions = set(plan_step.allowed_tools)
                allowed_actions.add(STEP_DONE_ACTION)
                if model_step.action not in allowed_actions:
                    observation = {
                        "ok": False,
                        "error": f"Action not allowed in this step: {model_step.action}",
                        "allowed_actions": sorted(allowed_actions),
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=trace_step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                        )
                    )
                    current_step_records.append(state.steps[-1])
                    trace_step_index += 1
                    continue

                if model_step.action == STEP_DONE_ACTION:
                    summary = str(model_step.action_input.get("summary", "")).strip() or model_step.thought.strip()
                    if not summary:
                        summary = f"Completed {plan_step.title}."
                    completed_step_summaries.append(f"{plan_step.step_id}: {summary}")
                    state.steps.append(
                        StepRecord(
                            step_index=trace_step_index,
                            thought=model_step.thought,
                            action=STEP_DONE_ACTION,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation={
                                "ok": True,
                                "step_id": plan_step.step_id,
                                "summary": summary,
                            },
                            ok=True,
                        )
                    )
                    trace_step_index += 1
                    break

                try:
                    tool_result = execute_tool(task, model_step.action, model_step.action_input)
                    observation = {
                        "ok": tool_result.ok,
                        "tool": model_step.action,
                        "content": tool_result.content,
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=trace_step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=tool_result.ok,
                        )
                    )
                    current_step_records.append(state.steps[-1])
                    trace_step_index += 1

                    if not tool_result.is_terminal:
                        continue

                    if tool_result.answer is None:
                        state.failure_reason = "Terminal tool returned without an answer payload."
                        return AgentRunResult(
                            task_id=task.task_id,
                            answer=state.answer,
                            steps=list(state.steps),
                            failure_reason=state.failure_reason,
                        )

                    verify_raw = ""
                    try:
                        verify_raw = self.model.complete(
                            self._verifier_messages(
                                task=task,
                                profile=profile,
                                plan=plan,
                                completed_step_summaries=completed_step_summaries,
                                answer=tool_result.answer,
                            )
                        )
                        verification = parse_verification_result(verify_raw, profile.available_tools)
                    except Exception as exc:
                        verification = VerificationResult(
                            approved=True,
                            issues=[f"Verifier fallback: {exc}"],
                            repair_goal="",
                            recommended_tools=[],
                        )

                    state.steps.append(
                        StepRecord(
                            step_index=trace_step_index,
                            thought="Verify the draft answer before final acceptance.",
                            action=VERIFY_ACTION,
                            action_input={"plan_summary": plan.summary},
                            raw_response=verify_raw,
                            observation={"ok": verification.approved, "verification": verification.to_dict()},
                            ok=verification.approved,
                        )
                    )
                    trace_step_index += 1

                    if verification.approved:
                        state.answer = tool_result.answer
                        return AgentRunResult(
                            task_id=task.task_id,
                            answer=state.answer,
                            steps=list(state.steps),
                            failure_reason=None,
                        )

                    if repair_attempts >= self.config.max_repair_attempts:
                        issues = "; ".join(verification.issues) or verification.repair_goal or "Verifier rejected final answer."
                        state.failure_reason = f"Verifier rejected final answer: {issues}"
                        return AgentRunResult(
                            task_id=task.task_id,
                            answer=None,
                            steps=list(state.steps),
                            failure_reason=state.failure_reason,
                        )

                    repair_attempts += 1
                    verifier_feedback = verification.repair_goal or "; ".join(verification.issues)
                    completed_step_summaries.append(f"verifier_feedback: {verifier_feedback}")
                    pending_steps.insert(
                        0,
                        build_repair_step(
                            profile=profile,
                            repair_attempt=repair_attempts,
                            verifier_result=verification,
                        ),
                    )
                    break

                except Exception as exc:
                    state.steps.append(
                        StepRecord(
                            step_index=trace_step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation={
                                "ok": False,
                                "error": f"Tool execution error: {exc}",
                            },
                            ok=False,
                        )
                    )
                    current_step_records.append(state.steps[-1])
                    trace_step_index += 1
            else:
                state.failure_reason = f"Executor did not finish within {self.config.max_steps} turns."
                return AgentRunResult(
                    task_id=task.task_id,
                    answer=state.answer,
                    steps=list(state.steps),
                    failure_reason=state.failure_reason,
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Plan completed without submitting an answer."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
