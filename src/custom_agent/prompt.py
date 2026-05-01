"""
v4 提示词模块

将原先单一的超长 ReAct Prompt 拆分为：
1. planner: 负责产出简短执行计划
2. executor: 负责按当前子目标调用工具
3. verifier: 负责校验最终答案是否满足问题要求
"""

from __future__ import annotations

import json
from typing import Iterable


def _render_lines(items: Iterable[str]) -> str:
    lines = [f"- {item}" for item in items]
    return "\n".join(lines) if lines else "- none"


def build_planner_system_prompt() -> str:
    return """
You are the planner subagent for a data-analysis agent.

Respond with ONE plain JSON object only. Do not use markdown fences.

Required format:
{
  "summary": "one short paragraph",
  "steps": [
    {
      "id": "step_1",
      "title": "short title",
      "goal": "what this step must accomplish",
      "allowed_tools": ["tool_name"],
      "exit_criteria": "how to know this step is complete"
    }
  ],
  "final_columns": ["expected", "output", "columns"],
  "verification_checks": ["check 1", "check 2"]
}

Rules:
- Produce 2 to 5 steps.
- Use only tools listed in the user message.
- Keep the plan sequential and concrete.
- Prefer structured tools before execute_python.
- Only include answer in the final step.
- Use extract_patterns only when documents are the primary source of truth.
""".strip()


def build_planner_user_prompt(
    *,
    question: str,
    profile_summary: str,
    available_tools: list[str],
    context_files: list[str],
    tool_catalog: str,
) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Task profile:\n{profile_summary}\n\n"
        f"Available tools:\n{_render_lines(available_tools)}\n\n"
        f"Exact context files:\n{_render_lines(context_files)}\n\n"
        f"Tool catalog:\n{tool_catalog}\n\n"
        "Create the shortest safe plan that can answer the question.\n"
        "If knowledge.md exists, treat it as a high-priority semantic guide before concluding data is missing."
    )


def build_executor_system_prompt() -> str:
    return """
You are the executor subagent for a data-analysis workflow.

Respond with ONE plain JSON object only. Do not use markdown fences.

Required format:
{
  "thought": "brief reasoning",
  "action": "tool_name_or___step_done__",
  "action_input": {}
}

Rules:
- Use only the allowed actions from the user message.
- Take exactly one action at a time.
- When the current step goal is complete, use action "__step_done__" with {"summary": "..."}.
- Use answer only when you have the final result table.
- Prefer inspect_sqlite before execute_sql when schema is unknown.
- Prefer read_csv/read_json/read_doc/execute_sql before execute_python.
- Keep execute_python snippets short and practical.
- When execute_python can directly compute the full answer, print FINAL_ANSWER: followed by JSON.
""".strip()


def build_executor_user_prompt(
    *,
    question: str,
    profile_summary: str,
    plan_summary: str,
    current_step_index: int,
    total_steps: int,
    current_step_title: str,
    current_step_goal: str,
    current_step_exit_criteria: str,
    allowed_actions: list[str],
    completed_step_summaries: list[str],
    context_files: list[str],
    tool_catalog: str,
    verifier_feedback: str | None = None,
) -> str:
    sections = [
        f"Question:\n{question}",
        f"Task profile:\n{profile_summary}",
        f"Plan summary:\n{plan_summary}",
        (
            f"Current step ({current_step_index}/{total_steps}): {current_step_title}\n"
            f"Goal: {current_step_goal}\n"
            f"Exit criteria: {current_step_exit_criteria}"
        ),
        f"Allowed actions:\n{_render_lines(allowed_actions)}",
        f"Completed step summaries:\n{_render_lines(completed_step_summaries)}",
        f"Exact context files:\n{_render_lines(context_files)}",
        f"Tool catalog:\n{tool_catalog}",
    ]
    if verifier_feedback:
        sections.append(f"Verifier feedback to address:\n{verifier_feedback}")
    sections.append(
        "Choose the single best next action.\n"
        "Paths must exactly match one of the listed context files.\n"
        "Do not invent folders or filenames.\n"
        "If knowledge.md exists and has not been read yet, do not conclude data is missing."
    )
    return "\n\n".join(sections)


def build_verifier_system_prompt() -> str:
    return """
You are the verifier subagent for a data-analysis workflow.

Respond with ONE plain JSON object only. Do not use markdown fences.

Required format:
{
  "approved": true,
  "issues": ["issue text"],
  "repair_goal": "empty string if approved",
  "recommended_tools": ["tool_name"]
}

Rules:
- Approve only if the answer directly addresses the question.
- Check column names, row shape, filtering logic, and obvious missing validation.
- If rejected, provide one concrete repair goal and up to 4 recommended tools.
- Use only tools that appear in the user message.
""".strip()


def build_verifier_user_prompt(
    *,
    question: str,
    profile_summary: str,
    plan_summary: str,
    verification_checks: list[str],
    completed_step_summaries: list[str],
    answer_columns: list[str],
    answer_rows: list[list[object]],
    available_tools: list[str],
) -> str:
    answer_preview = json.dumps(
        {
            "columns": answer_columns,
            "rows": answer_rows[:10],
            "row_count": len(answer_rows),
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        f"Question:\n{question}\n\n"
        f"Task profile:\n{profile_summary}\n\n"
        f"Plan summary:\n{plan_summary}\n\n"
        f"Verification checks:\n{_render_lines(verification_checks)}\n\n"
        f"Completed step summaries:\n{_render_lines(completed_step_summaries)}\n\n"
        f"Draft answer preview:\n{answer_preview}\n\n"
        f"Allowed repair tools:\n{_render_lines(available_tools)}"
    )


def build_observation_prompt(observation: dict) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}\n\nRespond with the next JSON object."
