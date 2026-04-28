"""
工具模块

定义 Agent 可使用的所有工具及其执行逻辑。
工具包括：
- 文件系统工具：list_context, read_csv, read_json, read_doc
- 数据库工具：inspect_sqlite, execute_sql
- 代码执行工具：execute_python
- 数据提取工具：extract_patterns
- 答案提交工具：answer

安全机制：
- 路径沙箱：所有文件操作限制在 context 目录内
- SQL 只读：只允许 SELECT/WITH/PRAGMA 语句
- Python 隔离：子进程执行，超时控制
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask


@dataclass(frozen=True, slots=True)
class ToolResult:
    """
    工具执行结果
    
    Attributes:
        ok: 执行是否成功
        content: 返回内容字典
        is_terminal: 是否为终止操作（如 answer 工具）
        answer: 如果是终止操作，包含最终答案表格
    """
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    """
    解析并验证上下文路径
    
    安全机制：确保请求的路径不会逃逸出任务的 context 目录。
    
    Args:
        task: 任务对象，包含 context_dir 属性
        relative_path: 相对于 context 目录的路径
    
    Returns:
        解析后的绝对路径
    
    Raises:
        ValueError: 路径逃逸出 context 目录
        FileNotFoundError: 文件不存在
    """
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    # 检查路径是否在 context 目录内
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def tool_list_context(task: PublicTask, action_input: dict) -> ToolResult:
    """
    列出上下文目录中的文件和子目录
    
    递归遍历 context 目录，返回文件树结构。
    
    Args:
        task: 任务对象
        action_input: 输入参数，包含 max_depth 控制遍历深度
    
    Returns:
        包含 root（根路径）和 entries（文件条目列表）的结果
    """
    max_depth = int(action_input.get("max_depth", 4))
    entries: list[dict] = []

    def walk(path: Path, depth: int):
        """递归遍历目录"""
        if depth > max_depth:
            return
        # 按类型和名称排序：目录在前，文件在后
        for child in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append({
                "path": rel_path,
                "kind": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            })
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return ToolResult(ok=True, content={"root": str(task.context_dir), "entries": entries})


def tool_read_csv(task: PublicTask, action_input: dict) -> ToolResult:
    """
    读取 CSV 文件预览
    
    智能预览策略：
    - 默认只返回前 50 行用于了解数据结构
    - 同时返回 total_rows 让 agent 知晓总行数
    - 如果请求行数超过 100，强制预览模式并提示使用 execute_python
    
    Args:
        task: 任务对象
        action_input: 包含 path 和可选的 max_rows 参数
    
    Returns:
        包含 columns、rows（预览）、total_rows、preview_mode 的结果
    """
    path = resolve_context_path(task, str(action_input["path"]))
    requested_rows = int(action_input.get("max_rows", 50))
    
    preview_limit = 50
    if requested_rows > 100:
        max_rows = preview_limit
        forced_preview = True
    else:
        max_rows = min(requested_rows, preview_limit)
        forced_preview = False
    
    with path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows:
        return ToolResult(ok=True, content={
            "path": str(action_input["path"]),
            "columns": [],
            "rows": [],
            "row_count": 0,
            "total_rows": 0,
            "preview_mode": False,
        })
    
    header = rows[0]
    data_rows = rows[1:]
    total_rows = len(data_rows)
    
    content = {
        "path": str(action_input["path"]),
        "columns": header,
        "rows": data_rows[:max_rows],
        "row_count": max_rows,
        "total_rows": total_rows,
        "preview_mode": total_rows > max_rows,
    }
    
    if forced_preview:
        content["hint"] = f"File has {total_rows} rows. Preview shows first {max_rows} rows. Use execute_python for full data processing."
    elif total_rows > max_rows:
        content["hint"] = f"File has {total_rows} rows. Showing first {max_rows}. Use execute_python to process all data."
    
    return ToolResult(ok=True, content=content)


def tool_read_json(task: PublicTask, action_input: dict) -> ToolResult:
    """
    读取 JSON 文件预览
    
    Args:
        task: 任务对象
        action_input: 包含 path 和可选的 max_chars 参数
    
    Returns:
        包含 preview（JSON 格式化预览）、truncated（是否截断）、total_chars（总字符数）的结果
    """
    path = resolve_context_path(task, str(action_input["path"]))
    max_chars = int(action_input.get("max_chars", 50000))
    payload = json.loads(path.read_text())
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return ToolResult(ok=True, content={
        "path": str(action_input["path"]),
        "preview": preview[:max_chars],
        "truncated": len(preview) > max_chars,
        "total_chars": len(preview),
    })


def tool_read_doc(task: PublicTask, action_input: dict) -> ToolResult:
    """
    读取文本文档（.md, .txt 等）
    
    Args:
        task: 任务对象
        action_input: 包含 path 和可选的 max_chars 参数
    
    Returns:
        包含 preview（文本预览）、truncated（是否截断）的结果
    """
    path = resolve_context_path(task, str(action_input["path"]))
    max_chars = int(action_input.get("max_chars", 8000))
    text = path.read_text(errors="replace")
    return ToolResult(ok=True, content={
        "path": str(action_input["path"]),
        "preview": text[:max_chars],
        "truncated": len(text) > max_chars,
    })


def tool_inspect_sqlite(task: PublicTask, action_input: dict) -> ToolResult:
    """
    检查 SQLite 数据库结构
    
    列出数据库中的所有表及其 CREATE TABLE 语句。
    
    Args:
        task: 任务对象
        action_input: 包含 path 参数
    
    Returns:
        包含 tables（表信息列表）的结果
    """
    path = resolve_context_path(task, str(action_input["path"]))
    # 使用只读模式连接
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables = [{"name": name, "create_sql": create_sql} for name, create_sql in rows]
    return ToolResult(ok=True, content={"path": str(action_input["path"]), "tables": tables})


def tool_execute_sql(task: PublicTask, action_input: dict) -> ToolResult:
    """
    在 SQLite 数据库上执行只读 SQL 查询
    
    安全机制：只允许 SELECT、WITH、PRAGMA 语句。
    
    Args:
        task: 任务对象
        action_input: 包含 path、sql 和可选的 limit 参数
    
    Returns:
        包含 columns（列名）、rows（数据行）、row_count（返回行数）、truncated（是否截断）的结果
    """
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 500))
    # 安全检查：只允许只读语句
    normalized = sql.lstrip().lower()
    if not normalized.startswith(("select", "with", "pragma")):
        raise ValueError("Only read-only SQL statements are allowed.")
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)
    truncated = len(rows) > limit
    return ToolResult(ok=True, content={
        "path": str(action_input["path"]),
        "columns": columns,
        "rows": [list(r) for r in rows[:limit]],
        "row_count": len(rows[:limit]),
        "truncated": truncated,
    })


# Python 执行超时时间（秒）
EXECUTE_PYTHON_TIMEOUT = 60

# 最终答案标记：当 Python 输出包含此标记时，自动提取答案
ANSWER_MARKER = "FINAL_ANSWER:"


def tool_execute_python(task: PublicTask, action_input: dict) -> ToolResult:
    """
    执行 Python 代码
    
    在子进程中执行代码，工作目录为任务的 context 目录。
    支持 FINAL_ANSWER 机制：当代码输出包含 "FINAL_ANSWER:" 标记时，
    自动解析后续 JSON 作为最终答案并终止任务。
    
    安全机制：
    - 子进程隔离，避免影响主进程
    - 超时控制，防止无限循环
    
    Args:
        task: 任务对象
        action_input: 包含 code 参数（要执行的 Python 代码）
    
    Returns:
        包含 success、output（stdout）、stderr 的结果
        如果检测到 FINAL_ANSWER，则 is_terminal=True 并包含 answer
    """
    code = str(action_input["code"])
    context_root = task.context_dir.resolve()
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            cwd=context_root,
            capture_output=True,
            text=True,
            timeout=EXECUTE_PYTHON_TIMEOUT,
        )
        output = result.stdout
        stderr = result.stderr
        success = result.returncode == 0
        error_msg = None if success else f"Exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        output = ""
        stderr = ""
        success = False
        error_msg = f"Timeout after {EXECUTE_PYTHON_TIMEOUT}s"
    except Exception as e:
        output = ""
        stderr = traceback.format_exc()
        success = False
        error_msg = str(e)
    
    # 检查输出是否包含最终答案标记
    if ANSWER_MARKER in output:
        try:
            answer_json = output.split(ANSWER_MARKER, 1)[1].strip()
            answer_data = json.loads(answer_json)
            columns = answer_data.get("columns", [])
            rows = answer_data.get("rows", [])
            if columns and rows:
                answer = AnswerTable(columns=list(columns), rows=[list(r) for r in rows])
                return ToolResult(
                    ok=True,
                    content={"success": True, "output": output, "stderr": stderr, "answer_submitted": True, "row_count": len(rows)},
                    is_terminal=True,
                    answer=answer,
                )
        except Exception:
            pass
    
    content = {"success": success, "output": output, "stderr": stderr}
    if error_msg:
        content["error"] = error_msg
    return ToolResult(ok=success, content=content)


def tool_answer(task: PublicTask, action_input: dict) -> ToolResult:
    """
    提交最终答案
    
    这是终止工具，调用后 Agent 循环结束。
    
    Args:
        task: 任务对象
        action_input: 包含 columns（列名列表）和 rows（数据行列表）
    
    Returns:
        is_terminal=True 的结果，包含 answer
    """
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    # 验证 columns
    if not isinstance(columns, list) or not columns:
        raise ValueError("columns must be a non-empty list")
    # 验证 rows
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    normalized_rows = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each row must be a list")
        if len(row) != len(columns):
            raise ValueError(f"Row length {len(row)} != column count {len(columns)}")
        normalized_rows.append(list(row))
    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolResult(
        ok=True,
        content={"status": "submitted", "column_count": len(columns), "row_count": len(normalized_rows)},
        is_terminal=True,
        answer=answer,
    )


def tool_extract_patterns(task: PublicTask, action_input: dict) -> ToolResult:
    """
    从文档中提取匹配模式的数据
    
    用于从非结构化文本（如 .md 文档）中提取结构化数据。
    支持多个命名 pattern，返回匹配结果列表。
    
    Args:
        task: 任务对象
        action_input: 包含以下参数：
            - path: 文档路径
            - patterns: 字典，key 为字段名，value 为正则表达式
            - combine: 可选，是否将同一位置的匹配组合为一条记录（默认 False）
            - include_context: 可选，是否包含匹配位置的上下文文本（默认 False）
            - context_chars: 可选，上下文字符数（默认 200）
            - all_groups: 可选，是否返回所有捕获组（默认 False，仅返回第一个）
            - search_window: 可选，combine 模式下搜索窗口大小（默认 500）
            - search_forward: 可选，combine 模式下是否只向前搜索（默认 True，避免匹配到前一条记录的数据）
            - limit: 可选，返回结果数量限制（默认 100，避免返回过多数据）
    
    Returns:
        包含 matches（匹配结果列表）和 total_matches（总匹配数）的结果
    
    Example:
        action_input = {
            "path": "doc/Laboratory.md",
            "patterns": {
                "patient_id": r"patient\s+(\d+)",
                "creatinine": r"creatinine[^;]*?(\d+\.\d+)\s*mg/dL"
            },
            "combine": True,
            "include_context": True,
            "all_groups": True
        }
    """
    path = resolve_context_path(task, str(action_input["path"]))
    patterns = action_input.get("patterns", {})
    combine = action_input.get("combine", False)
    include_context = action_input.get("include_context", False)
    context_chars = int(action_input.get("context_chars", 200))
    all_groups = action_input.get("all_groups", False)
    search_window = int(action_input.get("search_window", 500))
    search_forward = action_input.get("search_forward", True)
    limit = int(action_input.get("limit", 500))
    
    if not patterns:
        raise ValueError("patterns must be a non-empty dictionary")
    
    text = path.read_text(errors="replace")
    
    def extract_match_value(match: re.Match) -> Any:
        if not match.groups():
            return match.group(0)
        if all_groups:
            groups = match.groups()
            return groups[0] if len(groups) == 1 else list(groups)
        return match.group(1)
    
    def extract_match_context(match: re.Match, text: str, chars: int) -> str:
        start = max(0, match.start() - chars)
        end = min(len(text), match.end() + chars)
        return text[start:end]
    
    if combine:
        pattern_names = list(patterns.keys())
        primary_name = pattern_names[0]
        primary_pattern = patterns[primary_name]
        
        combined_matches = []
        total_count = 0
        for match in re.finditer(primary_pattern, text, re.IGNORECASE):
            total_count += 1
            if len(combined_matches) >= limit:
                continue
            
            record = {primary_name: extract_match_value(match)}
            record["_matched_text"] = match.group(0)
            
            if search_forward:
                context_start = match.start()
                context_end = min(len(text), match.end() + search_window)
            else:
                context_start = max(0, match.start() - search_window)
                context_end = min(len(text), match.end() + search_window)
            context = text[context_start:context_end]
            
            if include_context:
                record["_context"] = context
            
            for name in pattern_names[1:]:
                other_pattern = patterns[name]
                other_match = re.search(other_pattern, context, re.IGNORECASE)
                if other_match:
                    record[name] = extract_match_value(other_match)
                    record[f"_{name}_matched_text"] = other_match.group(0)
                else:
                    record[name] = None
            
            combined_matches.append(record)
        
        return ToolResult(ok=True, content={
            "path": str(action_input["path"]),
            "matches": combined_matches,
            "total_matches": total_count,
            "returned_matches": len(combined_matches),
            "truncated": total_count > limit,
            "mode": "combined",
        })
    else:
        results = {}
        matched_texts = {}
        contexts = {} if include_context else None
        total_count = 0
        for name, pattern in patterns.items():
            matches = []
            texts = []
            match_contexts = [] if include_context else None
            count = 0
            for match in re.finditer(pattern, text, re.IGNORECASE):
                count += 1
                total_count += 1
                if len(matches) < limit:
                    matches.append(extract_match_value(match))
                    texts.append(match.group(0))
                    if include_context:
                        match_contexts.append(extract_match_context(match, text, context_chars))
            results[name] = matches
            matched_texts[name] = texts
            if include_context:
                contexts[name] = match_contexts
        
        content = {
            "path": str(action_input["path"]),
            "matches": results,
            "matched_texts": matched_texts,
            "total_matches": total_count,
            "returned_matches": sum(len(v) for v in results.values()),
            "truncated": total_count > limit,
            "mode": "independent",
        }
        if include_context:
            content["contexts"] = contexts
        
        return ToolResult(ok=True, content=content)


# 工具规格定义：用于生成工具描述给模型
TOOL_SPECS = {
    "list_context": {
        "description": "List files and directories in context.",
        "input_schema": {"max_depth": 4},
    },
    "read_csv": {
        "description": "Preview CSV file structure (first 50 rows). Returns columns, preview rows, and total_rows. For large files, use execute_python.",
        "input_schema": {"path": "file.csv"},
    },
    "read_json": {
        "description": "Preview a JSON file.",
        "input_schema": {"path": "file.json", "max_chars": 50000},
    },
    "read_doc": {
        "description": "Read a text document.",
        "input_schema": {"path": "file.md", "max_chars": 8000},
    },
    "inspect_sqlite": {
        "description": "Inspect SQLite database schema.",
        "input_schema": {"path": "file.sqlite"},
    },
    "execute_sql": {
        "description": "Run read-only SQL query on SQLite database.",
        "input_schema": {"path": "file.sqlite", "sql": "SELECT ...", "limit": 500},
    },
    "execute_python": {
        "description": f"Execute Python code (timeout: {EXECUTE_PYTHON_TIMEOUT}s). Working dir is context/. To submit answer, print 'FINAL_ANSWER:' followed by JSON: {{\"columns\": [...], \"rows\": [...]}}",
        "input_schema": {"code": "import pandas as pd\n...\nprint('FINAL_ANSWER:', json.dumps({'columns': ['id'], 'rows': [[1], [2]]}))"},
    },
    "extract_patterns": {
        "description": "Extract structured data from documents using regex patterns. Use when data is embedded in text (e.g., patient data in .md files). Set include_context=true to see surrounding text for validation. Set all_groups=true to return all capture groups. Returns matched_texts for verification. By default, searches forward from primary match to avoid matching previous records. Default limit=500.",
        "input_schema": {"path": "doc/file.md", "patterns": {"patient_id": r"patient\s+(\d+)", "creatinine": r"creatinine[^;]*?(\d+\.\d+)\s*mg/dL"}, "combine": True, "include_context": True, "all_groups": False, "search_window": 500, "search_forward": True, "limit": 500},
    },
    "answer": {
        "description": "Submit final answer table. This terminates the task.",
        "input_schema": {"columns": ["col1", "col2"], "rows": [["val1", "val2"]]},
    },
}

# 工具处理器映射
TOOL_HANDLERS = {
    "list_context": tool_list_context,
    "read_csv": tool_read_csv,
    "read_json": tool_read_json,
    "read_doc": tool_read_doc,
    "inspect_sqlite": tool_inspect_sqlite,
    "execute_sql": tool_execute_sql,
    "execute_python": tool_execute_python,
    "extract_patterns": tool_extract_patterns,
    "answer": tool_answer,
}


def describe_tools() -> str:
    """
    生成工具描述字符串
    
    用于在系统提示词中列出所有可用工具。
    
    Returns:
        格式化的工具描述字符串
    """
    lines = []
    for name, spec in TOOL_SPECS.items():
        lines.append(f"- {name}: {spec['description']}")
        lines.append(f"  input_schema: {spec['input_schema']}")
    return "\n".join(lines)


def execute_tool(task: PublicTask, action: str, action_input: dict) -> ToolResult:
    """
    执行指定工具
    
    Args:
        task: 任务对象
        action: 工具名称
        action_input: 工具输入参数
    
    Returns:
        工具执行结果
    
    Raises:
        KeyError: 未知工具名称
    """
    handler = TOOL_HANDLERS.get(action)
    if not handler:
        raise KeyError(f"Unknown tool: {action}")
    return handler(task, action_input)
