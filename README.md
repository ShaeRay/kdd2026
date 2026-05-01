# DABench 数据代理基准测试

## 项目结构

- `src/data_agent_baseline/` - Baseline ReAct Agent
- `src/custom_agent/` - 改进的 Custom Agent（推荐使用）

## 运行步骤

### 1. 安装依赖
```bash
uv sync
```

### 2. 配置 API
编辑 `configs/custom_agent.yaml`，填入你的 API 信息：
```yaml
agent:
  model: 你的模型名称
  api_base: 你的API地址
  api_key: 你的API密钥
```

### 3. 运行任务

```bash
# 单任务
uv run custom-agent run-task task_11 --config configs/custom_agent.yaml

# 批量运行
uv run custom-agent run-benchmark --config configs/custom_agent.yaml
```

### 4. 其他命令

```bash
# 查看项目状态
uv run custom-agent status --config configs/custom_agent.yaml

# 查看任务详情
uv run custom-agent inspect-task task_11 --config configs/custom_agent.yaml
```

## Benchmark 结果

### Custom Agent (2026-04-28)

| 指标 | 数值 |
|------|------|
| 总任务数 | 50 |
| 成功 | 47 |
| 失败 | 3 |
| 成功率 | **94%** |
| 平均速率 | 0.8 task/min |

### 失败任务分析

| 任务 | 原因 |
|------|------|
| task_200 | 未使用 inspect_sqlite 检查表结构 |
| task_355 | JSON 格式错误 |
| task_418 | 非结构化文本解析，步骤不足 |

### 相比 Baseline 改进

| 指标 | Baseline | Custom Agent | 提升 |
|------|----------|--------------|------|
| 成功数 | 25 | 47 | +88% |
| 错误数 | 16 | 0 | -100% |
| 失败数 | 9 | 3 | -67% |

## 改进点

1. **FINAL_ANSWER 机制**：解决大量数据提交时 JSON 截断问题
2. **增强错误恢复**：连续 3 次格式错误才终止
3. **优化提示词**：强调工具使用顺序和输出格式
4. **增加 max_steps**：从 16 提升到 30

## 架构演进与教训

### v1 → v3 的尝试与反思

**v1 版本（成功）**
- 成功率：94% (47/50)
- 特点：简洁的 Prompt (~100 行)，专注核心工具
- 问题：无法处理 task_418（非结构化文档解析）

**v3 版本（失败）**
- 成功率：84% (42/50) ⚠️ **下降 10%**
- 尝试的改进：
  - 添加 `extract_patterns` 工具用于文档解析
  - 增加大量 Prompt 规则（JSON 格式、数据提取最佳实践等）
  - Prompt 膨胀至 ~247 行
- **失败原因**：
  1. **Prompt 过长**：LLM 被冗长指令干扰，产生更多格式错误
  2. **过度指导**：详细的"CRITICAL RULES"适得其反
  3. **工具滥用**：新工具在简单任务中造成干扰
  4. **回归问题**：39 个原本成功的任务中，15% 产生错误答案

### 关键洞察

**task_418 修复成功的关键**
- v1：陷入"寻找不存在的数据库"循环（25 步超时）
- v3：使用 `extract_patterns` 从文档提取数据（11 步完成）
- **结论**：工具选择比 Prompt 长度更重要

**v3 失败的教训**
- 简单任务被复杂 Prompt 拖累（步骤增加、格式错误增多）
- 新工具必须有明确的使用场景，不能全局启用
- 改进边缘案例不应牺牲整体性能

### v4 架构计划

v4 不再继续扩写单个 ReAct Prompt，而是改为 **Plan-and-Execute + Subagent Roles**：

1. **非 LLM 快速路由**
   - 通过 context 文件类型直接生成任务画像，而不是让模型先猜该看什么
   - `structured_db`：以 SQLite 为主
   - `structured_file`：以 csv/json 为主
   - `doc_heavy`：以 `doc/` 或大段文本为主
   - `mixed`：结构化数据与文档并存

2. **Planner / Executor / Verifier 分层**
   - `planner`：只负责生成 2 到 5 步的短计划，不执行工具
   - `executor`：每次只执行当前子目标，并且只看到该子目标允许使用的工具
   - `verifier`：对最终答案做一次轻量校验，必要时生成 repair goal

3. **工具隔离**
   - `extract_patterns` 只在 `doc_heavy` 和 `mixed` 路由中出现
   - 结构化任务优先 `inspect_sqlite` / `execute_sql` / `read_csv` / `read_json`
   - `execute_python` 作为补充工具，而不是默认主路径

4. **协议收缩**
   - executor 与 verifier 都只返回单个 JSON 对象，不再依赖超长 fenced JSON Prompt
   - parser 对尾部多余的 ```` ``` ```` / `}` 做有限容错，降低格式型回归

5. **验证闭环**
   - v4 首轮采用 canary 验证，而不是直接跑全量
   - 优先关注：
   - 修复组：`task_200`, `task_355`, `task_418`
   - 回归组：`task_173`, `task_250`, `task_269`, `task_344`, `task_352`, `task_396`, `task_408`, `task_420`

### v4 当前实现范围

当前仓库里的 v4 第一版先完成以下内容：

1. 在 `custom_agent.agent` 中落地任务路由、planner、executor、verifier 控制流
2. 保持现有 CLI 与 runner 不变，方便直接复用 `run-task` / `run-benchmark`
3. 将工具描述改为可按路由裁剪，避免简单任务看到无关工具
4. 修正 `execute_python` 的本地解释器调用方式，兼容 Windows 环境

### v4 后续迭代

第一版 v4 还不是终点，后续继续沿这几个方向收敛：

1. 给 planner / verifier 增加更稳定的结构化输出能力
2. 为关键路由补单元测试与 canary 回归脚本
3. 针对 `doc_heavy` 任务补更强的文档摘要与片段检索能力
4. 继续压缩 executor Prompt，只保留与当前子目标直接相关的信息
