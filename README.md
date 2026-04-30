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

### 下一步架构设计

基于教训，下一版本将采用 **Plan-and-Execute 架构**：

1. **快速评估阶段**：基于文件类型选择策略（非 LLM 判断）
   - 结构化数据 (csv/db/json) → 简单模式
   - 文档数据 (doc/) → 复杂模式（启用文档解析工具）

2. **精简 Prompt**：回退到 v1 的简洁风格

3. **工具隔离**：新工具仅在复杂模式下可用

4. **验证策略**：任何改进必须通过完整回归测试
