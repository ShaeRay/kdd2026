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
