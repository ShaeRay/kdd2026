运行步骤
1. 安装依赖
uv sync
2. 配置 API
编辑 configs/react_baseline.yaml，填入你的 API 信息：
agent:
  model: 你的模型名称
  api_base: 你的API地址
  api_key: 你的API密钥
3. 运行单个任务
uv run dabench run-task task_11 --config configs/react_baseline.yaml
4. 运行所有任务
uv run dabench run-benchmark --config configs/react_baseline.yaml
其他命令
# 查看项目状态
uv run dabench status --config configs/react_baseline.yaml
# 查看任务详情
uv run dabench inspect-task task_11 --config configs/react_baseline.yaml  