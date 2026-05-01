"""
Custom Agent 模块

这是一个自定义的 ReAct 风格数据代理，用于 DABench 基准测试。
相比 baseline 版本，主要改进包括：
1. 优化的提示词模板，强调输出格式和工具使用
2. 支持 FINAL_ANSWER 机制，解决大量数据提交时的 JSON 截断问题
3. 增强的错误恢复机制
"""

__all__ = ["__version__"]

__version__ = "0.2.0"
