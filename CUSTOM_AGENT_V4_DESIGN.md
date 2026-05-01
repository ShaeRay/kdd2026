# Custom Agent v4 改造思路

## 1. 背景

这次 v4 改造，不是一次单纯的 Prompt 微调，而是一次 **agent 控制流的重构**。

问题的起点很明确：

- v1 的整体成功率高，但 `task_418` 这类以文档为主的数据任务处理不好。
- v3 为了修 `task_418`，引入了 `extract_patterns`、更多规则、更长的 Prompt。
- 结果是复杂任务有所改善，但简单任务开始回归，出现了更多格式错误、超步、工具误用。

从 `artifacts/runs/custom_v1` 和 `artifacts/runs/custom_v3` 的对比可以抽象出两个结论：

1. **复杂任务需要更明确的规划与工具选择路径**。
2. **不能继续用一个超长的统一 Prompt 同时负责规划、执行、纠错、验证**。

所以 v4 的核心目标不是“再写一个更聪明的 ReAct Prompt”，而是：

- 把任务先做路由
- 把规划和执行拆开
- 把验证单独拿出来
- 把工具暴露范围缩小

这就是本轮 v4 的设计出发点。

## 2. 设计目标

### 2.1 目标一：保住 v1 的简单任务稳定性

v1 最大的优点是路径短、心智负担小、模型不容易被复杂规则干扰。

因此 v4 的第一原则不是“能力扩张”，而是：

- 让简单任务继续走简单路径
- 不让文档抽取工具污染结构化任务
- 不让模型在每一轮都重新阅读一大段全局规则

### 2.2 目标二：给文档型复杂任务单独通道

`task_418` 暴露的问题不是模型完全不会推理，而是它不知道：

- 当前到底应该把什么当作主数据源
- 什么时候该读 `knowledge.md`
- 什么时候该读 `Patient.md`
- 什么时候该用 `extract_patterns`
- 什么时候该停下并汇总

所以复杂任务需要一条专门路径，而不是给所有任务一起增压。

### 2.3 目标三：降低协议层脆弱性

v3 失败里有一大类不是业务错误，而是协议错误：

- JSON 末尾多 `}`
- JSON 后面多了 ```` ``` ````
- `execute_python` 的 `code` 字段多行导致 JSON 失效

这些问题说明系统并不是只缺“更强推理”，而是缺“更稳的协议层”。

因此 v4 里同步做了：

- 响应解析容错
- 多角色 Prompt 收缩
- `execute_python` 路径兼容修正

## 3. 总体架构

v4 采用的是一个 **轻量级 Plan-and-Execute 架构**。

注意，这里的 “planner / executor / verifier” 是 **角色分层**，不是多进程并行的真实外部 agent，也不是比赛环境之外的额外系统。实现上仍然是当前 `custom_agent` 里的单一 agent 内核，只是把单轮 ReAct 拆成多个角色阶段。

总体流程如下：

1. **Task Router**
   - 先根据 context 文件类型给任务打画像
2. **Planner**
   - 生成 2 到 5 步的短计划
3. **Executor**
   - 逐步执行当前计划步骤
   - 每一步只暴露允许的工具
4. **Verifier**
   - 对最终答案做一次轻量校验
   - 如果失败，给出 repair goal
5. **Repair Step（可选）**
   - 按 verifier 的反馈做一次修补

这个流程的关键点不是“层数更多”，而是：

- 每一层职责更单一
- 每一层看到的信息更少
- 每一层的失败影响范围更小

## 4. 代码层面的主要改动

## 4.1 `src/custom_agent/agent.py`

这是本轮改动最大的文件。

### 4.1.1 新增任务画像 `TaskProfile`

位置：

- `infer_task_profile()` at `src/custom_agent/agent.py:135`

这里做了一个 **非 LLM 的任务路由器**，直接根据 context 里的文件类型判断任务模式：

- `structured_db`
- `structured_file`
- `doc_heavy`
- `mixed`

这样做的原因：

- 文件类型是稳定信号，不需要让模型重复猜测
- 路由越早做，后面的工具暴露越容易控制
- 文档型任务与数据库任务天然是两条不同路径

### 4.1.2 新增计划结构与验证结构

在 `agent.py` 里引入了：

- `ExecutionPlanStep`
- `ExecutionPlan`
- `VerificationResult`

这些结构的作用是把 planner/executor/verifier 的输出固定下来，避免继续把所有状态都塞在松散文本里。

### 4.1.3 新增更稳的 JSON 解析链

关键函数：

- `load_single_json_payload()` at `src/custom_agent/agent.py:258`
- `parse_model_response()` at `src/custom_agent/agent.py:278`
- `parse_execution_plan()` at `src/custom_agent/agent.py:308`
- `parse_verification_result()` at `src/custom_agent/agent.py:350`

这一层解决的是 v3 的协议问题，而不是业务问题。

改造重点：

- 允许去掉 code fence 后再解析
- 对尾部多余的 ```` ``` ```` 和 `}` 做有限容错
- 对 `execute_python` 的 `code` 字段多行进行修复
- planner / verifier 与 executor 各自走不同的解析入口

这一步非常关键，因为如果协议层不稳，planner 再聪明也会被格式错误打断。

### 4.1.4 新增 fallback plan

函数位置：

- `build_default_plan()` at `src/custom_agent/agent.py:377`

Planner 本身也可能失败，所以不能让计划层成为单点故障。

因此实现上做了 fallback：

- 如果 planner 输出不可解析
- 或者步骤不合法

就根据任务模式生成一个默认计划。

这样可以保证：

- planner 失败不至于让整个任务直接崩掉
- 结构化任务仍然能继续走基本路径

### 4.1.5 重写 `CustomAgent.run()`

类位置：

- `CustomAgent` at `src/custom_agent/agent.py:475`

这是整轮改造的核心。

旧版是：

- 单一 ReAct 循环
- 每一步重新把所有历史和所有规则喂给模型

新版变成：

1. 先路由
2. 再规划
3. 然后按 step 执行
4. 最后 verifier 检查

执行层的几个关键设计：

- 每个 step 只暴露允许的工具
- `__step_done__` 作为子任务完成信号
- verifier 不通过时，插入 repair step
- repair 不是无限循环，只允许有限次数

这样做的本质是：

- 缩小每一轮决策空间
- 降低工具误用概率
- 让“完成一个子目标”先于“立即给最终答案”

## 4.2 `src/custom_agent/prompt.py`

这个文件从“一个大 Prompt”拆成了“多角色短 Prompt”。

关键函数：

- `build_planner_system_prompt()` at `src/custom_agent/prompt.py:21`
- `build_planner_user_prompt()` at `src/custom_agent/prompt.py:53`
- `build_executor_system_prompt()` at `src/custom_agent/prompt.py:72`
- `build_executor_user_prompt()` at `src/custom_agent/prompt.py:97`
- `build_verifier_system_prompt()` at `src/custom_agent/prompt.py:138`
- `build_verifier_user_prompt()` at `src/custom_agent/prompt.py:160`

### 4.2.1 为什么拆 Prompt

v3 的问题不只是内容多，而是“职责混杂”：

- 一边要求严格 JSON
- 一边教工具怎么用
- 一边讲特殊业务规则
- 一边要求最终答案格式

模型每一轮都要同时处理这些东西，极易分心。

所以 v4 拆 Prompt 的核心逻辑是：

- planner 只负责生成计划
- executor 只负责下一步动作
- verifier 只负责审答案

### 4.2.2 Prompt 收缩策略

executor prompt 里只保留：

- 当前问题
- 当前任务画像
- 当前计划步骤
- 已完成步骤摘要
- 当前允许的工具
- 精确 context 文件列表

不再全量灌入复杂业务长规则。

这个变化的目标是：

- 缩短上下文
- 提升当前步骤的专注度
- 减少模型自己“发散找路”的概率

## 4.3 `src/custom_agent/tools.py`

虽然工具主体没有全面重写，但做了两处关键调整。

### 4.3.1 修复 `execute_python` 的解释器调用

位置：

- `tool_execute_python()` at `src/custom_agent/tools.py:285`

之前调用的是：

- `python3`

这在 Windows 环境下不稳定，容易直接失败。

现在改成：

- `sys.executable`

好处是：

- 始终使用当前 `uv` 环境里的解释器
- 本地与 benchmark 命令路径更一致
- 避免平台差异导致的假失败

### 4.3.2 工具描述支持按路由裁剪

位置：

- `describe_tools()` at `src/custom_agent/tools.py:591`

旧版所有任务都看到全部工具。

新版可以按 `tool_names` 传入需要暴露的工具列表。

这是 v4 “工具隔离” 的基础设施。

没有这层裁剪，前面做的任务路由就会失去一半价值。

## 4.4 `src/custom_agent/runner.py`

这个文件改动不大，但改动很关键。

位置：

- `build_model_adapter()` at `src/custom_agent/runner.py:125`

新增逻辑：

- 优先读取 `MODEL_NAME`
- 优先读取 `MODEL_API_URL`
- 优先读取 `MODEL_API_KEY`

如果环境变量不存在，再回退到配置文件。

这样做有两个原因：

1. **比赛规则兼容**
   - 比赛环境通过环境变量注入模型配置
2. **本地开发兼容**
   - 本地仍然可以保留配置文件兜底

## 4.5 `configs/custom_agent.yaml`

这里不是 agent 核心逻辑，但和执行路径直接相关。

本轮做了两个修正：

- 清掉硬编码的模型地址和 API key
- 去掉固定 `run_id`

原因：

- 硬编码密钥不适合提交，也不符合比赛环境设计
- 固定 `run_id` 会导致 benchmark 重复运行时撞目录失败

现在更合理的使用方式是：

```powershell
$env:MODEL_NAME="..."
$env:MODEL_API_URL="..."
$env:MODEL_API_KEY="..."
uv run custom-agent run-benchmark --config configs/custom_agent.yaml
```

## 4.6 `tests/test_custom_agent_v4.py`

新增了几条最小回归测试，用来守住这轮最容易再次坏掉的地方。

覆盖内容：

1. JSON 解析能够容忍尾部 fence / brace 噪音
2. `execute_python` 的多行 `code` 字段能被修复
3. `doc_heavy` 路由正确
4. `mixed` 路由正确

这些测试不是全面测试，但它们至少守住了 v4 的几个基本支点。

## 5. 本轮改造中的关键设计判断

## 5.1 不做真实并行子代理

虽然一开始的方向里有 “subagent” 的讨论，但最终代码没有做成真实并行 worker。

原因有三个：

1. 比赛环境更需要稳定，而不是复杂并行
2. 当前问题的主要瓶颈在控制流与协议层，不在吞吐量
3. 多真实子代理会引入更多上下文同步与状态管理问题

所以本轮采取的是：

- **角色级 subagent**
- 不是 **系统级多 agent**

这让结构更清晰，但风险更可控。

## 5.2 先做工具隔离，再做更强抽取

v3 的教训说明：

- 工具越多，不代表结果越好
- 无限制地把新工具暴露给所有任务，反而更容易破坏整体稳定性

所以 v4 先解决的是：

- 让什么任务看见什么工具

而不是：

- 再堆一堆新工具

## 5.3 verifier 是保护机制，不是终点答案来源

verifier 的作用不是“替 executor 再做一遍任务”，而是：

- 检查明显的逻辑漏洞
- 检查答案结构是否满足问题
- 提供一次 repair goal

这种定位能让 verifier：

- 足够便宜
- 足够聚焦
- 不至于再变成一个新的大型 agent

## 6. 本轮验证情况

### 6.1 静态验证

已经通过：

- `uv run ruff check src\custom_agent tests`
- `uv run pytest tests\test_custom_agent_v4.py`
- `uv run custom-agent --help`
- `uv run custom-agent run-benchmark --help`

### 6.2 benchmark 路径验证

已经验证以下命令仍然可启动：

```powershell
uv run custom-agent run-benchmark --config configs/custom_agent.yaml --limit 1
```

这说明：

- CLI 入口没有被 v4 改坏
- runner 到 agent 的接线仍然成立
- 输出目录和 `summary.json` 写入路径仍然成立

### 6.3 真实任务烟雾测试

围绕 `task_418` 做了多次 smoke run。

结果说明：

- v4 的 planner/executor/verifier 链路已经打通
- verifier 能抓住执行层给出的不可靠答案
- 目前最大的剩余问题不再是“完全不会走文档路径”，而是“文档型信息抽取还不够稳”

这是一个积极信号，因为说明问题已经从“架构层混乱”收缩到“具体抽取策略不够强”。

## 7. 当前还没有彻底解决的问题

## 7.1 `doc_heavy` 任务仍然是主要短板

现在 v4 对文档任务已经比原来的单回路结构清晰很多，但还存在几个问题：

- 年龄字段的叙事抽取不稳定
- 文档中同一字段可能有多个数字，容易匹配错
- `extract_patterns` 适合 preview，不一定适合最终精确抽取

也就是说：

- 架构问题初步缓解了
- 业务抽取问题还需要继续打磨

## 7.2 当前 `max_steps=20` 可能仍然偏紧

配置里现在仍是：

- `max_steps: 20`

这会影响两类任务：

- 路径本来就长的复杂任务
- repair 需要回合数的任务

后续要区分：

- 是架构真的没收敛
- 还是步数预算过紧

这两个问题不能混在一起看。

## 7.3 verifier 目前只做一次修补

当前设计是保守的：

- `max_repair_attempts` 默认有限

这是为了防止系统进入自我修补死循环。

但副作用是：

- 某些任务可能离正确答案只差一次额外补救，却在第一轮修补后仍然失败

后续可以考虑：

- 对结构化任务保持保守 repair
- 对文档型任务允许更细粒度的二次修补

## 8. 如果继续迭代，我建议下一步怎么做

### 8.1 优先继续打 `task_418` 这条文档链

建议不要立刻全量 benchmark，而是先打 canary：

- `task_200`
- `task_355`
- `task_418`
- `task_173`
- `task_344`
- `task_352`
- `task_396`
- `task_420`

原因是这组任务已经覆盖了：

- 格式稳定性
- 结构化任务
- 文档型任务
- 超步问题

### 8.2 给 `doc_heavy` 单独补“候选片段定位”层

当前 `doc_heavy` 的最大问题是直接抽字段。

更好的思路可能是：

1. 先定位候选片段
2. 再在小片段内做精确抽取
3. 最后做 patient-level merge

这样会比“对全文直接抽 age / creatinine”更稳。

### 8.3 让 verifier 更偏规则，不偏重新推理

verifier 最好越来越像：

- 校验器
- 结构审计器
- 过滤条件检查器

而不是另一个模糊推理 agent。

这样能避免 verifier 反过来把系统重新拉回“大 Prompt 多职责”的老路。

## 9. 总结

这次 v4 改造的本质，不是“把 Prompt 改了一版”，而是：

- 把 `custom_agent` 从单回路 ReAct
- 改造成了一个带任务路由、计划、执行、验证的分层 agent

它解决的是 **结构问题**：

- 简单任务和复杂任务不再共享同一套重型控制逻辑
- 工具不再无差别暴露
- 最终答案不再完全依赖 executor 一步到位

它暂时还没有彻底解决的是 **文档型抽取问题**：

- 特别是 `task_418` 这类 narrative-heavy 任务，仍需要更稳的片段定位与字段抽取策略

所以这次改造可以理解成：

- v4 的“骨架”已经成型
- 现在进入的是“沿着正确骨架补强能力”的阶段

这比继续在 v3 的超长 Prompt 上打补丁，要健康得多，也更有迭代价值。
