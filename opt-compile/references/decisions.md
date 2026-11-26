# 编译边界与配置决策

本文用于按具体问题选定 `torch.compile` 边界、配置参数与接入方式（按问题检索见 [跨模型问题参考](compile-problem-patterns.md)）。

优先确认目标项目的 PyTorch 版本与实际平台，有条件时按对应版本官方文档核对支持范围。没有运行环境时可读取项目依赖声明并标记实际版本未确认；离线时按本文继续分析，把无法核对的 API/后端行为列为待验证，不要求下载文档或安装环境。以下均为候选方向，不构成所有版本和硬件的性能承诺。

## 比较边界

| 候选 | 适合考虑的情况 | 必须检查 |
| --- | --- | --- |
| 顶层 nn.Module / forward | 主要计算在模型内部、调用路径可捕获 | train/eval、buffer/随机性、shape、hooks、自定义算子 |
| 单步训练/推理函数 | 希望跨模块、loss 或优化器操作优化 | 排除无关 IO/外层迭代调度，检查 backward 和状态更新支持 |
| 重复子模块/区域 | 全模型编译时间太长，或外围控制流复杂 | 是否重复复用、是否触发重新特化、是否牺牲跨边界融合 |
| 局部 Tensor 函数 | 日志/CPU 调用等包围着独立计算热点 | 拆分是否保持异常、状态、梯度、alias 和执行顺序 |
| eager | 没有热点证据、收益被启动成本吞没或不兼容 | 记录原因；若只是证据不足，标 D 类而非宣称不可编译 |

当已安装版本支持并推荐时，顶层 `nn.Module` 可使用其 `.compile()` 接口；函数可使用 `torch.compile` 包装。不要假设所有版本具有相同接口细节。类对象、模块实例和 forward 方法的编译入口不能直接互换，需核对目标版本对类包装的支持及可能的类状态修改。已有训练框架提供编译入口时优先复用，避免多处包装同一调用链。

## 子系统粒度选择

在异构模型中逐项比较，不把“LLM 逐层、视觉整体”当固定答案：

| 粒度 | 可考虑的条件 | 必须核对 |
| --- | --- | --- |
| 重复层级 | 整体边界捕获不稳定或编译成本高，且重复单元容易独立验证 | 真实层类型、包装层次、递归范围、匹配数量和跨层融合损失 |
| 子系统整体 | 子系统内部路径相对稳定，外围控制/调度不适合一起捕获 | 实际包含哪些模块、输入变化、状态与单元边界 |
| 整模型 | 整体调用链可捕获，跨子系统融合有潜在收益 | 冷启动、图中断、通信/包装及整体回归成本 |

配置开启不等于有模块命中。发现规则需具体到调用时的类与包装结构；无匹配时记录 0 个包装及原因，不能以 setup 成功日志当作已编译证据。配置表至少区分工厂默认、recipe 覆盖、实际运行参数和命中范围。

示例使用 `dynamic=True`、`fullgraph=False` 不代表应覆盖目标项目默认策略；按输入分布、版本和重编译日志选择。类型结构和 FSDP 版本不匹配时先分析，不机械复制私有属性访问或扩大匹配范围。静态实例见 [16 模型编译证据](turbophysai-compile-evidence.md#ev-p1)。

## mode / dynamic / fullgraph 配置表

| 设置 | 使用依据 | 代价或注意点 |
| --- | --- | --- |
| `default` | 初始对照配置 | 不是保证最快，需要实测 |
| `reduce-overhead` | 小 batch、Python/启动开销突出且 CUDA Graph 适用 | 可能增加显存；输入修改、CPU 操作等会影响适用性，检查 perf_hints |
| `max-autotune` | 目标设备支持、矩阵计算占比高且运行足够长 | 增加调优启动成本；CUDA Graph 与内存约束仍需检查 |
| `max-autotune-no-cudagraphs` | 有理由做 autotune 但不使用 CUDA Graph | 不保证一定节省内存或优于 `default` |
| `dynamic` 默认策略 | 先观察代表性输入变化 | 通常从默认策略开始，发现重复编译后再定向调整 |
| `dynamic=True/False` | 输入分布和日志证明需要 | `True` 不能保证无重编译；`False` 可能为新 shape 特化；`dynamic=False` 与显式 `mark_dynamic` 可按轴并存 |
| `fullgraph=True` | 严格捕获要求或最小范围诊断 | 不是速度开关；会改变捕获要求，语义细节需按版本核对 |

不要为了动态 shape 直接 padding/bucketing 而忽略 mask、损失、梯度和输出协议；这类重构需要单独验证。

## 分布式与自定义算子

检查现有框架的 DDP/FSDP/DTensor 流程和版本文档，再决定 inner module、子模块或训练步骤的边界。记录参数 identity、优化器构建、权重加载、包装与编译的顺序；不得仅凭一段通用代码重排现有生命周期。

自定义 CUDA/Triton 算子并非天然不能 compile；检查注册、fake/meta 实现、autograd、mutation/alias 和动态输出 shape 的契约（接入细节交 `opt-kernel` K6）。若是三方扩展不支持，提出有依据的局部 eager 边界，不借机将项目迁移到另一构建或分发平台。

非 Tensor 工作可以保留在外层；不要因“Python 出现在函数中”就否定整个 forward。Tensor 值依赖的 Python 控制流需看实际跟踪行为；`torch.cond` 或张量化方案需满足目标版本限制及原始语义，不能机械替换。

## 阶段拆分、eager 岛、重复层复用与嵌套入口

- **阶段拆分**：优先识别可独立验证的连续阶段（如几何矩阵变换、embedding/位置编码准备、特征格式整理、decoder 预测和匹配代价）。每次拆分记录输入输出、状态、异常、梯度与调用次数；小函数要有调用频率/融合证据，不能只按装饰器数量评价优化程度。
- **eager 岛**：把 SciPy 匹配、NumPy/PIL 增强等保留在 eager 岛时，检查 CPU 往返、结果对象和状态更新成本。按安装版本使用 `torch.compiler.disable` 等受支持入口；项目已有 `torch._dynamo.disable` 只作兼容性线索。禁用区域与 `fullgraph` 要求可能冲突，先明确任务目标。
- **嵌套入口与重复层复用（见 [P7](compile-problem-patterns.md#p7)）**：建立 `parent → child` 编译入口图，区分独立 eager 调用与外层追踪时的内联行为；不能认为“父子都装饰”意味着双重收益。共享 callable、包装缓存与共享图是不同层次——外部查出模块再传参可能减少索引 guards，但模块类型、参数和状态仍能特化。checkpoint 状态容器浅拷贝不隔离 Tensor 原地数据。OpenWAM 的私有 `skip_code` 只作受限版本参考，不默认复制。

## 按需局部 AOT 与状态契约

复杂外层调度包围稳定的重复 Tensor 计算时，可保留切块、循环和拼接，只编译局部计算；状态缓存显式输入输出，并检查首次、过渡、稳态、重置及尾块。列表浅拷贝不隔离其内部 Tensor 数据。先比较普通 `torch.compile`；export/AOTInductor 是另一个按需方案，不是 compile mode，也不保证更快或支持反向。

AOT 需另外核对导出覆盖、示例 batch/dtype/device/layout、权重绑定、变体 key、跨 rank 包可见性及版本/设备兼容性。区分未命中 eager、准备失败、全部加载失败及命中后执行异常；关闭替换 Group 可能恢复原项目已有编译，而非纯 eager。完整实例见 [16 模型编译证据](turbophysai-compile-evidence.md#ev-p2)。

## 精度、重编译、状态与静态化契约

- **先看 `recompiles` guards 再选方案**：先确定变化的 guards，再考虑参数分桶或形状契约。分桶只改变允许取上界的参数，保留真实长度与消费者；`_check` 只表达已有不变量，不删除合法空输入或必要检查。调大缓存（如 `recompile_limit`/`cache_size_limit`）仅容纳更多版本；私有设置缺失时的跳过不能当作已生效，也不能证明重编译减少。
- **装饰器顺序**：装饰器从下到上应用。分别检查 `compile(force_fp32(fn))` 与 `force_fp32(compile(fn))` 实际包装的代码及目标输入 dtype，避免随手换序。
- **控制流与状态**：带静态循环、容器追加、`ModuleList` 索引的计算可作为候选，需验证循环次数、配置分支及 train/eval 改变时的 guards。输入 `squeeze_`、切片赋值、共享 view、缓存首次构建等可能影响重放、alias 或特化，保持原语义并单独覆盖首次和后续调用。
- **padding 与离散决策**：padding 必须记录容量、pad 值、mask、length 及消费者。测试 `N=0, 1, 容量-1, 容量, 容量+1`；明确超限行为，不截断有效数据，不移除原始空输入处理来追求静态图。对匹配/排序等离散决策，除数值容差外检查有效索引、padding 过滤和 ties 行为；元数据转换、numpy 输入和 `.to(device)` 追踪到被调用函数，不以 helper 名称判断“纯 Tensor”。

## 包装式接入与缓存

- **基线验证与计数口径**：先在干净基线验证目标符号存在；已有方法可直接包装，优化版才有的 helper 必须追踪其调用方和输入输出契约再重建边界。不得把 5 个重建调用方与 8 个内部 helper 重复计作 13 个编译区域；源码调用点、目标数量和运行图数量各自统计。
- **prepare vs runtime**：区分包装工厂的 prepare 阶段与 `runtime_condition` 的逐次调用阶段。只做 `torch.compile` 属性可调用性检查不能证明后端能执行；prepare 创建包装也不证明已完成首次图捕获。若包装创建/执行异常未被捕获，如实报告失败，不描述为自动 eager 回退。
- **缓存与开关生命周期**：模块级缓存需核对 key、活动配置与重置时机。按 `(helper, mode)` 缓存时注意全局 active mode 对多配置使用者的影响；单一槽缓存不应误认作多 mode 隔离。Group 先准备后应用时，沿真实调用链确认后续包装是否捕获旧函数；关闭环境变量不自动撤销已返回的包装或已有类状态，需检查 wrapper 重查、engine 回转与 helper lookup。mode/参数按当前来源记录，不借用其他案例的显式参数。实例见 [16 模型编译证据](turbophysai-compile-evidence.md#ev-p2)。

<a id="全局-inductor-环境与厂商后端配置"></a>
## 全局 Inductor 环境与厂商后端配置

Inductor 环境变量与产物目录属于进程级 runtime 配置，不在 `torch.compile` 调用参数内，对照时必须单独记录并与基础 `torch.compile` 消融：

- **语料各模型 `configs/runtime*.yaml` 核实清单**（均属模型调优默认，非通用要求；以下环境变量与缓存目录对编译时长与吞吐的影响均未在本仓复测）：
  - **MapTRv2**：`TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1`、`COORDINATE_DESCENT_TUNING=1`、`MAX_AUTOTUNE_CONV_BACKENDS=ATEN`（配置文件同时提示 `environment.set` 强制写入值，调用者 shell 已导出的其他变量会原样透传，复现时须记录完整生效环境）。
  - **BEVFusion**：`TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1`、`MAX_AUTOTUNE_GEMM_BACKENDS=ATEN`、`CACHE_DIR=/tmp/bevfusion/torchinductor`。
  - **FlashOCC**：`TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1`、`COORDINATE_DESCENT_TUNING=1`。
  - **BEVFormer / Sparse4D**：`TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN`。
  - **LeRobot-ACT**：`TORCHINDUCTOR_COMPILE_THREADS=1`、`CACHE_DIR=/workspace/act_runs/inductor_cache`。
  - **OpenPI**：`TORCHINDUCTOR_ALIGNMENT_ASSERTS=0`、`EMULATE_PRECISION_CASTS=0`、`CACHE_DIR=/dev/shm/openpi_8gpu_cache/inductor`。
  - **OpenWAM**：`runtime.optimized.yaml` 为 `COMPILE_THREADS=1`、`COORDINATE_DESCENT_TUNING=0`；`runtime.online.yaml` 为 `COMPILE_THREADS=1`、`COORDINATE_DESCENT_TUNING=1`。
- **缓存目录与厂商后端边界**：
  - 按模型划分的缓存目录（如 `/tmp`、`/dev/shm`）其跨进程复用安全性需按目标环境验证。
  - 保存 `torch` 构建标识、后端/驱动、设备、容器标签和实际启动命令；不要因厂商实现复用 `torch.cuda` 命名就推断是 NVIDIA CUDA。
  - `max-autotune-no-cudagraphs`、显式 `triton.cudagraphs`/`cudagraph_trees`、布局优化、调优环境变量和通信开关另列配置；私有 options 的存在性与效果由目标构建决定，不硬编码迁移。
  - 厂商平台上的精度/TF32 开关、通信重叠和 NUMA 绑定与 compile 分开归因，不借改变这些条件制造编译收益。
