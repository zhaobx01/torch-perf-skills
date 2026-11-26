# 按问题查阅跨模型编译经验（P1–P7）

先从当前项目的源码、配置与实际数据确定问题，再按需查本表。模型名只是来源标签，不是选择方案的依据。没有性能数据时可给出候选与取证计划，不把静态相似性当作瓶颈或收益证明；证据不足时允许暂缓 compile。所有跨模型 `corpus/all-models` 分支链接与核查 SHA 索引统一见 [TurboPhysAI 16 模型编译证据](turbophysai-compile-evidence.md)。

| 当前问题 | 查阅位置 | 核心判断 |
| --- | --- | --- |
| 不知道编译范围怎么选 | [P1](#p1) | 比较范围与对象，而不是按热点机械添加装饰器 |
| 外层流程复杂、内部计算重复 | [P2](#p2) | 保留调度，必要时抽出计算与状态契约 |
| 输入变化导致重复编译 | [P3](#p3) | 先定位 guards，再比较动态策略、分桶或契约 |
| 图中断、捕获障碍或局部 eager 边界 | [P4](#p4) | 按原因和实际影响选择最小处理，低影响项记录后暂缓 |
| 希望低侵入接入、隔离配置或做全局 Inductor 调优 | [P5](#p5) | 集中接入，明确开关生命周期及真实关闭路径 |
| 配置已开启，但不确定是否有效 | [P6](#p6) | 配置、命中、捕获、正确性、收益五层分别验证 |
| 重复层或内外层分别编译，复用不稳定 | [P7](#p7) | 区分共享包装与共享图，检查闭包/分派及缓存生命周期 |

---

<a id="p1"></a>
## P1 不知道编译范围怎么选

- **证据状态**：已核实改造案例（`lingbot_vla_v2`、`openvla`、`bevformer`、`bevfusion`、`maptrv2`、`openwam`、`flashocc`、`lerobot_act`、`dreamzero`、`sparse4d`、`pointpillars`）
- **现象与所需证据**：候选包括整模型、重复层、特征阶段或小函数；需要实际调用链、类型及包装关系、调用频次、热点与形状分布。没有这些证据时先列候选，不宣称最优粒度。
- **可选处理**：比较能够稳定捕获的较高层边界，再因外围控制流、准备成本、重复特化或内存约束缩小。异构多模态可按子系统分粒度（如 LLM 逐层 + 视觉整塔，或 Vision/Text/Expert/Teacher 四分区独立开关）；核对编译对象是类、实例还是方法，以及父子包装是否重叠。不要穷举所有组合。
- **不适用条件**：小函数不一定值得编译，单层不一定优于整体；类型不匹配、第三方模块未注册必要编译支持或无计算热点时，不照搬案例。
- **代表实证**：见 [编译范围与子系统粒度](turbophysai-compile-evidence.md#ev-p1)。
- **最小验证**：选一个首选及必要备选，记录匹配对象/数量、正确性、编译准备成本、稳态与端到端指标；没有足够证据可以暂缓。边界比较规则见 [决策参考](decisions.md#比较边界)。

<a id="p2"></a>
## P2 外层流程复杂、内部计算重复

- **证据状态**：已核实改造案例（`maptrv2`、`cosmos3`、`openwam`）
- **现象与所需证据**：日志、框架调度、切块循环或 CPU 操作包围重复 Tensor 热点；需知道调用方、状态读写、alias、异常、随机性和梯度契约。
- **可选处理**：保留外层控制，必要时提取最小 helper；将状态缓存作为明确输入输出。在原项目缺少 helper 时，先追踪调用方重建契约，不能仅注册基线不存在的函数名。普通 `torch.compile` 与局部 AOTInductor 分开评估。
- **不适用条件**：拆分破坏融合、核心不够重、状态无法保持或无适用导出环境时不强拆；局部 AOT 的无梯度案例不能证明训练 backward 可用。
- **代表实证**：见 [复杂外层、Helper 重建与局部 AOT](turbophysai-compile-evidence.md#ev-p2)。
- **最小验证**：原实现／重构后 eager／相同重构后 compiled 三组对照，覆盖首次和连续调用、状态重置、尾块及失败路径；浅拷贝列表不隔离 Tensor 数据。

<a id="p3"></a>
## P3 输入变化导致重复编译

- **证据状态**：已核实改造案例（`groot`、`openvla`、`bevformer`、`maptrv2`、`cosmos3`、`openwam`）
- **现象与所需证据**：日志显示 shape、Python 值或其他 guards 变化；需要 `recompiles` 日志、输入长度/尺寸分布、变化字段与其消费者。只有 `recompiles` 日志不代表已证明图中断。
- **可选处理**：按目标版本比较动态形状策略、允许上界的参数分桶（优先在数据预取侧 CPU 分桶避免 GPU 同步）、保留真实有效长度的定长 padding（注意将受宽度影响的 Dropout 等随机算子放到真实宽度下执行，如 GR00T `CollatorWidthDropout`），以及已成立的形状约束。
- **不适用条件**：无重编译证据时不要预先固定所有尺寸。不能截断有效数据、删除合法空输入、把容量当有效长度或用抬高 `recompile_limit`/`cache_size_limit` 替代根因排查；静态 padding 与 DDP `static_graph=True` 的联动契约须显式核对。
- **代表实证**：见 [重编译治理、分桶与定长 Padding 契约](turbophysai-compile-evidence.md#ev-p3)。
- **最小验证**：测零、桶边界两侧、上限与超限，检查有效元素映射、mask、dropout 语义与输出/梯度；记录预热后重编译、版本数及显存，前后使用相同输入分布。

<a id="p4"></a>
## P4 图中断诊断、修复与局部 eager 边界

- **证据状态**：已核实改造案例（`dit4dit`、`groot`、`maptrv2`、`openvla`、`bevfusion`、`openwam`）
- **现象与所需证据**：结合源码、配置和上游 `trace-bottleneck-report` 识别热点，再从 `TORCH_LOGS=graph_breaks`、异常栈或已有 `tlparse` 报告定位原因。无日志时允许静态风险分析，但实际断图与影响标为待验证。
- **先分类与筛选**：图中断按本节处理；shape/guard 变化转 [P3](#p3)，重复层/嵌套入口关联 [P7](#p7)，普通代码错误先调试。按实际影响、证据可信度与修改成本排序，低影响或高成本项暂缓，阻断目标捕获或影响正确性的项单独处理。
- **三类核心处理**（按原因择一，不固定先后顺序）：

| 核心方法 | 适用证据与例子 | 直接处理要求 | 限制与验证重点 |
| --- | --- | --- | --- |
| 调整编译边界 | 外围调度、输入整理或 IO 复杂，内部有稳定计算热点 | 保留外围行为，通过接入点/配置缩小至内部核心 | 检查调用顺序、状态、实际捕获区域、融合损失与端到端收益 |
| 保持语义的 Tensor / Dynamo-safe 改写 | 日志确认标量提取、只读 NumPy 视图或 Dynamo 内部限制（如 `set(n)`）导致断图 | 保留 Tensor 计算或改写等价容器；前置检查选项存在性（如 `fallback_random`） | 不机械用 `torch.where` 替 `if`；覆盖分支、空输入、梯度与随机性 |
| 局部 eager 隔离 | 外部库调用、CPU 求解、必要日志或改写风险高 | 用 `@torch.compiler.disable` 隔离最小区域（注意同步替换全局注册表如 `AttentionInterface._global_mapping`） | 检查递归禁用范围、搬运/同步成本及与 `fullgraph` 的冲突；外部 kernel 需进图时交 `optimize-kernel-replacement`（K6 `custom_op`） |

- **不适用 / 禁区**：无具体原因时不批量 `disable`；不删除必要日志/检查、不放宽数值容差、不开 `suppress_errors` 掩盖失败。
- **代表实证**：见 [图中断、Dynamo-Safe 改写与局部 Eager](turbophysai-compile-evidence.md#ev-p4)。
- **最小验证**：分别验收 ① 正确性（输出/loss/梯度/状态）、② 捕获（日志/Trace 确认边界与剩余断图）、③ 关闭 profiler 与诊断日志后的正式稳态性能与峰值显存。

<a id="p5"></a>
## P5 低侵入接入、作用域隔离与全局 Inductor 调优

- **证据状态**：已核实改造案例（`openvla`、`maptrv2`、`bevformer`、`bevfusion`、`cosmos3`、`openwam`；`pointpillars` `optimize_ddp` 属实现参考，基线待核实；`lingbot_vla_v2` `hip_triton_configs` 属实现参考，基线待核实｜受限参考｜版本兼容性待核对）
- **现象与所需证据**：需要快速切换候选或回退、与 DDP/FSDP 补丁组合、或评估全局 Inductor 环境变量与启发式配置；需核清生命周期入口、prepare/runtime 条件、缓存及真实返回对象。
- **可选处理**：
  - 在初始化入口集中接入，支持总开关与单项/组合消融；涉及 `torch._dynamo.config` 修改（如 `optimize_ddp=False`）时优先用 `with torch._dynamo.config.patch(...):` 作用域隔离；与 FSDP 补丁组合时动态查找类（避免模块级早绑定丢补丁），保存 checkpoint 时按需清洗 `._orig_mod.`。
  - **全局 Inductor 调优**：`TORCHINDUCTOR_LAYOUT_OPTIMIZATION`、`COORDINATE_DESCENT_TUNING`、`MAX_AUTOTUNE_*_BACKENDS` 及 HIP Triton 启发式候选集补充（如 `hip_triton_configs`：按 `torch.version.hip` 过滤、核对 `fn.arg_names` 并去重前插、保留原 configs 兜底）必须与基础 `torch.compile` 分开消融。
- **不适用条件**：不默认全局 monkey patch；配置 Group、环境变量、显式 `disable` 与类状态不是同一个开关；关闭后恢复原实现、重构后 eager 或原项目已有 compile 也不是同一对照。
- **代表实证**：见 [接入顺序、Checkpoint Key、作用域隔离与全局 Inductor 调优](turbophysai-compile-evidence.md#ev-p5)。
- **最小验证**：覆盖首次关闭/开启、切 mode、异常回退与非目标平台零侵入；对全局 Inductor 启发式同时记录首次编译/autotune 增幅与稳态收益。

<a id="p6"></a>
## P6 配置已开启，但不确定是否有效

- **证据状态**：已核实改造案例（`bevfusion`、`openvla`、`cosmos3`、`maptrv2`、`openwam`）
- **现象与所需证据**：只有配置、setup 日志或包装成功；还需从调用点反查实际符号/实例命中、执行路径、编译诊断、正确性和公平测速。
- **可选处理**：按“**配置开启 → 匹配/包装 → 运行实际捕获 → 正确性 → 稳态性能改善**”五层分别记录，每层都可失败或未知；有显式 eager 或零命中（例如注册了别名但调用方实际调了同文件 CPU 实现）时如实报告。
- **不适用条件**：不把 API 存在、fullgraph 成功或图数减少当成加速；无运行时不补造命中率与提速百分比。
- **代表实证**：见 [注册≠运行命中、重复层复用与闭包缓存](turbophysai-compile-evidence.md#ev-p67)。
- **最小验证**：日志/Trace 确认实际命中，正式测速关闭 profiler 与诊断日志，按 [验证规范](validation.md#分开测量成本) 对比冷启动、稳态与峰值显存。

<a id="p7"></a>
## P7 重复层复用与嵌套编译

- **证据状态**：已核实改造案例（`openpi`、`openwam`、`openvla`、`maptrv2`）
- **现象与所需证据**：形状相近的重复层反复特化、嵌套闭包每步重建包装、或外围已编译而内部仍创建包装；需实际 guards/recompiles、调用栈、编译对象与缓存归属。
- **可选处理**：外部查出模块显式传入；对嵌套 checkpoint 闭包按 `(function.__code__, mode, direct)` 缓存在 owner 模型上；外层已捕获（`torch.compiler.is_compiling()`）时内部小函数直接 eager 执行。保留 SAC 等优先分派条件，不以“一律取消内层编译”替代具体分支。
- **不适用条件**：不同模块类型、参数属性、状态和 guards 仍可能生成不同图；共享 callable 不证明只编译一次；私有 `skip_code` 只作版本受限参考。
- **代表实证**：见 [注册≠运行命中、重复层复用与闭包缓存](turbophysai-compile-evidence.md#ev-p67)。
- **最小验证**：分别测试同对象/多对象、多层类型、不同形状、外层开关及缓存生命周期；记录包装创建次数、重编译次数、正确性与显存。
