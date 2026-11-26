# 分布式通信优化模式（C1–C8）

先用 [分流与 RCCL 防护](triage-and-rccl.md) 排查慢 rank。无 Trace 时可以核查配置、调用链和版本，将候选标为静态。案例证据和运行条件见 [TurboPhysAI 双侧证据](turbophysai-comm-evidence.md)。以下证据状态描述源码核查范围，运行结果按 [验收规范](validation.md) 单独验证。

## 一级 · 公开 API、配置和业务集合通信

### C1 DDP reducer 和 buffer 配置 `[已核实改造案例]`

- **Before 信号** `find_unused_parameters=True` 触发额外遍历、静态图只在部分卡数启用、梯度桶有重复复制，或前向广播不需要同步的 buffer。OpenPI Before 已启用 `gradient_as_bucket_view`，LeRobot-ACT 的 `static_graph` 由选项决定。
- **改法与前提** 在普通动态 reducer 路径，确认所有训练参数在代表性步里都参与反向后可关闭 unused 搜索。图训练方式与 used/unused 集合全程固定时可评估 `static_graph=True`，包括固定存在 unused 参数的图。验证条件分支、梯度累积和冻结策略。独立评估 `gradient_as_bucket_view`，核对梯度原地操作（如 `detach_()`）与 `zero_grad` 行为。关闭 buffer 广播前核对各 rank 是否依赖 rank 0 的 BN running state 或其他动态 buffer。
- **编译交接** PointPillars 的 epoch 级 `torch._dynamo.config.patch(optimize_ddp=False)` 属「实现参考，基线待核实」。若主要问题是 DDP bucket 导致的编译切图，交编译专项。通信侧独立测 backward overlap，关闭切分可能降低重叠。
- **最小验证** 用代表性分支做多步梯度/参数对照。按改动核对图遍历、梯度复制或 buffer 广播的变化。出现 unused 参数或 buffer 漂移时，重建原 DDP 包装。

### C2 FSDP1 预取与 AllGather 限流 `[已核实改造案例]`

- **Before 信号** 固定层序且 Trace 显示 GPU 等 AllGather，或调用点仍使用 `limit_all_gathers=True`、未显式开启前向预取。CPU 限流间隙本身不足以证明 GPU 饥饿，`BACKWARD_PRE` 在部分版本已是默认值。
- **改法与前提** 分别评估 `forward_prefetch`、`backward_prefetch`、`limit_all_gathers`。预取和放宽限流均需显存余量，变更后测峰值，必要时恢复限流或尝试 `BACKWARD_POST`。动态层序保留原前向预取策略。
- **调用点契约** OpenVLA 的包装用同名、同模块子类与 `setdefault`。原构造显式传 `limit_all_gathers=True` 时保留该值。已核实的组合路径由 `compile_fsdp1` 在调用时解析已包装的类并省略该参数，才注入 `False`。核对已导入别名、参数优先级和最终构造值。
- **最小验证** 打印实际包装和参数，比较各 rank 峰值与 T5 暴露/重叠，并做多卡 loss 和梯度对照。

### C3 权重常驻与 FSDP2 分片单元粒度 `[已核实改造案例]`

- **权重常驻** 显存有余量、反向重复 AllGather 暴露时，评估 FSDP2 子单元 `reshard_after_forward=False`，LingBot-VA 有双侧证据。根和子单元的默认语义按目标版本和实际层级核对。FSDP1 对照为 OpenVLA 启动配置从 `fsdp-full-shard` 切到 `fsdp-shard-grad-op`，基线实现分别映射到 `HYBRID_SHARD` 和 `_HYBRID_SHARD_ZERO2`。核对分片组、复制组及目标版本的枚举。
- **合并细碎单元** 在连续、同类、调用顺序相容的块间用 `fully_shard(list_of_blocks)` 共享通信单元。Cosmos3 通过 `fsdp_layers_per_group` 控制，默认为 1。列表构造和自定义 collective 接口均按目标 PyTorch 构建核对。分组越粗，完整权重可能同时驻留更多，通信发射粒度也会改变。
- **拆分超大根单元** 若根单元聚合 embedding 等大组件导致尾部归约缓冲峰值，先单独 wrap 大组件，再 wrap root。LingBot-VLA-V2 的示例是 `embed_tokens`。确认共享/绑定权重与单元执行顺序仍正确。
- **最小验证** 检查单元参数量、真实 AllGather/ReduceScatter 次数、窗口暴露、峰值显存和 checkpoint 往返。显存不足或行为不等价时恢复原分片策略或单元布局。

### C4 ZeRO-2 标准配置 `[已核实改造案例]`

- **Before 信号** 实际 `zero_optimization.stage == 2`，`overlap_comm` 被关闭，或 `contiguous_gradients`、bucket 大小需要核对。DreamZero 的对应配置只把 overlap 从 `false` 改为 `true`，GR00T 基线已开启 overlap 和 contiguous。
- **改法与前提** 区分原文件、启动参数、运行时合并后的配置。按层/桶体量分别调 `reduce_bucket_size` 和 `allgather_bucket_size`，两者以元素计。前者影响反向归约发射，后者影响参数收集分片与开销。显存紧张时核对双缓冲和临时缓冲峰值。步内吞吐计时器同步交 `optimize-sync-elimination` P6。
- **最小验证** 确认实际 ZeRO-2 运行配置、目标归约流和 T5 改善。开关为真但重叠仍低时，先判断慢 rank、再查目标版本的等待点。

### C5 Owner 分片计算与双向 `all_to_all_single` `[已核实改造案例。受限单模型参考]`

- **Before 信号** 每个 rank 全量 gather 同一批独立矩阵、重复执行相同更新，再各自保留本地分片。LingBot-VLA-V2 的 Muon 有双侧证据。
- **改法与原子契约** 调用者先统一决定 owner 路径。按确定性 owner 区间打包输入，第一轮 `all_to_all_single` 送给计算 owner，第二轮交换把结果发回参数 shard。成组启停和回退调用者分支、owner/本地 shard 区间、split sizes、打包与回传。Muon 的实现涵盖 `muon.py`、`muon_gather.py`、`muon_owner_ns.py`。
- **前提与回退** 各矩阵计算独立，rank 间有一致的参数顺序、dtype 和分片计划。用 split sizes 或 padding 覆盖非整除、空 shard、矩阵数小于 rank 数。Muon 在形状或代价门槛不满足时走全量 gather，部分小批次补齐到 rank 数，收发辅助函数拒绝零长度切分。将 padding、回传及拓扑纳入流量评估。
- **最小验证** 对照每个 shard 的参数更新，覆盖非整除、空 shard 和小批次回退，并验证各 rank 命中同一分支。融合 Newton–Schulz 的速度由算子专项归因。

## 二级 · ZeRO-2 私有框架改写

先验证相关且适用的一级手段。仅在多 rank Trace 给出仍可回收的暴露证据，并获用户二次明确授权后实施。新实施须校验目标 `torch`、`deepspeed` 构建、关键方法签名与源码 AST hash。区分安装前不兼容、运行时整组回退和异步工作已部分提交的失败。源码案例的已有检查见证据文档。

### C6 双缓冲事件调度 `[已核实改造案例。强版本绑定参考]`

- **Before 信号** DeepSpeed v0.18.4 在 contiguous 和 overlap 均启用、且 accelerator 不自行处理依赖时，满桶发射后会在两条流之间双向等待。只有命中该分支并观察到反向被阻断时，才考虑替换。
- **共同契约** 归约流先取得生产依赖，完成对半缓冲的最后一次读写后记录事件，计算流仅在下次覆写该半缓冲前等待该事件。GR00T 用 `BucketEventScheduler`，OpenWAM 在 owner 归约安装器里对还含 `copy_grads_in_partition` 的工作记事件。
- **前提与回退** 双缓冲、同步点和桶索引复位必须覆盖所有入口。核对 CPU offload、MoE、通信 dtype、设备依赖语义及累积边界。GR00T 的构建号、签名、AST 和 rank 布局门禁均需满足。兼容性失败时整组拒装或走原路径。
- **最小验证** 门禁拒绝、rank 分歧拒绝、连续复用两半的事件顺序，随后在代表性累积条件下比对梯度/更新、多卡稳定性与 T5。

### C7 Owner 归约与 staging 复用 `[已核实改造案例。强版本绑定参考]`

- **Before 信号** 上游多 owner 路径以桶内切片顺序做 AllReduce 后保留 owner 段。直接把偏斜桶一律改成 ReduceScatter 会引入 padding，且临时打包缓冲会增加内存压力。
- **共同契约** 各 rank 以一致顺序生成 owner、偏移、长度和 dtype。均衡桶可用 ReduceScatter，偏斜桶用定向 Reduce。阈值例为 `max(lengths) * world_size <= 2 * sum(lengths)`。按块复用有预算的 staging，等待其实际消费者后再覆写。OpenWAM staging 允许惰性申请/扩容，评估首次和稳态分配成本。
- **前提与回退** 仅在下游只消费 owner 段时允许用非 owner 段作 scratch。GR00T C7 依赖 C6。OpenWAM 的两者由同一安装器接入。保留归约前后缩放、dtype、舍入和流顺序。OpenWAM 的融合打包与范数侧流为受限变体。
- **最小验证** 覆盖均衡、极偏、尾块、不同 dtype、fallback 路径，逐 owner 对照，并验证峰值和缓冲复用无踩写。

### C8 参数收集流水 `[已核实改造案例。强版本绑定参考]`

**共同 Before** DeepSpeed ZeRO-2 `step` 末尾同步执行 `all_gather_dp_groups`，内部按组、按分片收集。先确认这一段是关键路径，再选以下子模式。

- **C8a · GR00T step 内按参数组流水** 在 DeepSpeed partition 前 regroup 优化器，把各组更新与该组异步 AllGather 重叠，正常 step 返回前排空。示例锁定构建与私有方法，要求 BF16、累积步数为 1、合适的优化器、分片与 rank 布局。默认限制保存/续训，显式启用后仍须匹配 rigid ZeRO-2 的布局清单。允许的 rank-count fallback 使用原生布局并拒绝流水 checkpoint。失败或仍有在途工作时，先清理并阻止从半更新状态保存/重用。验收覆盖同布局往返、错误清单拒绝、关闭开关时原生兼容 checkpoint 续训。
- **C8b · OpenWAM 跨步延迟 Broadcast** 按下步首次读权重顺序规划 owner 分段，用独立 `ProcessGroup` 提交，逐层等待。所有 rank 同序建组、校验同一计划并完成资源清理。联合 eager 路径逐层等，编译联合循环、非联合路径、eval、checkpoint 和退出等旁路按需先 `finish()`。编译路径的提前排空是正确性保护，会牺牲该段可重叠窗口。在线编码接管发起点时，与数据管线专项协同验证冻结编码器输入和 batch 顺序。验收覆盖 eager、compiled、旁路、提交中途失败和训练结束。

单卡只做静态与 CPU/mock 验证。慢 rank 根因未解时先完成分流。增显存方案遇到 OOM 边界时优先恢复原策略。兼容性补丁和 [R1–R3](triage-and-rccl.md) 按正确性保护验收。
