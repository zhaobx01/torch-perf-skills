# 分布式通信优化模式（C1–C8）

以下模式按 **层级（公开 API/配置 → 集合通信重构 → 受限框架内部改写）** 与 **Before 坏味道 → After 手法 → 前提与不适用 → 最小验证** 组织。所有 `[已核实改造案例]` 均已对照 TurboPhysAI 16 模型快照的 Before 基线与 After 实现源码核实（见 [turbophysai-comm-evidence.md](turbophysai-comm-evidence.md)），不含未复测的性能承诺。

> **执行顺序铁律**：动任何模式前，必须先按 [triage-and-rccl.md](triage-and-rccl.md) 排除慢 rank 偏斜；实施时先穷尽一级模式（`C1`–`C5`），仅当暴露通信仍高且获用户二次明确授权后，才考虑二级模式（`C6`–`C8`）。

---

## 一级：公开 API 与配置级调优

### C1 DDP 静态图与通信配置调优 `[已核实改造案例]`

- **Before 坏味道**：DDP 构造保守开启 `find_unused_parameters=True`（每步反向需额外遍历 autograd 图）、未开启 `static_graph=True`（或仅在 `world_size >= 8` 时开启）、未开 `gradient_as_bucket_view=True`（梯度多一次拷贝）或对无有效 BN running-stat 需求的模型保留默认 `broadcast_buffers=True`。
- **After 手法**：
  - 确认所有可训练参数每步均参与反向后，设 `find_unused_parameters=False`。
  - 确认计算图拓扑跨步固定（无条件跳层/动态开关参数）后，设 `static_graph=True`、`gradient_as_bucket_view=True`；确认不依赖每步广播 buffer（如无同步 BN/无动态更新 buffer）时设 `broadcast_buffers=False`。
  - **与 `torch.compile` 交互**：若编译区域横跨 DDP 且 `DDPOptimizer` 切分导致图过碎或回归，可在训练循环局部使用 `torch._dynamo.config.patch(optimize_ddp=False)`（需与 `torch-compile-integration` 联合验证是加速还是降低重叠）。
- **前提与不适用**：
  - 存在条件分支导致部分参数在部分步不参与 loss 时，严禁强设 `find_unused_parameters=False` 或 `static_graph=True`（会报错或静默漏归约）。
  - 模型包含需要跨 rank 同步的 `BatchNorm` / 动态 buffer 时，不可关 `broadcast_buffers`。
- **最小验证**：多卡跑满代表性多步（含动态 batch / 校验分支），确认无 unused-parameter 报错，各 rank 参数权重和梯度在步末保持一致。

### C2 FSDP1 通信预取与解除 AllGather 限流 `[已核实改造案例]`

- **Before 坏味道**：`FullyShardedDataParallel(...)` 显式或默认设 `limit_all_gathers=True`（CPU 线程主动限流等待前序 AllGather 完成，压制重叠），且未开启 `forward_prefetch=True` 与 `backward_prefetch=BackwardPrefetch.BACKWARD_PRE`。
- **After 手法**：
  - 注入 `limit_all_gathers=False`、`forward_prefetch=True`、`backward_prefetch=BackwardPrefetch.BACKWARD_PRE`（建议用 `setdefault` 尊重调用方显式传参）。
  - 若通过包装 `FullyShardedDataParallel` 类注入参数，需保持类名与 `__module__` 不变（如动态子类），并确保 `torch.compile` 路径在运行时动态解析 FSDP 类以免预取被静默绕开。
- **前提与不适用**：
  - `forward_prefetch` 与 `limit_all_gathers=False` 会提前保留下一层完整权重，**必然抬高峰值显存**；显存已逼近上限时须先评估余量或退回 `BACKWARD_POST`。
  - 模型前向执行顺序在步间动态变化（非固定层序）时，开启 `forward_prefetch=True` 可能预取错层甚至引发异常。
- **最小验证**：记录开关前后峰值显存（`max_memory_allocated`）与 T5 重叠率，确认无 OOM 且多卡 loss/梯度一致。

### C3 FSDP2 常驻权重与分片单元粒度调优 `[已核实改造案例]`

- **Before 坏味道**：
  1. 显存尚有明显余量时，FSDP2 仍默认 `reshard_after_forward=True`（ZeRO-3 行为：前向算完立即丢弃权重，反向再额外做一次 AllGather）。
  2. 逐个小层 `fully_shard(block)` 导致大量细碎 AllGather/ReduceScatter，通信启动开销大、带宽打不满。
  3. 根模块（root unit）残余超大未拆分子模块（如 `embed_tokens` 与 vision 留在同一根 unit），导致 backward 末尾一次性分配超大梯度归约缓冲并拉长尾部同步。
- **After 手法**：
  - **显存换通信（权重常驻）**：显存充足时将 `reshard_after_forward` 设为 `False`（保留前向 AllGather 后的参数供反向直接复用，等价 ZeRO-2 通信量）。
  - **合并细碎分片单元**：对连续同类 Transformer block，利用 FSDP2 的 `fully_shard(list_of_blocks, ...)`（如按 `fsdp_layers_per_group` 分组）共享一个 `FSDPParamGroup`，合并通信次数并放大单次消息字节数。
  - **拆分超大根单元**：将根模块内的大体量独立子模块（如 `embed_tokens`）单独先调用 `fully_shard(embed, ...)`，再 `fully_shard(root, ...)`，缩小单次尾部归约缓冲。
- **前提与不适用**：
  - `reshard_after_forward=False` 与增大 `fsdp_layers_per_group` 均会增加峰值显存，须以不 OOM 为前提；`fully_shard(list_of_blocks)` 需确认 PyTorch 版本支持（`torch >= 2.4` 的 composable FSDP2，标为「版本兼容性待核对」）。
  - 变更分片单元会影响 `state_dict` / DCP 加载或 `torch.compile` 边界，须验证 checkpoint 往返。
- **最小验证**：核对各 FSDP unit 参数量分布、峰值显存、checkpoint 保存/加载与 T5 通信次数/暴露时长。

### C4 DeepSpeed ZeRO-2 标准配置级基线核对 `[已核实改造案例]`

- **Before 坏味道**：DeepSpeed ZeRO-2 JSON/YAML 配置中关闭了 `overlap_comm` 或 `contiguous_gradients`，或 `reduce_bucket_size` / `allgather_bucket_size` 过小（通信碎）或过大（延迟首桶发射，难以与反向尾部重叠）。
- **After 手法**：
  - 确认 `zero_optimization.stage == 2` 时开启 `overlap_comm: true`、`contiguous_gradients: true`，结合模型单层梯度大小扫查合理的 `reduce_bucket_size` 与 `allgather_bucket_size`。
  - 同步核对是否残留每步同步计时器（`timers.throughput`，具体处理见 `optimize-sync-elimination` P6）。
- **前提与不适用**：显存极紧时盲目放大 bucket size 会增加常驻临时缓冲导致 OOM。
- **最小验证**：确认实际生效的 DeepSpeed 运行配置与 T5 暴露通信变化，再决定是否真的需要升级到二级改写。

---

## 一级（中侵入）：集合通信重构

### C5 按 Owner 分片计算 + `all_to_all_single` 替代全量 `all_gather` `[已核实改造案例]`

- **Before 坏味道**：分布式优化器（如 FSDP2/DTensor 下的 `DistributedMuon`）在每步更新时先用 `dist.all_gather` 把完整参数/梯度拉到每张卡，再让全部 `world_size` 张卡重复执行相同的重算（如 Newton-Schulz 迭代），既多传输 `world_size` 倍数据，又多算 `(world_size - 1) / world_size` 的重复计算。
- **After 手法**：
  - 将独立的矩阵/参数按确定性规则（`partition_range(num_matrices, world_size, rank)`）划分给各 owner rank；
  - 用 packed `dist.all_to_all_single` 仅把本 rank 负责的矩阵切片从其他 rank 收集过来；
  - 每个 rank 只对 `1 / world_size` 的矩阵执行计算，再按分片规则 `all_to_all_single` 分发回各 rank 更新本地分片。
- **强耦合原子性**：通信收发（`gather_owned_updates`）、owner 区间划分（`partition_range`）与打包/解包格式必须**成组启停与回退**，严禁只改一半。
- **前提与不适用**：
  - 仅适用于各参数/矩阵间更新相互独立的优化器或算子；若不同 rank 分片维度不能整除，须严格处理不均匀长度（padding 或 split sizes）避免越界或数值错位。
  - 参数数量极少（少于 `world_size`）时，负载均衡收益有限。
- **最小验证**：单卡/多卡对照验证每个参数更新后的数值与原始全量 `all_gather` 版本逐位或在容差内一致，覆盖维度不可整除 `world_size` 的边界形状。

---

## 二级（高风险门禁）：DeepSpeed ZeRO-2 深度框架内部流水改造

> **二级门禁硬要求**：`C6`–`C8` 均侵入 DeepSpeed 私有内部实现。**仅当 `C4` 标准配置用尽、Trace 证实暴露通信仍是主要瓶颈，且获用户二次明确授权后才允许实施**；实施代码必须校验目标 `torch` / `deepspeed` 版本及关键函数源码 AST hash，不匹配即拒绝安装或回退。

### C6 ZeRO-2 双缓冲事件调度（消除第二次 `wait_stream`）`[已核实改造案例｜受限/强版本绑定参考]`

- **Before 坏味道**：特定版本 DeepSpeed（已核实 `v0.18.4` `stage_1_and_2.py` `reduce_independent_p_g_buckets_and_remove_grads`）在开启 `contiguous_gradients` 与 `overlap_comm` 后，每次满桶调用 `reduce_ipg_grads()` 紧接着执行 `current_stream().wait_stream(self.reduction_stream)`，迫使计算流立刻等本次归约结束，彻底破坏双缓冲重叠。
- **After 手法**：用事件调度器（`BucketEventScheduler`）替代无条件 `wait_stream`——reduction stream 消费完半缓冲后 `record_event`，计算流仅在**切换并准备复用同一个半缓冲**时才 `wait_event`；并锁定 `reduce_ipg_grads` 等 helper 的 AST hash 与框架版本。
- **前提与不适用**：依赖双缓冲 `bucket.buffer[0/1]` 存在且无其他路径绕开调度器覆写同一缓冲；`gradient_accumulation_steps > 1` 或开启 CPU offload 时须单独核对 DeepSpeed 实际分支。
- **最小验证**：先跑版本/AST 门禁失败回退单测，再在多卡验证与原版 ZeRO-2 梯度/权重一致且 T5 反向重叠率显著提升。

### C7 ZeRO-2 按 Owner 分流 ReduceScatter 与 Directed Reduce `[已核实改造案例｜受限/强版本绑定参考]`

- **Before 坏味道**：DeepSpeed `allreduce_and_scatter` 按反向产生顺序打包，对目标 owner 分布不均的 bucket 若直接走 `ReduceScatter` 会传输大量零填充，若走 `AllReduce` 则产生额外冗余流量；且在 backward hook 内动态分配临时显存，显存紧绷时会阻塞主机数秒。
- **After 手法**：
  - 按目标 owner rank 重排切片并统计各 owner 长度；
  - 均衡 bucket（例如 `max(lengths) * world_size <= 2 * sum(lengths)`）打包后走原生 `ReduceScatter`；偏斜 bucket 改走按 owner 的定向 `Reduce`（dtype 一致时可原地归约），避免传输 `world_size` 倍零填充；
  - 预分配按 `(device, dtype)` 复用的持久化受限 staging buffer（设上限如 64M 元素），禁止在 backward hook 内按桶新分配大缓冲。
- **强耦合原子性**：在 `GR00T` 中本优化显式依赖 `C6`（`zero2_overlap`）；两者的缓冲与流同步契约须联合验证。
- **前提与不适用**：仅读取 owner 切片、非 owner 切片作为 scratch 覆写——须确认模型与优化器严格遵守 ZeRO-2 分片语义。
- **最小验证**：构造均衡与极端偏斜两种梯度分布，验证各 owner 归约结果与原版等价，且 backward 期间无峰值显存突增。

### C8 ZeRO-2 step 末 AllGather 流水化与延迟 Broadcast `[已核实改造案例｜受限/强版本绑定参考]`

- **Before 坏味道**：ZeRO-2 每步 `optimizer.step()` 末尾同步执行一次全量参数 `AllGather`，计算流必须等全部参数聚齐才能启动下一步前向；当离线缓存或轻量数据准备无法掩盖它时，整段 AllGather 暴露在关键路径上。
- **After 手法**：
  - **参数组流水 AllGather**：将优化器参数更新与分片 AllGather 流水衔接，同时包装 `state_dict`/`load_state_dict` 保障未完成同步前不暴露半更新权重。
  - **按前向首次使用层序延迟 Broadcast（`deferred_gather`）**：各 owner 按下一步前向首次使用顺序（prologue → layer 0 → layer 1 ...）分段异步 `broadcast`，**使用独立 `ProcessGroup`** 避免与步间小指标通信互相阻塞；训练流不在 step 末等待，而是在 prologue 与每层前向开始前按需 wait 对应分片；在 **checkpoint 保存、eval、非主干前向路径与训练结束前** 强制调用 `finish()` 等待全部完成。
- **前提与不适用**：
  - 延迟 Broadcast 强依赖前向层序固定，且必须拦截所有可能读取参数的旁路（eval、EMA、checkpoint、异常退出），漏拦任意一处都会读到陈旧权重导致静默错误！
  - 创建额外 `ProcessGroup` 会增加通信器显存占用，须纳入显存验收。
- **最小验证**：专门验证训练→eval→保存 checkpoint→新进程加载续训全链路权重一致性，核查无陈旧权重读取与无死锁。

---

## 通用不适用场景

- 单卡训练（无分布式集合通信）。
- T5 起始极差大或最慢 rank 空闲差主导的“假通信瓶颈”（先排查慢 rank 根因，不调通信重叠）。
- 显存已处于 OOM 边缘时的增显存类重叠手段（`limit_all_gathers=False`、`forward_prefetch=True`、`reshard_after_forward=False`、放大 bucket/group）。
- 仅用于解决框架版本兼容性的 DDP 包装（如 MMCV `_use_replicated_tensor_module` 兼容补丁）属于兼容适配，不归为通信性能优化。
