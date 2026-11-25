# TurboPhysAI 跨模型分布式通信优化证据

> **来源与核实边界**：
> - 汇总仓库与分支：[zhaobx01/TurboPhysAI (`corpus/all-models`)](https://github.com/zhaobx01/TurboPhysAI/tree/corpus/all-models)
> - 本次核实快照 SHA：`796d3f6bce`（基于官方 `HYGON-AI/TurboPhysAI` `main` `1eb24269da` + 13 个 Open PR，共覆盖 16 个模型）
> - 本文件全部结论来自**静态对比本地 Before 基线源码与 After 实现源码**，不含未复测的性能收益；凡转述源码注释中的显存/字节/加速数字，均标注 **「来源自述，未复测」**。外部链接仅作溯源参考，离线环境仍可完整使用本 Skill。

---

## 1. 按机制分组的跨模型速查表

（可点击链接统一指向 `corpus/all-models` 分支对应文件，不写死行号，以关键函数/类名定位；同质模型合并列举。）

| 模式 | 机制子项 | 代表模型（原始 PR） | 基线 SHA | After 实现链接与关键符号 | 核实过的 Before 痛点 → After 要点 | 证据级别 | 同质 / 关联模型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **C1** | DDP 静态图、免遍历未使用参数与零拷贝视图 | OpenPI (`#29`) | `215abfb217` | [ddp_state.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openpi/ddp_state.py) (`wrap_ddp_static_fast_path`) | Before 设 `find_unused_parameters=True`、`static_graph=(world_size>=8)`、默认 `broadcast_buffers=True`；After 统一设 `find_unused_parameters=False`、`broadcast_buffers=False`、`gradient_as_bucket_view=True`、`static_graph=True` | 已核实改造案例 | LeRobot-ACT (`#44`, [ddp_options.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lerobot_act/ddp_options.py), Before `3f2179f3b6` `lerobot_train.py` 设 `find_unused_parameters=True`) |
| **C1** | 编译路径下局部关闭 Dynamo `optimize_ddp` | PointPillars (`#33`) | `fe25f7a51d` | [optimize_ddp.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/pointpillars/optimize_ddp.py) (`disable_dynamo_ddp_wrapper`) | 在 epoch 训练循环内用 `torch._dynamo.config.patch(optimize_ddp=False)` 避免 Dynamo 按 DDP bucket 过度切分图 | 实现参考，基线待核实（`mmengine` 训练循环默认启用 Dynamo DDP 切分） | — |
| **C2** | FSDP1 解除 AllGather 限流并开前反向预取 | OpenVLA (`main`) | `c8f03f48af` | [fsdp_prefetch.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openvla/fsdp_prefetch.py) (`fsdp_prefetch_wrapper`) | Before `fsdp.py` 显式设 `limit_all_gathers=True` 且未配 prefetch；After 通过保留类名的动态子类用 `setdefault` 注入 `limit_all_gathers=False`、`forward_prefetch=True`、`backward_prefetch=BACKWARD_PRE` | 已核实改造案例 | — |
| **C3** | FSDP2 关闭前向后重分片（权重常驻） | LingBot-VA (`#34`) | `7c6ffa9bfc` | [fsdp.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_va/fsdp.py) (`residency_factory`) | Before `wan_va/distributed/fsdp.py` 设 `reshard_after_forward=True`；After 改为 `reshard_after_forward=False`（可选配置），显存富余时省去反向 AllGather | 已核实改造案例 | — |
| **C3** | FSDP2 合并连续同类层为一个分片单元 | Cosmos3 (`#19`) | `9726697a83` | [cosmos-framework.patch](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/model_examples/Cosmos3/cosmos-framework.patch) (`parallelize_vlm.py` `fsdp_layers_per_group`) | Before 逐层 `fully_shard(block)`；After 按 `fsdp_layers_per_group` 把连续同类块以 `list[nn.Module]` 打包进单一 `FSDPParamGroup` 共享通信 | 已核实改造案例（版本兼容性待核对：`torch >= 2.4` FSDP2） | — |
| **C3** | FSDP2 将大 `embed_tokens` 从根单元拆出 | LingBot-VLA-V2 (`#40`) | `be969b8fd1` | [split_root_fsdp.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/split_root_fsdp.py) (`fully_shard`) | Before `embed_tokens` 留在根 unit（注释称单缓冲 3.65 GiB → 2.6 GiB，来源自述，未复测）；After 先单独 `fully_shard(embed_tokens)` 再 wrap 根模块 | 已核实改造案例 | — |
| **C5** | 分布式 Muon 按 owner 分片 + `all_to_all_single` | LingBot-VLA-V2 (`#40`) | `be969b8fd1` | [muon_gather.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/muon_gather.py) (`gather_owned_updates`) + [muon_owner_ns.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/muon_owner_ns.py) (`partition_range`, `pack_result`) | Before `muon.py` 全量 `dist.all_gather` 且每张卡重复算全部矩阵的 Newton-Schulz；After 按 owner 分配矩阵并用 packed `all_to_all_single` 只拉本 rank 负责切片 | 已核实改造案例（强耦合原子组） | — |
| **C6** | ZeRO-2 双缓冲事件调度（删第二次 `wait_stream`） | GR00T (`#41`) | `deepspeed` `b35d9eb01b` | [zero2_overlap.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/groot/zero2_overlap.py) (`BucketEventScheduler`, `vendor_helpers_check`) | Before `stage_1_and_2.py` 每次满桶立刻 `current_stream().wait_stream(reduction_stream)` 阻断重叠；After 改为复用同一半缓冲前才 `wait_event`，硬锁版本与 AST hash | 已核实改造案例（受限/强版本绑定参考） | — |
| **C7** | ZeRO-2 按 owner 分流 ReduceScatter / Directed Reduce | OpenWAM (`#27`) | `deepspeed` `b35d9eb01b` | [zero.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/zero.py) + [zero_fused.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/zero_fused.py) | Before `allreduce_and_scatter` 按反向顺序打包；After 均衡桶走 `ReduceScatter`、偏斜桶走定向 `Reduce`，并用持久化受限 staging buffer 避免 hook 内分配阻塞 | 已核实改造案例（受限/强版本绑定参考） | GR00T (`#41`, [zero2_owner_reduce.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/groot/zero2_owner_reduce.py), `is_balanced` 阈值 `<= 2x` 填充，依赖 `zero2_overlap`) |
| **C8** | ZeRO-2 step 末 AllGather 流水 / 延迟 Broadcast | OpenWAM (`#27`) | `0c9ed80d80` + `deepspeed` `b35d9eb01b` | [deferred_gather.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/deferred_gather.py) | Before step 末同步等全量 AllGather；After 各 owner 按下步前向首用层序在独立 `ProcessGroup` 上分段异步 `broadcast`，按层 wait，checkpoint/eval 前调 `finish()` | 已核实改造案例（受限/强版本绑定参考） | GR00T (`#41`, [zero2_allgather.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/groot/zero2_allgather.py), 优化器组 AllGather 流水 + state_dict 保护) |
| **R1–R3** | HIP/RCCL 有序提交流、DCP Gloo 组与分支次数对齐 | Cosmos3 (`#19`) | `9726697a83` | [cosmos-framework.patch](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/model_examples/Cosmos3/cosmos-framework.patch) (`fsdp_ordered_comm.py`, `dcp.py`, `omni_mot_model.py`) | 多 stream 提交易触发 RCCL 乱序死锁、DCP `gather_object` 故障及动态分支下 FSDP collective 不齐；分别以单 group 规范 stream、独立 Gloo group、dummy forward 对齐防护 | 「正确性保护（非提速）」 | — |

---

## 2. 高价值代表案例与陷阱细节

### 2.1 DeepSpeed ZeRO-2 隐藏的“双 `wait_stream`”根因（GR00T `#41`）
- **核实要点**：直接对照 `baselines/libs/deepspeed` (`v0.18.4` `stage_1_and_2.py`) 确认，即便配置里已经开了 `overlap_comm=true` 与 `contiguous_gradients=true`，上游在 `reduce_independent_p_g_buckets_and_remove_grads` 里仍同时写了 `reduction_stream.wait_stream(current_stream)` 和 `current_stream.wait_stream(reduction_stream)`，导致反向计算每满一桶就被迫等待归约结束。
- **实战陷阱**：`zero2_overlap.py` 严格锁定了 Torch/DeepSpeed 精确构建号和 4 个私有 helper 的 AST hash，升级 DeepSpeed 后可能已修复或签名变化，严禁跳过门禁直接套用。

### 2.2 FSDP1 预取包装与 `torch.compile` 导入顺序联动（OpenVLA `main`）
- **核实要点**：`fsdp_prefetch.py` 用 `type(original.__name__, (original,), ...)` 生成同名子类以保持 Dynamo 识别，且用 `setdefault` 保留调用方显式传参。
- **实战陷阱**：若同时启用 `compile_fsdp1.py`，编译路径必须在调用时动态查找 `FullyShardedDataParallel` 类（不能在模块顶层缓存未打补丁的旧引用），否则预取参数会被静默丢弃。

### 2.3 `deferred_gather` 的旁路拦截与独立 ProcessGroup 隔离（OpenWAM `#27`）
- **核实要点**：`deferred_gather.py` 将 step 末 AllGather 拆为按前向首次使用层序（不依赖 flat buffer 物理排列）的 owner 分段 broadcast，并使用独立 `ProcessGroup` 避免卡住 `host_async` 的小指标 `all_gather`。
- **实战陷阱**：所有可能在前向主循环之外读取参数的入口（checkpoint 保存、eval 循环、训练结束）都必须先调 `finish()`，漏拦截任何一处都会静默读到上一步旧权重。

### 2.4 `OpenWAM` 的 `zero_norm.py` 侧流范数准备（受限参考）
- **核实要点**：[zero_norm.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/zero_norm.py) 在最终梯度累积结束后于 side stream 上准备本地梯度范数（该 stream 上**绝不跑任何集合通信**），保留 DeepSpeed 原有的 overflow、DP/MP norm 归约、unscale 与 clip 顺序。

---

## 3. 正确性保护、未发现项与邻近方向排除

- **「正确性保护（非提速）」**：`Cosmos3` 的 `OrderedAllGather` / `OrderedReduceScatter` / `ordered_all_reduce`、DCP 的 Gloo 元数据组、以及 `omni_mot_model.py` 的 FSDP collective 对齐属于防死锁/防崩溃保护，不得当提速项汇报或误删。
- **纯兼容补丁（非通信性能优化）**：`MapTRv2` ([training.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/maptrv2/training.py)) 与 `BEVFusion` ([training.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevfusion/training.py)) 的 `ddp_forward_compat_wrapper` 仅补齐新版 PyTorch 缺失的 `_use_replicated_tensor_module` 属性以适配 MMCV 1.6 eval，不构成通信性能优化。
- **本次覆盖范围内未发现项**：16 模型快照中未发现 Tensor Parallelism（TP）通信异步重叠（如 Async-TP / GEMM-AllGather 融合）、Pipeline Parallelism（1F1B 调度）或 DeepSpeed ZeRO-3 深度通信重写；未在 `DreamZero`、`FlashOCC`、`Sparse4D`、`DiT4DiT`、`BEVFormer` 中发现独立的分布式通信优化模块。
- **邻近方向排除**：
  - `LingBot-VA` `train_sync.py` 删梯度累积微步 `dist.barrier()` → 归 `optimize-sync-elimination`（P6）；
  - `LingBot-VA` `group_backend.py` → 实为 FlexAttention Triton 模板后端选择，归 `optimize-kernel-replacement`；
  - `LeRobot-ACT` `pool_backward_gather.py` → 实为单卡通道布局与池化反向算子改写，非分布式通信。
