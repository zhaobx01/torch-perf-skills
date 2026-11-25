# TurboPhysAI 跨模型同步消除证据

> **来源与核实边界**：
> - 汇总仓库与分支：[zhaobx01/TurboPhysAI (`corpus/all-models`)](https://github.com/zhaobx01/TurboPhysAI/tree/corpus/all-models)
> - 本次核实快照 SHA：`796d3f6bce`（基于官方 `HYGON-AI/TurboPhysAI` `main` `1eb24269da` + 13 个 Open PR，共覆盖 16 个模型）
> - 本文件全部结论来自**静态对比本地 Before 基线源码与 After 实现源码**，不含未复测的性能收益；凡转述源码注释中的性能或次数说法，均标注 **「来源自述，未复测」**。外部链接仅作溯源参考，离线环境仍可完整使用本 Skill。

---

## 1. 按机制分组的跨模型速查表

（可点击链接统一指向 `corpus/all-models` 分支对应文件，不写死行号，以关键函数/类名定位；同质模型合并列举，避免流水账。）

| 模式 | 机制子项 | 代表模型（原始 PR） | 基线 SHA | After 实现链接与关键符号 | 核实过的 Before 痛点 → After 要点 | 证据级别 | 同质 / 关联模型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **P1** | 归一化标量留设备 + 删 bool 冗余清洗 | MapTRv2 (`#45`) | `e03f097abe` | [sync_elimination.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/maptrv2/sync_elimination.py) (`maptrv2_head_loss_single`, `bevformer_point_sampling`) | Before 调 `torch.clamp(reduce_mean(num_total_pos), min=1).item()` 且对 `torch.bool` `bev_mask` 做 `nan_to_num`（旧分支含 `.cpu().numpy()`）；After 改 `.reshape(())` 并删除冗余清洗 | 已核实改造案例 | 同文件另覆盖 `maptr_head`、`bevformer_head` |
| **P1** | 绕开框架 getter 隐式 `.item()` | OpenWAM (`#27`) | `0c9ed80d80` | [host_async.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/host_async.py) (`device_grad_norm`) | Before 通过 Accelerate/DeepSpeed getter 取 `grad_norm` 触发 `.item()`；After 在校验 ZeRO-2 无 CPU offload 后直接读取设备端 `_global_grad_norm` 零维张量 | 已核实改造案例（受限/单框架参考：DeepSpeed ZeRO-2） | — |
| **P2** | Qwen3-VL 视觉位置编码与 mRoPE 整段 CPU 化 + 缓存 | GR00T (`#41`) | `transformers` `47b0e478f3` | [qwen_position.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/groot/qwen_position.py) (`fast_pos_embed_interpolate`, `rot_pos_emb`, `get_rope_index`) | Before 在 `for` 循环内把 GPU 标量传入 `linspace`/`arange` 并反复 `.item()` / `argwhere`（注释称“数千次同步 D2H”，来源自述，未复测）；After 一次性 `.detach().to("cpu").tolist()`，CPU 构造并按 `grid_thw` 缓存 | 已核实改造案例 | LingBot-VLA-V2 (`#40`, [qwen3vl_cache_gridTHW.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/qwen3vl_cache_gridTHW.py), [qwen3vl_rope_index_cpu.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/qwen3vl_rope_index_cpu.py))；Cosmos3 (`#19`, [cosmos-framework.patch](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/model_examples/Cosmos3/cosmos-framework.patch) `qwen3_vl_hcu.py`) |
| **P2** | 搬运边界拦截元数据留在 CPU | Cosmos3 (`#19`) | `9726697a83` | [cosmos-framework.patch](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/model_examples/Cosmos3/cosmos-framework.patch) (`cosmos_framework/utils/misc.py` `to`) | Before 将整个 batch 含 `image_size` 一并上卡；After 仅将 `image_size` 留在 CPU，其余字段正常上卡 | 已核实改造案例 | OpenWAM (`#27`, [ports/encode.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/ports/encode.py) 在 CPU 侧算文本长度) |
| **P3** | 布尔掩码索引改为显式索引 `index_select` | BEVFusion (`main`) | `326653dc06` | [common/mmdet3d/bev_pool.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/common/mmdet3d/bev_pool.py) (`quick_cumsum_forward`, `quick_cumsum_backward`) | Before `QuickCumsum` 直接用 `x[kept], geom_feats[kept]` 与 `back[kept] -= 1`；After 改用 `nonzero().flatten()` + `index_select` 与 `back -= kept.to(back.dtype)` | 已核实改造案例 | — |
| **P3** | 删除 `nonzero` 后冗余 `.unique()` | PointPillars (`#33`) | `fe25f7a51d` | [unique_free_sampler.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/pointpillars/unique_free_sampler.py) (`unique_free_sampler_wrapper`) | Before `PseudoSampler.sample` 对 `torch.nonzero(...).squeeze(-1)` 额外调用 `.unique()`；After 直接去除冗余 `.unique()` | 已核实改造案例 | — |
| **P4** | 满足前提时绕开 FA2 varlen/unpad 同步 | OpenVLA (`main`) | `c8f03f48af` | [skip_fa2_unpad.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openvla/skip_fa2_unpad.py) (`make_fast_fa2_causal_mask_wrapper`) | Before 走 FA2 varlen 需每步 `nonzero` + `seqlens.max().item()`（来源自述，未复测）；After 在 FA2 + prefill + 右 padding 时返回 `None` 跳过，左 padding 显式回退 | 实现参考，基线待核实（旧版 `transformers` `_update_causal_mask`） | Cosmos3 (`#19`, `flash2/functions.py` 直传/缓存 `cu_seqlens`) |
| **P5** | 梯度累积微步跳过 `.item()` | DiT4DiT (`#28`) | `d8854a1171` | [training.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/dit4dit/training.py) (`_train_step`) | Before 每个 micro-step 末尾都对 3–4 个 loss 调 `.item()`，但外层仅在 `sync_gradients=True` 时消费；After 在 `not sync_gradients` 时直接 `return {}` | 已核实改造案例 | — |
| **P5** | 前向延迟 `.detach()` 到日志边界（成组原子） | LeRobot-ACT (`#44`) | `3f2179f3b6` | [deferred_metrics.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lerobot_act/deferred_metrics.py) (`deferred_metrics_policy_forward`, `deferred_metrics_wandb_wrapper`) | Before `ACTPolicy.forward` 每次前向都执行 `l1_loss.item()` / `mean_kld.item()`；After 返回 `.detach()` 设备张量，仅在 `WandBLogger.log_dict` 边界才 `.item()` | 已核实改造案例 | — |
| **P5** | 多标量合并单次归约与单次 D2H | BEVFusion (`main`) | `326653dc06` | [training.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevfusion/training.py) (`parse_losses`) | Before 循环内逐个 loss `dist.all_reduce` + `.item()`；After `torch.stack` 后单次 `all_reduce` + 单次 `reduced.cpu().tolist()` | 已核实改造案例 | — |
| **P5 + P8** | pinned buffer + `cuda.Event` 延迟一步读取 | OpenWAM (`#27`) | `0c9ed80d80` | [host_async.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/host_async.py) (`LaggedMetrics`, `upload`) | Before 每步阻塞 `.item()` 读 loss/grad_norm；After `stack` 后异步拷入 `pin_memory=True` host buffer，下一步 `push` 时再 `event.synchronize()` + `.tolist()` | 已核实改造案例 | — |
| **P6** | 梯度累积微步移除多余 `dist.barrier()` | LingBot-VA (`#34`) | `7c6ffa9bfc` | [train_sync.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_va/train_sync.py) (`train`, `sync_wrapper`) | Before `wan_va/train.py` 将 `dist.barrier()` 放在 `if losses['should_log']:` 之外，每个 micro-step 都全卡同步；After 仅在 `should_log` 步执行 `barrier()` | 已核实改造案例 | OpenWAM (`#27`, `host_async.configure_deepspeed` 关闭 DeepSpeed `timers.throughput` 同步) |
| **P7** | 必须留在 CPU 的匈牙利匹配收敛为单次 D2H/H2D | MapTRv2 (`#45`) | `e03f097abe` | [assigner.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/maptrv2/assigner.py) (`_hungarian_match_impl`) | 代价矩阵一次性 `.detach().cpu()`，匹配结果一次性 `torch.as_tensor(..., device=device)` 搬回，隔离于编译图外 | 已核实改造案例 | BEVFusion (`main`, [transfusion.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevfusion/transfusion.py)) |

---

## 2. 高价值代表案例与陷阱细节

### 2.1 Qwen3-VL 位置编码 CPU 化与缓存边界（GR00T `#41` / LingBot-VLA-V2 `#40`）
- **核实要点**：Before（`transformers` `47b0e47` `modeling_qwen3_vl.py`）在 `fast_pos_embed_interpolate`、`rot_pos_emb`、`get_rope_index` 三处均存在 GPU 张量进入 Python `for` 循环逐元素读取的问题。
- **实战陷阱**：`LingBot-VLA-V2` 对查表函数额外加了 `@torch._dynamo.disable` 并设缓存上限 `_POS_EMBED_CACHE_LIMIT = 16`（满则清空，防止动态分辨率场景显存泄漏）；`GR00T` 在 `get_rope_index` 中保留纯文本单模态分支走回原路径，并在缓存前检查 `not torch.jit.is_tracing()`。

### 2.2 前向延迟指标的强耦合原子性（LeRobot-ACT `#44`）
- **核实要点**：`deferred_metrics.py` 把 `ACTPolicy.forward` 里的 `.item()` 改为 `.detach()` 后，必须同步包装下游日志消费点 `WandBLogger.log_dict` 做 `isinstance(..., torch.Tensor)` 转换。
- **实战陷阱**：如果只改前向不改消费端，日志序列化或 JSON/WandB 上报会直接报错；因此必须作为**强耦合原子组**成对启停（源码里也显式禁止单独开关组内子选项）。

### 2.3 延迟指标与异步集合通信共存时的 `record_stream` 保护（OpenWAM `#27`）
- **核实要点**：`host_async.LaggedMetrics._gather_aside` 在与 `deferred_gather` 共存时，不能在主训练 stream 上直接做指标 `all_gather`（否则会被 rank 偏斜卡住），而是改用独立 `cuda.Stream` 执行 `work.wait()` + `host.copy_(..., non_blocking=True)` + `event.record(self.stream)`。
- **实战陷阱**：`values` 和 `gathered` 在主训练 stream 分配、在 side stream 消费，必须调用 `values.record_stream(self.stream)` 和 `gathered.record_stream(self.stream)`，否则 PyTorch 缓存分配器可能提前回收显存导致静默读脏数据。

### 2.4 局部回退路径残留同步点（BEVFormer `main`）
- **核实要点**：[geometry_sca.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevformer/geometry_sca.py) (`encoder_forward`) 优先走预取好的 `_prefetched_sca_inputs` 避开现场 `int(index_lengths.max().item())`，但在未启用预取的验证/测试 loader 路径上仍会回退触发 `.item()`——排查时务必确认实测走的是哪条分支。

---

## 3. 正确性保护、未发现项与邻近方向排除

- **「正确性保护（非提速）」——不应作为提速项删除**：
  - `Cosmos3` `checkpoint/dcp.py` 中的 `allocation_failed = bool(failed.item())`、`staging_stream.synchronize()` 与 `dist.barrier()` 属于分布式 checkpoint 一致性保护；
  - `MapTRv2` `optimizer.py` 加载 checkpoint 校验 step 的一次性 `.item()` 不在训练热路径上。
- **本次覆盖范围内未发现项**：
  - `OpenWAM` `ports/vae.py` 与 `pointwise.py` 未发现步内标量 D2H 同步改造；
  - `Sparse4D` 的 `instance_bank.py` / `grid_mask.py` 及 `FlashOCC` / `OpenPI` / `DreamZero` 主要优化集中在编译、算子、布局、显存或数据管线，未发现独立的步内 D2H 同步消除模块（`Sparse4D instance_bank.py` 有一处将 `new_tensor(self.default_time_interval)` 改为直接传标量给 `torch.where`，避免单步小 H2D）。
  - C++/HIP 扩展源码内部是否含显式 `hipStreamSynchronize` 未在本次 Python 层静态核实范围内。
- **邻近但归属其他方向（本 Skill 不收纳）**：
  - **DataLoader / batch 级异步 H2D 预取**：`BEVFormer`/`BEVFusion` `CudaPrefetchLoader`、`LeRobot-ACT` `nonblocking_h2d.py` & `h2d_prefetch_stream.py`、`MapTRv2` `datacontainer_pin.py`、`FlashOCC` `pin_memory.py` → 归**数据管线**方向；
  - **编译期 `max_seqlen` 分桶 / 定长 padding**：`Cosmos3` `sequence_packing.py`、`GR00T` `fixed_padding.py`、`OpenVLA` `text_len_bucket.py` → 归 `torch-compile-integration`；
  - **分布式通信重叠与有序调度**：`OpenWAM` `deferred_gather.py`、`GR00T` `zero2_*.py`、`LingBot-VLA-V2` `split_root_fsdp.py` → 归**分布式通信**方向。
