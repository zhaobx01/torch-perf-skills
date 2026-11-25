# TurboPhysAI 跨模型数据管线与异步 H2D 优化证据

> **来源与核实边界**：
> - 汇总仓库与分支：[zhaobx01/TurboPhysAI (`corpus/all-models`)](https://github.com/zhaobx01/TurboPhysAI/tree/corpus/all-models)
> - 本次核实快照 SHA：`796d3f6bce`（基于官方 `HYGON-AI/TurboPhysAI` `main` `1eb24269da` + 13 个 Open PR，共覆盖 16 个模型）
> - 本文件全部结论来自**静态对比本地 Before 基线源码与 After 实现源码**，不含未复测的性能数字；凡转述源码注释中的说法，均标注 **「来源自述，未复测」**。外部链接仅作溯源参考，离线环境仍可完整使用本 Skill。

---

## 1. 按机制分组的跨模型速查表

（可点击链接统一指向 `corpus/all-models` 分支对应文件，不写死行号，以关键函数/类名定位；同质模型合并列举。）

| 模式 | 机制子项 | 代表模型（原始 PR） | 基线 SHA | After 实现链接与关键符号 | 核实过的 Before 痛点 → After 要点 | 证据级别 | 同质 / 关联模型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **D1** | DataLoader 并发、`pin_memory` 与 `persistent_workers` | Sparse4D (`#43`) | `249ffbb695` | [dataloader.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/sparse4d/dataloader.py) (`dataloader_wrapper`) | Before `datasets/builder.py` 显式传 `pin_memory=False` 且未设 `persistent_workers`；After 统一设 `pin_memory=True`，按 `num_workers > 0` 条件设 `persistent_workers` 与 `prefetch_factor=2` | 已核实改造案例 | FlashOCC (`#30`, [pin_memory.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/flashocc/pin_memory.py), [persistent_workers.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/flashocc/persistent_workers.py))；DiT4DiT (`#28`, [dataloader.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/dit4dit/dataloader.py) 含 `num_workers==0` 参数剔除) |
| **D1** | Accelerate 非阻塞 H2D 配置 | LeRobot-ACT (`#44`) | `3f2179f3b6` / `accelerate` `6afc1e5ee2` | [nonblocking_h2d.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lerobot_act/nonblocking_h2d.py) (`nonblocking_h2d_wrapper`) | Before `lerobot_train.py` 构造 `Accelerator()` 未传 `DataLoaderConfiguration(non_blocking=True)`，`accelerate/data_loader.py` 默认 `_non_blocking=False`；After 保留已有配置字段并打开 `non_blocking=True` | 已核实改造案例 | DiT4DiT (`#28`, `dataloader.py` `prepare_data_wrapper`) |
| **D2** | MMCV `DataContainer` 支持原生 pin 线程 | MapTRv2 (`#45`) | `e03f097abe` / `mmcv` `ccdc61c087` | [datacontainer_pin.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/maptrv2/datacontainer_pin.py) (`datacontainer_pin_wrapper`) | Before `mmcv/parallel/data_container.py` 的 `DataContainer` 无 `pin_memory` 方法导致标准 pin 线程跳过；After 递归 pin `not data.cpu_only` 的内部数据，防重复包装并保留 `namedtuple`/`Mapping` 语义 | 已核实改造案例 | Sparse4D (`#43`, 复用 `common/mmcv/data_container.py`)；BEVFormer (`main`, `data.py` `_pin_memory`) |
| **D3** | RLDS/TF 数据集 `spawn` Worker + 可 pickle 重建 | OpenVLA (`main`) | `c8f03f48af` | [spawn_dataloader.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openvla/spawn_dataloader.py) (`dataloader_spawn_wrapper`, `_picklable_dataset`) | Before `prismatic/training/strategies/base_strategy.py` 对 RLDS 写死 `num_workers=0`（避免 `fork` 活跃 TF 线程池死锁）；After 仅对 RLDS 注入 `spawn` context + `num_workers=2` + `pin_memory=True`，通过 `__reduce__` 在子进程重建 TF graph | 已核实改造案例 | — |
| **D4** | 独立 Copy-Stream 异步 H2D 预取 | LeRobot-ACT (`#44`) | `accelerate` `6afc1e5ee2` | [h2d_prefetch_stream.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lerobot_act/h2d_prefetch_stream.py) (`prefetch_dataloader_iter`, `supports_prefetch_dataloader`, `_record_batch`) | Before `DataLoaderShard.__iter__` 在默认流执行 `send_to_device`；After 仅在 `cuda + pin_memory + non_blocking` 时启用独立 `copy_stream`，用 `Event`、`record_stream`、`pending_sources` 保活与 `finally` 同步保障安全 | 已核实改造案例 | BEVFormer (`main`, [data.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevformer/data.py) `CudaPrefetchLoader` 后台线程 + 独立流 + 几何预计算) |
| **D5** | HF Dataset 按需列解码（跳过图像列） | LingBot-VLA-V2 (`#40`) | `be969b8fd1` | [fast_hf_query.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/fast_hf_query.py) (`query_hf_dataset`, `_ColumnView`, `wrap_dataset_init`) | Before `base_dataset.py` 用 `self.hf_dataset[relative_indices][key]` 与 `self.hf_dataset[idx]` 触发无关图像列解码；After 用 `select_columns([key])` 与 `_ColumnView` 裁剪相机列，仅解码目标列 | 已核实改造案例 | — |
| **D5** | 采样器向量化与跨 epoch 连续分片 | LeRobot-ACT (`#44`) | `3f2179f3b6` | [sampler_vector.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lerobot_act/sampler_vector.py) (`vectorized_sampler_wrapper`) + [infinite_batch_sampler.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lerobot_act/infinite_batch_sampler.py) (`continuous_batches_wrapper`, `preserve_resume_epoch_wrapper`) | Before `EpisodeAwareSampler` 逐帧构造索引且每 epoch 重建分片；After 用 `torch.randperm` + `np.searchsorted` 向量化，并在校验分片契约与保护断点 `state_dict` 的前提下衔接连续 epoch | 已核实改造案例 | MapTRv2 (`#45`, [dataset_vectorization.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/maptrv2/dataset_vectorization.py) 用 `np.interp` 替掉 Shapely `LineString.interpolate` 循环) |
| **D6** | 冻结编码器两阶段离线特征缓存 | OpenWAM (`#27`) | `0c9ed80d80` | [cache_vae_latents.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/cache_vae_latents.py) (`_Episodes`, `_episode_frames`) + [text_store.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/text_store.py) | 连续训练窗口重叠导致重复解码与重复跑冻结 VAE/T5（来源自述，未复测）；After 分步按 episode 编码并原子重命名落盘 `.npy` + `index.json`，训练期直接读缓存且 T5 留 CPU | 已核实改造案例（受限工作流参考） | Cosmos3 (`#19`, 补丁含 `local_avae_cache.py`) |
| **D7** | 冻结编码器单流前瞻在线编码（与 ZeRO-2 AllGather 重叠） | OpenWAM (`#27`) | `0c9ed80d80` | [online_encode.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/online_encode.py) (`OnlineEncoder`) | 在 ZeRO-2 `all_gather_dp_groups` 异步执行期间，单流预编码下一 batch；严格校验冻结 VAE/T5 不在 ZeRO 分组内、`grad_accum==1`、`num_workers>=1` | 已核实改造案例（受限/单模型参考） | — |

---

## 2. 高价值代表案例与陷阱细节

### 2.1 Copy-Stream 预取必须同时管住“设备复用”与“主机保活”（LeRobot-ACT `#44`）
- **核实要点**：`h2d_prefetch_stream.py` 的 `prefetch_dataloader_iter` 展示了生产级异步 H2D 预取的四个缺一不可的组件：① 入口门禁 `supports_prefetch_dataloader` 仅在 `cuda + pin_memory + non_blocking` 全满足时开启；② `consumer.wait_event(event)` 后递归调用 `_record_batch(device_batch, consumer)` 防止设备分配器复写；③ 用 `pending_sources` 持有未完成 DMA 拷贝的主机 pinned batch，仅在 `event.query()` 完成后释放；④ `finally: copy_stream.synchronize()` 防止循环提前退出时主机释放正在拷贝的缓冲。

### 2.2 切勿对 `num_workers == 0` 透传多进程专用参数（DiT4DiT `#28` / Sparse4D `#43`）
- **核实要点**：`dit4dit/dataloader.py` 与 `sparse4d/dataloader.py` 都显式检查 `num_workers == 0` 边界：PyTorch `DataLoader` 在 `num_workers=0` 时传入 `persistent_workers=True`、`prefetch_factor` 或 `multiprocessing_context` 会直接抛 `ValueError`，包装或修改 DataLoader 工厂时必须按 `num_workers > 0` 做条件分支。

### 2.3 连续 Batch 采样与断点 Resume 的隐蔽冲突（LeRobot-ACT `#44`）
- **核实要点**：`infinite_batch_sampler.py` 揭示了 Accelerate 断点恢复的一个常见坑：Accelerate 启动时会对 sampler 调 `set_epoch(0)`，若不加 `preserve_resume_epoch_wrapper` 保护，会直接把 checkpoint 恢复出来的 `epoch` / `start_index` 清零；此外连续拼接 epoch 必须断言 `split_batches=False, even_batches=True, drop_last=False` 且 `start_index % global_batch == 0`。

### 2.4 OpenVLA RLDS 为何必须用 `spawn` 而非 `fork`（OpenVLA `main`）
- **核实要点**：`spawn_dataloader.py` 明确指出 RLDS 底层由 TensorFlow 线程池驱动，主进程 `fork` 会把活跃线程池锁状态一并复制进子进程造成死锁；正确解法是 `mp.get_context("spawn")` 配合 `__reduce__` 仅序列化构造参数、在子进程内重建 TF graph，且只匹配 `RLDSDataset`/`EpisodicRLDSDataset` 而不伤及 eval DataLoader。

---

## 3. 未发现项与邻近方向排除

- **本次覆盖范围内未发现项**：16 模型快照中未发现引入 NVIDIA DALI / WebDataset 专用 GPU 端图像解码管线替换原始框架 DataLoader 的案例（GR00T `compat.opencv_video` 属 CPU OpenCV 视频解码兼容适配）。
- **邻近方向排除（本 Skill 不收纳）**：
  - **训练步内标量 D2H 同步与位置元数据 CPU 化**：`GR00T` `qwen_position.py`、`LingBot-VLA-V2` `qwen3vl_rope_index_cpu.py`、`MapTRv2` `sync_elimination.py`、`LeRobot-ACT` `deferred_metrics.py` → 归 `optimize-sync-elimination`；
  - **为稳住 `torch.compile` shape 而做的定长填充/分桶**：`GR00T` `fixed_padding.py`（`pad_156` + `vl_padding_dropout`）与 `OpenVLA` `text_len_bucket.py` → 归 `torch-compile-integration`（P3 重编译治理）；
  - **训练步内零散主机矩阵搬运（如 `Sparse4D` `instance_bank.py`、`MapTRv2` `assigner.py`、`BEVFusion` `transfusion.py`）**：发生在模型 forward/assigner 内部而非 DataLoader 边界 → 归 `optimize-sync-elimination`（P7）。
