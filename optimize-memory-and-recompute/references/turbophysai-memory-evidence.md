# TurboPhysAI 跨模型显存与重算优化证据

> **来源与核实边界**：
> - 汇总仓库与分支：[zhaobx01/TurboPhysAI (`corpus/all-models`)](https://github.com/zhaobx01/TurboPhysAI/tree/corpus/all-models)
> - 本次核实快照 SHA：`796d3f6bce`（基于官方 `HYGON-AI/TurboPhysAI` `main` `1eb24269da` + 13 个 Open PR，共覆盖 16 个模型）
> - 本文件全部结论来自**静态对比本地 Before 基线源码与 After 实现源码**，不含未复测的性能或显存收益；凡转述源码注释中的数字或说法，均标注 **「来源自述，未复测」**。外部链接仅作溯源参考，离线环境仍可完整使用本 Skill。

---

## 1. 按机制分组的跨模型速查表

（可点击链接统一指向 `corpus/all-models` 分支对应文件，不写死行号，以关键函数/类名定位；同质模型合并列举。）

| 模式 | 机制子项 | 代表模型（原始 PR） | 基线 SHA | After 实现链接与关键符号 | 核实过的 Before 痛点 → After 要点 | 证据级别 | 同质 / 关联模型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **M1** | 显存富余时整模块关闭骨干 AC | FlashOCC (`#30`) | `449dc7947d` | [disable_activation_checkpoint.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/flashocc/disable_activation_checkpoint.py) (`disable_activation_checkpoint_wrapper`) | Before 配置 `bevdet-occ-r50-4d-stereo.py` 设 `with_cp=True`；After 复制 `img_backbone` 配置并改 `with_cp=False` | 已核实改造案例 | — |
| **M1** | 关闭前/尾 N 层整层重算 | LingBot-VLA-V2 (`#40`) | `be969b8fd1` | [ckpt_policy.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/ckpt_policy.py) (`enable_gradient_checkpointing`) | Before `gradient_checkpointing_enable()` 开启全部层重算；After 关掉前 `expert_layers` 个 `Qwen2DecoderLayer` 的 `gradient_checkpointing = False` | 已核实改造案例 | LingBot-VA (`#34`, [fsdp.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_va/fsdp.py) `checkpoint_factory` `tail_layers` 跳过尾部 N 层) |
| **M1** | 按 `layer_stride` 间隔重算（兼避 compile 多图变体） | OpenPI (`#29`) | `215abfb217` | [gemma_checkpoint_stride.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openpi/gemma_checkpoint_stride.py) (`wrap_gemma_forward`) | Before `gemma_pytorch.py` 循环内每层调 `torch.utils.checkpoint.checkpoint`；After 接管 forward 循环按 `layer_stride` 间隔重算（`stride=0` 全直连），避免外层 `torch.compile` 按层号生成多图变体 | 已核实改造案例 | — |
| **M2** | 选择性激活检查点（SAC）按算子策略保存 | OpenWAM (`#27`) | `0c9ed80d80` | [sac.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/sac.py) (`_ReplayRegionMode`, `create_selective_checkpoint_contexts`) | 使用 PyTorch 标准 SAC 仅对末尾 `sac_layers` 个已 checkpoint 层按 `sac_attention`/`sac_ffn_full`/`sac_gemm` 策略保留高代价算子输出，并保持跨 eager 区域重放位置对齐 | 已核实改造案例 | LingBot-VA (`#34`, [fsdp.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_va/fsdp.py) `save_matmul` 仅 `MUST_SAVE` `aten.mm`/`addmm` + `sac_layer_interval`) |
| **M3** | 关闭训练 prefill 隐式 KV Cache | GR00T (`#41`) | `transformers` `47b0e478f3` | [kv_cache.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/groot/kv_cache.py) (`disable_kv_cache_wrapper`) | Before `Qwen3VLTextModel.forward` 在 `use_cache` 默认为 `True` 且 `past_key_values is None` 时隐式建 `DynamicCache`；After 仅在未显式传 `use_cache` 且无 `past_key_values` 时注入 `use_cache=False` | 已核实改造案例 | — |
| **M4** | 同 dtype 先 cat 再 cast + 切片 `.clone()` 断 `_base` 滞留 | LingBot-VLA-V2 (`#40`) | `be969b8fd1` | [decoder_memory.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/decoder_memory.py) (`cat_float`, `local_attention_input`, `forward`) | Before 逐段转 float 再 cat 且直接传 `att_output[:, start:end]` 切片导致整块大缓冲滞留至反向结束；After 同 dtype 先 cat 后转，并对 `_base is not None` 的切片显式 `.clone()` | 已核实改造案例 | — |
| **M5** | 裁剪训练期未消费死分支与冗余 `train/eval` 遍历 | DreamZero (`#42`) | `ab790c198f` | [skip_unused_vae.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/dreamzero/skip_unused_vae.py) (`encode_image`, `forward`) + [train_mode_cache.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/dreamzero/train_mode_cache.py) | Before `encode_image` 每步执行下游未消费的首帧 VAE 编码；After 增加 `encode_y=False` 跳过，并缓存冻结模块 `train/eval` 遍历状态 | 已核实改造案例 | LingBot-VLA-V2 (`#40`, [moge_depth_only.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/moge_depth_only.py), Before `moge` `b942f00bdc` 跳过未用的 `normal_head`) |
| **M6** | Warmup 后多卡对齐冻结 Python GC | DreamZero (`#42`) | `ab790c198f` | [Fix_GC.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/dreamzero/Fix_GC.py) (`freeze_gc_after_warmup`) | 在 `current_step >= 50` 后先 `dist.barrier()` 再 `gc.collect(2)` + `gc.freeze()`，消除多卡不同步 GC 停顿 | 已核实改造案例 | OpenVLA (`main`, [gc_freeze.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openvla/gc_freeze.py))；OpenWAM (`#27`, `runtime.online.yaml` `OPENWAM_GC_FREEZE=1`) |
| **M7** | 分配器可扩展段防碎片配置 | LingBot-VA (`#34`) | `7c6ffa9bfc` | [runtime.yaml](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_va/configs/runtime.yaml) | 启用 `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` 降低动态分配碎片 | 已核实改造案例（版本兼容性待核对） | OpenWAM (`#27`, [runtime.online.yaml](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/configs/runtime.online.yaml))；`HIP_BUFFER_EXTRA_SIZE=0` 见 `lerobot_act`/`lingbot_vla_v2`/`openpi` |

---

## 2. 高价值代表案例与陷阱细节

### 2.1 `torch.compile` 下的 `layer_stride` 陷阱（OpenPI `#29`）
- **核实要点**：`gemma_checkpoint_stride.py` 指出，在外层有 `torch.compile` 时，如果仅包装内部 `checkpoint` 辅助函数并按 Python 层号 `layer_idx` 做条件判断，Dynamo 会为每个层号追踪出不同图变体；必须直接接管 forward 循环本身，让不重算的层走直接调用路径。

### 2.2 切片视图 `_base` 导致的隐蔽显存滞留（LingBot-VLA-V2 `#40`）
- **核实要点**：`decoder_memory.py` 的 `local_attention_input` 对 `att_output[:, start:end]` 转 dtype 后专门检查 `local._base is not None` 并执行 `.clone()`。若只做切片不 clone，PyTorch autograd 为反向保存 `local` 时会连带锁住整块联合 `att_output` 的 storage，使峰值显存无谓翻倍。

### 2.3 SAC 跨区域重放与未重算层隔离（OpenWAM `#27`）
- **核实要点**：`sac.py` 明确两点工程约束：① SAC 只施加在本身已开 AC 的末尾 `sac_layers` 层，未开 AC 的层本就由 autograd 保存激活，不得二次缓存；② 当一次 checkpoint 内跨越多个独立 eager 区域时，用 `_ReplayRegionMode` 保留 `func_counter` 进度，避免反向重放时算子计数错位。

### 2.4 多卡 `gc.freeze()` 的时机与对齐（DreamZero `#42` vs OpenVLA `main`）
- **核实要点**：`OpenVLA` 在训练循环启动前直接 `gc.freeze()`；`DreamZero` 则选择在 `current_step >= 50`（经过 warmup、编译与动态缓存稳定）后先 `dist.barrier()` 再 `gc.collect(2)` + `gc.freeze()`。后者能把 warmup 期间新分配的编译/缓存常驻对象一并冻结，同时对齐多卡回收时机。

---

## 3. 未发现项与邻近方向排除

- **本次覆盖范围内未发现项**：16 模型快照中未发现 CPU Activation Offloading（把激活异步卸载到主机内存）或 ZeRO-Offload 优化器卸载作为提速手段（`OpenWAM` README 反而明确要求关闭 checkpoint offload 和 CPU optimizer offload）。
- **邻近方向排除（本 Skill 不收纳）**：
  - **两阶段离线编码落盘缓存**：`OpenWAM` ([cache_vae_latents.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/cache_vae_latents.py) / `latent_cache.py` / `text_store.py`) 与 `DiT4DiT` ([cosmos_cache.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/dit4dit/cosmos_cache.py)) 涉及离线数据构建与训练输入切换 → 归**数据管线 / 训练工作流**方向；
  - **FSDP2 拆分超大根单元降尾部梯度缓冲**：`LingBot-VLA-V2` `split_root_fsdp.py` 与 `LingBot-VA` `reshard_after_forward=False` 涉及并行分片拓扑与通信量 → 归 `optimize-distributed-comm`（`C3`），仅与本 Skill 协同核算显存预算；
  - **周期性 `empty_cache` 旁边的微步 `dist.barrier()`**：`LingBot-VA` `train_sync.py` 删微步多余 `barrier()` → 归 `optimize-sync-elimination`（`P6`）。
