# 显存、重算与 GC 抖动优化模式（M1–M7）

以下模式按 **Before 坏味道 → After 手法 → 前提与不适用 → 最小验证** 组织。所有 `[已核实改造案例]` 均已对照 TurboPhysAI 16 模型快照的 Before 基线与 After 实现源码核实（见 [turbophysai-memory-evidence.md](turbophysai-memory-evidence.md)），不含未复测的性能或显存收益承诺。

---

## M1 显存富余时削减或关闭冗余激活检查点（AC）`[已核实改造案例]`

- **Before 坏味道**：显存余量充足，但模型或骨干网络仍全层开启 activation checkpointing（如 `with_cp=True`、全层 `gradient_checkpointing=True` 或循环内每层调 `torch.utils.checkpoint.checkpoint`），导致反向阶段白白重算约一遍前向。
- **After 手法（由粗到细三档）**：
  1. **整模块关闭**：仅当实测峰值显存与最坏输入形状均留有充足余量时，将子模块 `with_cp` 设为 `False`（注意复制 config 避免原地污染调用方字典）。
  2. **按前/尾 N 层关闭**：关闭前 N 个小层或尾部 `tail_layers` 层的整层重算（`layer.gradient_checkpointing = False` 或跳过包装），在显存预算内线性回收计算量。
  3. **按步长间隔重算（`layer_stride`）**：每 `stride` 层仅 checkpoint 一次（`stride=0` 全关、`stride=2` 隔层重算）。**与 `torch.compile` 耦合要点**：若外层存在 `torch.compile`，不要对 `checkpoint` helper 做按层号分支的薄包装（Dynamo 会为每个层号产出独立图变体），应直接在 forward 循环里按步长分支调用或让 `stride=0` 走直连路径。
- **前提与不适用**：
  - **硬前置门槛**：必须先有基线实测峰值显存（`allocated` & `reserved`），并按**最坏序列长度 / 最大 batch**估算安全余量；不可用短步小输入代替验收。
  - 若后续计划叠加 FSDP `forward_prefetch`、`reshard_after_forward=False` 或 `torch.compile`，须把它们的显存增量一并计入预算。
- **最小验证**：同种子对比前向输出、loss 与梯度；记录最坏形状下峰值显存增量及无 Profiler 稳态步长。

---

## M2 选择性激活检查点（SAC）：只存重算最贵的算子 `[已核实改造案例]`

- **Before 坏味道**：整层 AC 重算代价过大（反复重算昂贵的 Attention 与大矩阵乘 GEMM），而直接关掉整层 AC 又会 OOM。
- **After 手法**：
  - 使用 PyTorch 标准 `create_selective_checkpoint_contexts` 与 `CheckpointPolicy`，对重算最贵的算子（如 `aten.mm.default`、`aten.addmm.default`、FlashAttention 前向）返回 `CheckpointPolicy.MUST_SAVE`，对廉价逐元素/归一化/激活算子返回 `CheckpointPolicy.PREFER_RECOMPUTE`。
  - 可与层间隔/尾部层策略组合（仅对末尾 `sac_layers` 层或按 `sac_layer_interval` 启用 SAC），且仅作用于已开启 AC 的层（未开 AC 的层本就由 autograd 保留激活，禁止二次缓存）。
  - 确认模型不需要逐层严格复现 dropout RNG 状态时，可将 `preserve_rng_state=False` 减少状态保存开销；涉及跨多个 eager 区域重放时，须维持 checkpoint `func_counter` 位置对齐。
- **前提与不适用**：
  - 需核实目标 PyTorch 版本对 `create_selective_checkpoint_contexts` 与自定义算子（如 `flash_attn`）的支持情况（「版本兼容性待核对」）。
  - 依赖 `use_reentrant=False`（非重入 checkpoint）；旧版 `use_reentrant=True` 与 FSDP/SAC 组合易触发挂起或梯度丢失。
- **最小验证**：对比全量 AC 与 SAC 的前向/反向梯度数值等价性、峰值显存变化与反向时间下降。

---

## M3 关闭训练期隐式 KV Cache `[已核实改造案例]`

- **Before 坏味道**：HuggingFace Transformers 等 LLM/VLM 骨干在 `config.use_cache` 默认为 `True` 且 `past_key_values is None` 时，训练前向（prefill）会隐式创建并填充 `DynamicCache`，白占显存与分配开销，且与 gradient checkpointing 冲突。
- **After 手法**：
  - 在训练配置中显式设 `model.config.use_cache = False`，或在骨干 `forward` 包装层检查：仅当调用方**未显式传入 `use_cache`** 且 **`past_key_values is None`** 时注入 `use_cache=False`。
- **前提与不适用**：
  - 绝不可无条件覆写显式传入 `use_cache=True` 或带 `past_key_values` 的自回归生成/推理路径，否则会破坏推理与评估速度。
- **最小验证**：训练路径确认未构造 `DynamicCache` 且 loss/梯度不变；推理/生成路径传 `use_cache=True` 时仍正常走缓存。

---

## M4 压缩中间激活副本与断开切片 `_base` 视图引用 `[已核实改造案例]`

- **Before 坏味道**：
  1. 拼接多个同 dtype 张量后再转 fp32 时，写成 `torch.cat([p.float() for p in parts], dim=1)`，导致先在显存里多分配一组逐段 fp32 副本再拷贝拼接。
  2. 对超大临时张量做切片 `local = big_tensor[:, start:end]` 后直接传给后续算子并被 autograd 保存用于反向——因为 `local._base` 仍指向整块 `big_tensor` 的底层 storage，整块大显存被迫活到反向结束才能释放。
- **After 手法**：
  - **同 dtype 先拼再转**：校验所有分片同 dtype 时改为 `torch.cat(parts, dim=1).float()`，混合 dtype 时回退原逐段转换。
  - **断开视图滞留**：对需要进入后续长生命周期/反向保存的切片，若 `local._base is not None` 显式执行 `local = local.clone(memory_format=torch.contiguous_format)`（否则仅在非连续时 `.contiguous()`），让上游大张量立刻可被回收。
- **前提与不适用**：
  - 仅对**大张量的小切片且上游大张量此后不再使用**的场景合算；若切片几乎等于整张量或上游张量本来就要留到反向，`.clone()` 反而额外增加显存与拷贝。
- **最小验证**：对比开关前后 `max_memory_allocated` 与数值逐位等价性。

---

## M5 裁剪训练期未消费的死分支与冗余状态遍历 `[已核实改造案例]`

- **Before 坏味道**：
  1. 多模态/教师分支在训练前向里顺带算了多个 head 或编码路径（如首帧条件 VAE 编码、MoGe `normal_head`），但当前训练 loss 从未消费这些输出，白白耗费显存与算力。
  2. 每步重复对深层模块树调用 `model.train()` / `set_frozen_modules_to_eval_mode()` 触发大量无意义递归遍历。
- **After 手法**：
  - 在确认训练 loss 与回调确实不消费对应输出的前提下，增加显式控制标志（如 `encode_y=False`、`_depth_only=True`），仅在训练分支跳过未用计算，保留评估/完整推理分支走原路径。
  - 对频繁重复的 `train/eval` 模式设置增加带失效钩子的状态缓存（子模块替换或显式切换模式时失效）。
- **强耦合原子性**：裁剪输出（返回 `None` 或减少返回键）必须与上游调用方解包逻辑成组核对与启停，防止下游在 eval 或特殊配置下解包 `None` 崩溃。
- **前提与不适用**：严禁在未核实全部训练/评估调用点及辅助 loss 开关时盲目跳过分支。
- **最小验证**：验证训练前向输出/loss/梯度与基线逐位一致，同时验证 eval/推理路径仍走完整分支不缺失字段。

---

## M6 Warmup 后冻结 Python GC，消除多卡周期性抖动 `[已核实改造案例]`

- **Before 坏味道**：Python 周期性触发第 2 代垃圾回收（扫描数万个长期存活的模型/配置/缓存对象），在单卡上造成数百毫秒停顿；在多卡分布式训练中，各 rank 触发 GC 的步号不同步，导致整集群被轮流拖慢，在 Trace 里常伪装成偶发慢 rank 等待。此外，训练循环每步无条件调用 `torch.cuda.empty_cache()` / `gc.collect()` 也会制造同步与分配器抖动。
- **After 手法**：
  - 在模型、优化器、DataLoader 初始化及若干 warmup 步（例如 `step >= 50`，对象图已稳定）完成后：多卡先 `dist.barrier()` 对齐，再一次性执行 `gc.collect(2)` 与 `gc.freeze()`，把长生命周期对象移出后续 GC 扫描范围（新产生的短命对象仍由常规 GC 回收）。
  - 检查并移除热路径中无依据的每步 `torch.cuda.empty_cache()`，如确有碎片清理需求应拉大间隔并验证必要性。
- **安全防线（必查）**：
  - 禁止在初始化未完成前过早冻结，也禁止用 `gc.disable()` 彻底关停 GC 而不排查泄漏。
  - 启用 `gc.freeze()` 后，必须长跑足够步数持续监控**主机 RSS 内存**，确认无单调持续增长（防止循环引用泄漏被掩盖）。
- **最小验证**：对比开启前后的步长 P50/P95/P99 与最大尖刺步长，并确认多卡步末 RSS 平稳无泄漏。

---

## M7 分配器防碎片配置（`expandable_segments`）`[已核实改造案例｜版本兼容性待核对]`

- **Before 坏味道**：显存快照或日志显示 `max_memory_reserved` 远大于 `max_memory_allocated`，或在动态序列长度下因显存碎片频繁触发分配器重试/OOM（即使总已分配显存远未达到卡容）。
- **After 手法**：
  - 进程启动前设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（视 PyTorch/ROCm 构建支持情况；部分 HCU/HIP 构建配套设置 `HIP_BUFFER_EXTRA_SIZE=0` 减少额外缓冲预留）。
- **前提与不适用**：
  - 属**进程级环境变量**，必须在 `import torch` 与首次显存分配前设置，且受 PyTorch、CUDA/ROCm 驱动与内核版本约束（「版本兼容性待核对」）；开启后若与 CUDA Graphs 或特定驱动不兼容须立即回退。
- **最小验证**：新进程对比开关前后的 `max_memory_reserved` / `allocated` 差值、分配器 `num_alloc_retries` 与稳态步长。

---

## 通用不适用场景

- 瓶颈在通信暴露、DataLoader 饥饿或步内 `.item()` 同步（应先交给对应专项，不要盲目关 AC 冒 OOM 风险）。
- 尚无任何基线峰值显存数据时，直接关闭全模型 AC。
- 涉及两阶段离线预编码落盘（如离线生成 VAE latent / text embedding 磁盘库）→ 归数据管线/训练工作流方向。
