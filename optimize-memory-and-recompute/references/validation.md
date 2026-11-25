# 数值、显存余量、RSS 泄漏防线与性能验收

先约定项目数值容差、最坏输入形状、峰值显存红线与主机 RSS 监控方式，再修改代码。严格区分四层结论：**配置/开关已开启 ≠ 重算/显存/GC 行为实际改变 ≠ 数值与显存/RSS 安全等价 ≠ 端到端稳态提速**。

---

## 1. 数值与功能正确性

在相同权重、输入、随机种子及相同分布式/compile 设置下对照开关关闭与开启：

- **通用要求**：前向输出、loss、反向梯度与参数更新逐位一致或在项目既有容差内，无新增 NaN/Inf。
- **AC / SAC 调整（`M1` / `M2`）**：
  - 若改动同时将 `preserve_rng_state` 设为 `False` 且模型内含 dropout，须明确区分“无 dropout（逐位一致）”与“含 dropout（仅统计等价，需单独确认精度/任务指标）”；
  - SAC（`M2`）需验证前向与反向均命中预期策略且无张量版本冲突（inplace 修改被保存激活报错）。
- **关训练 KV Cache（`M3`）**：分别验证训练路径不分配 `DynamicCache` 且 loss/梯度不变，以及推理/评估传 `use_cache=True` 或非空 `past_key_values` 时仍正常工作。
- **激活视图与死分支裁剪（`M4` / `M5`）**：覆盖动态序列长度、混合 dtype 与 eval/辅助 loss 分支，确保裁剪分支不导致 eval 解包 `None` 或漏算有效 loss。

---

## 2. 两项硬安全防线：最坏形状峰值显存 + 长跑 RSS 泄漏核查

### 2.1 减 AC / 改 SAC 的最坏形状显存验收（硬门槛）
- 必须记录开关前后的 `torch.cuda.max_memory_allocated()`、`torch.cuda.max_memory_reserved()` 以及分配器 `num_alloc_retries`（来自 `torch.cuda.memory_stats()`）。
- 测试样本**必须覆盖训练中可能出现的最坏形状**（最大序列长度、最大图像分辨率/帧数、完整 batch size 与梯度累积窗口），并预留足够余量给后续并发特性（如 FSDP prefetch、`torch.compile`、checkpoint 保存）。
- 出现 OOM 或 `num_alloc_retries` 显著上升时，保留日志、出错步号与 rank，必要时在授权预算内采集显存快照交 `analyze-memory-snapshot` 诊断，严禁靠缩 batch 掩盖。

### 2.2 `gc.freeze()` 的主机 RSS 内存防线（硬门槛）
- 确认 `gc.freeze()` 在模型、优化器、DataLoader 及 warmup 步完成**之后**执行，且多卡先 `dist.barrier()` + `gc.collect(2)` 再冻结。
- 连续运行足够长步数（建议覆盖至少 200+ 步或数倍原 GC 周期），定期记录各 rank 主机 RSS 内存（如 `/proc/self/status` `VmRSS`），确认稳态期间 RSS 趋于平稳、无持续单调上涨；若 RSS 持续爬升，说明循环内存在真实引用环或对象累积，须关闭 `gc.freeze()` 并排查泄漏源。

---

## 3. 性能与抖动验收

- **吞吐与显存并报**：涉及 `M1`/`M2`/`M4`/`M7` 的改动，验收报告必须同时给出**无 Profiler 稳态 step 时间中位数**与**峰值显存变化（allocated / reserved）**。
- **抖动指标（针对 `M6` GC 冻结）**：不能只看步长中位数（P50），必须对比开关前后的 **P95 / P99 / Max 步长**与多卡步长尖刺频率，确认长尾抖动确实消除。
- **单变量消融**：禁止将 `torch.compile`、算子替换或通信重叠的收益混算到本 Skill；若 `M1`（如 `layer_stride`）与 `torch.compile` 交互，须同时检查重编译次数是否减少或增加。

---

## 4. 按生命周期界定的回退与离线边界

- **常规局部回退**：仅改纯函数分支开关（如 `M4`、`M5`、`M3` 包装）且无持久化状态时，关闭开关即可恢复。
- **必须重启进程并重建对象**：凡涉及以下任一项，切换开关做回退或 A/B 对照时**必须重启训练进程并重新构建模型**：
  1. 进程级分配器环境变量（`PYTORCH_CUDA_ALLOC_CONF` / `HIP_BUFFER_EXTRA_SIZE`）；
  2. `gc.freeze()` 已冻结的 Python 对象代状态；
  3. 初始化阶段安装的 `checkpoint_wrapper` / `with_cp` / `gradient_checkpointing` 层级包装；
  4. 与 `torch.compile` 联动捕获的编译图缓存。
- **无 GPU / 离线交付边界**：无目标硬件时仅完成静态扫描、代码接入与 CPU/mock 边界单测，在交付清单中明确标注“峰值显存、最坏形状 OOM 边界、长跑 RSS 趋势与吞吐收益均未验证”。
