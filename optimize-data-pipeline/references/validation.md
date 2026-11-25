# 样本等价、多流安全、主机资源防线与性能验收

先约定项目数据顺序要求、断点恢复契约、主机内存/`/dev/shm` 上限与数值容差，再修改代码。严格区分四层结论：**配置/开关已开启 ≠ 数据搬运或预取路径实际命中 ≠ 样本内容/顺序/多流安全等价 ≠ 端到端稳态提速**。

---

## 1. 样本内容、数据顺序与断点恢复正确性

在相同数据集、随机种子与分布式配置下对照开关关闭与开启：

- **样本逐字段等价（`D2`、`D5`、`D6`）**：同一索引产出的 batch 中，每个字段的 key、shape、dtype 与数值与基线一致（几何向量化允许浮点舍入在约定容差内）；列裁剪后确认无 `KeyError` 或丢失字段。
- **采样顺序与断点恢复（`D3`、`D4`、`D5`）**：
  - 采样器向量化或连续 epoch 改造必须验证：**同种子下生成的索引序列与基线逐项一致**；
  - 专门验证从非零 `epoch` / `start_index` / `skip_batches` 恢复训练时，首批及后续批次顺序与基线对齐，且不会被框架初始化的 `set_epoch(0)` 覆盖。
- **自定义容器 `pin_memory`（`D2`）**：验证 `cpu_only=True` 字段保持在 CPU 且未被改动，`cpu_only=False` 的内部 Tensor 确实满足 `is_pinned() == True`，且 `namedtuple`/自定义 `Mapping` 类型保持不变。

---

## 2. 三项硬安全防线：多流内存安全、进程死锁与主机资源

### 2.1 独立 Copy-Stream 异步预取安全防线（`D4`）
- **前提门禁**：确认非 CUDA/HIP、未 `pin_memory` 或未 `non_blocking` 时自动回退原路径，不静默启用。
- **生命周期与异常早退**：编写测试覆盖 **提前 `break`、循环内抛异常、空/单 batch epoch**，确认 `finally: copy_stream.synchronize()` 被执行，pinned 源张量在拷贝结束前保持存活，无内存读穿或残留线程。
- **`record_stream` 检查**：确认所有移入计算流的设备张量均递归调用了 `record_stream(consumer_stream)`，在长步数与动态 shape 下无随机 NaN/数据串批。

### 2.2 多进程与死锁防线（`D1`、`D3`）
- 含 TF/RLDS、HF tokenizers 或 OpenCV 线程池的场景，验证 `spawn` worker 能正常启动、跨多步迭代并干净退出（无僵尸 worker 残留，无 `fork` 死锁）。
- 核实 `num_workers == 0` 边界下不透传 `persistent_workers` / `prefetch_factor` / `multiprocessing_context`。

### 2.3 主机内存与 `/dev/shm` 防线（`D1`、`D4`、`D6`）
- 记录多卡满编运行时的主机总 RSS（`world_size × (主进程 + num_workers)`）与 `/dev/shm` 占用峰值，确认远低于宿主机/容器上限且稳态不持续攀升。
- 若使用 `D6` 离线缓存，额外核实磁盘空间预算、原子落盘（中断重跑不产生损坏 `.npy`/`.json`）以及缓存目录与预处理参数版本的隔离。

---

## 3. 性能验收与归因隔离

- **先验命中，再谈提速**：
  - 用 Profiler Trace 核对 `trace-bottleneck-report` §3（步首/数据空闲）与 §5（H2D 拷贝耗时、带宽与拷贝前空档）是否真实下降；
  - 确认 H2D 确实走 `cudaMemcpyAsync`（pinned）而非同步 pageable 拷贝。
- **无 Profiler 稳态测速**：关闭 Profiler 与额外校验，充分预热后报告多次测量的 **step 时间中位数、P90/P99 与跨 rank 最大值**（排查是否同时消除了慢 rank 拖尾）。
- **单变量消融**：不把 `torch.compile`、算子替换、同步消除或通信重叠的收益混算到数据管线优化上。

---

## 4. 按生命周期界定的回退与离线边界

- **常规局部回退**：仅涉及无状态纯函数替换（如 `D5` 单函数向量化）时，关闭开关即可恢复。
- **必须重启进程或重建 DataLoader**：凡涉及以下任一项，切开关回退或做 A/B 对照时**必须重建 DataLoader（涉及全局状态时重启训练进程）**：
  1. `num_workers`、`persistent_workers`、`pin_memory`、`multiprocessing_context` 等 DataLoader 构造参数；
  2. 全局 `pin_memory` 递归入口包装（`D2`）或 `TOKENIZERS_PARALLELISM` 环境变量；
  3. 已启动的后台预取线程与独立 `copy_stream`（`D4`）；
  4. 冻结编码器离线缓存切换或 ZeRO-2 前瞻在线编码包装（`D6`、`D7`）。
- **无 GPU / 离线交付边界**：缺少目标 GPU 或真实大规模数据集时，仅完成静态扫描、代码接入与 CPU/mock 单元测试（覆盖参数边界、容器递归、采样顺序与异常早退），在交付清单中明确标注“设备 H2D 吞吐、主机 RSS/`/dev/shm` 峰值与端到端稳态收益均未验证”。
