# 数据管线与异步 H2D 通用模式

每条模式均对照本地 Before（基线）与 After（优化）源码核实，并注明证据级别与适用边界；不含未复测的性能数字（具体案例索引见 [turbophysai-data-evidence.md](turbophysai-data-evidence.md)）。

---

## D1 DataLoader 并发、固定内存与非阻塞开关（已核实改造案例）

- **Before 坏味道**：
  - 训练 `DataLoader` 显式设 `pin_memory=False`、`persistent_workers=False`、`num_workers=0`，或未设合理的 `prefetch_factor`；
  - Accelerate 默认构造 `Accelerator()` 时未传 `DataLoaderConfiguration(non_blocking=True)`，导致内部 `send_to_device(..., non_blocking=False)` 每次拷贝都阻塞；
  - 多 worker `fork` 与 Hugging Face `tokenizers` 并行线程冲突引发死锁或警告。
- **After 手法**：
  - `num_workers > 0` 时开启 `pin_memory=True`、`persistent_workers=True`、`prefetch_factor=2`；若 `num_workers == 0`，**必须剔除** `persistent_workers`、`prefetch_factor`、`multiprocessing_context` 等仅多进程合法的参数，防止构造期抛 `ValueError`；
  - Accelerate 路径将 `DataLoaderConfiguration(non_blocking=True)` 传入（或通过 `dataclasses.replace` 保留调用方其他字段）；
  - 使用 HF tokenizer 且多 worker 时显式设置 `TOKENIZERS_PARALLELISM=false`。
- **前提与不适用**：
  - `pin_memory=True` 与过大 `num_workers`/`prefetch_factor` 会线性放大主机锁页内存与 `/dev/shm` 占用；多卡单机（如 8 卡）总 worker 数 = `world_size × num_workers`，不可盲目调高导致主机 OOM 或 NUMA 跨节点争抢；
  - 仅设 `non_blocking=True` 而源张量未 pinned 时不会真异步（见 `D2`）。
- **最小验证**：确认 `batch` 抵达训练步前张量 `is_pinned()`（CPU 侧）且无 `ValueError`/worker 泄漏；连续跑跨 epoch 验证无重建停顿与 `/dev/shm` 溢出。

---

## D2 自定义 Batch 容器的 `pin_memory` 适配（已核实改造案例）

- **Before 坏味道**：
  - 框架使用自定义批容器（典型如 MMCV `DataContainer`），其类未实现 `pin_memory()` 方法；PyTorch 原生 `DataLoader` 的 pin-memory 线程遇到未知容器会直接跳过，导致开启 `pin_memory=True` 后内部 Tensor 仍是 pageable 内存。
- **After 手法**：
  - 为自定义容器实现 `pin_memory()` 或包装全局 `pin_memory` 递归入口：仅对 `not data.cpu_only` 的容器内部递归固定内存；遇到已在加速设备或已 `is_pinned()` 的 Tensor 直接返回；
  - 保留原生容器语义（如 `namedtuple`、自定义 `Mapping` 子类），并做幂等防重入检查避免同一 batch 被双重遍历。
- **前提与不适用**：
  - `cpu_only=True` 的元数据字段（如图像元信息、字符串、路径）绝对不能强行 pin 或上卡。
- **最小验证**：构造含普通 Tensor、`cpu_only=True`、`cpu_only=False` 嵌套自定义容器的样本，验证前者不变、后者内部 Tensor `is_pinned() == True` 且值与结构一致。

---

## D3 含活跃线程池数据集的 `spawn` Worker + 可序列化重建（已核实改造案例）

- **Before 坏味道**：
  - 数据集底层依赖 TensorFlow / RLDS 或其他在构造期即拉起 C++ 线程池的库，为避免 `fork` 复制活跃线程池引发死锁，上游被迫写死 `num_workers=0`，数据读取完全串行卡死主训练线程。
- **After 手法**：
  - 仅为该特定数据集的 `DataLoader` 注入 `multiprocessing_context=torch.multiprocessing.get_context("spawn")`、`num_workers > 0` 与 `pin_memory=True`（不影响 eval 或普通数据集）；
  - 为数据集实现 `__reduce__`：只序列化构造参数 `(module, qualname, args, kwargs)`，在 `spawn` 出来的干净子进程中重新构造数据集与底层 TF/C++ 图，并正确传入 worker 级随机种子。
- **前提与不适用**：
  - 构造参数本身必须可 pickle；若数据集对象携带不可重建的运行时状态，则不能简单按构造参数重建；
  - `spawn` 启动比 `fork` 慢，需配合 `persistent_workers=True` 或长步数训练摊薄进程拉起开销。
- **最小验证**：真实跑通多 worker 启动、取多批数据与正常退出（无死锁/pickle 错误）；对比单/多 worker 下样本格式、dtype 及可复现性（若要求严格数据顺序，须核实 worker 分片策略）。

---

## D4 独立 Copy-Stream 异步 H2D 预取流水（已核实改造案例）

- **Before 坏味道**：
  - 即使开启了 `pin_memory=True` 与 `non_blocking=True`，若 H2D 拷贝仍在默认计算流上逐批发起（如 Accelerate 默认 `DataLoaderShard.__iter__`），拷贝仍可能排在上一计算 kernel 之后或与主机准备串行。
- **After 手法**：
  - 仅在同时满足 **CUDA/HIP 设备 + `pin_memory=True` + `non_blocking=True`** 时启用独立 `copy_stream` 预取（不满足则回退原迭代器）：
    1. 在 `with torch.cuda.stream(copy_stream):` 内调用 `send_to_device(..., non_blocking=True)`，并在 `copy_stream` 上 `event.record(copy_stream)`；
    2. 消费端在当前计算流执行 `consumer_stream.wait_event(event)`，并对 batch 内全部设备张量递归调用 `tensor.record_stream(consumer_stream)`（防分配器把仍在计算流使用的显存提前复用）；
    3. **主机 pinned 源张量保活**：维护 `pending_sources` 列表持有 `(source_batch, event)`，仅在 `event.query()` 为真后释放，且在 `try ... finally:` 中执行 `copy_stream.synchronize()`，确保 `break`/异常早退时主机内存不被 DMA 读穿；
    4. 若配后台预取线程（如 `CudaPrefetchLoader`），必须捕获后台线程异常并在主线程 `join()` 时重新抛出，防止静默吞错。
- **前提与不适用**：
  - 若源张量未 pinned 或单卡显存极度紧绷（多驻留 1 个 device batch 即 OOM），不应启用；
  - 必须保持原框架的 `set_epoch`、RNG 状态同步、`skip_batches` 与 `state_dict` 更新时机完全一致。
- **最小验证**：对照关闭/开启路径的逐 batch 数值与顺序完全一致；专门测试提前 `break`、抛出异常、空/短 epoch 与 `skip_batches` 恢复场景下无段错误、无显存/pinned 泄漏。

---

## D5 按需列解码与 CPU 采样/几何预处理向量化（已核实改造案例）

- **Before 坏味道**：
  - **HF Dataset 整行误解码**：`self.hf_dataset[indices][key]` 或 `self.hf_dataset[idx]` 会触发整行所有列（包括无需在此处解码的图像/视频列）解码；
  - **低效采样器与跨 epoch 停顿**：Python 循环逐个展开 episode 帧索引，或每轮 epoch 结束重新构建 `BatchSamplerShard` 导致周期性卡顿；
  - **CPU 几何/GT 构造循环**：在 `__getitem__` 里用 Python 循环调 Shapely `LineString.interpolate` 等逐点分配对象，把多核 CPU worker 打满。
- **After 手法**：
  - **按需列裁剪**：改用 `hf_dataset.select_columns([key])` 或惰性列视图 `_ColumnView`，跳过已被视频解码器或其他路径处理的图像列；
  - **向量化与连续采样**：用 `torch.randperm` + `np.searchsorted` 替代逐条循环生成采样索引；需要消除 epoch 边界停顿时，在严格校验 `split_batches=False, even_batches=True, drop_last=False` 且保留 checkpoint `start_index`/`epoch` 恢复语义的前提下衔接连续 epoch；
  - **NumPy 向量化重采样**：用 `np.hypot` + `np.cumsum` + `np.interp` 一次性计算整条折线采样点，精确保留基线端点重复与数值契约。
- **前提与不适用**：
  - 采样器改写必须与基线**逐索引一致**（同种子、同 epoch、同断点 `start_index`），不能悄悄改变数据分布或破坏 resume；
  - 列裁剪需确保被跳过的列确实由其他分支读取，不会导致下游 `KeyError`。
- **最小验证**：同种子对比改前/改后采样序列逐元素相等（含断点恢复场景）；对比列裁剪/向量化前后单个 batch 全部字段的 shape、dtype 与数值误差（容差内等价）。

---

## D6 冻结编码器两阶段离线特征缓存（受限工作流参考）

- **Before 坏味道**：
  - 多模态/世界模型训练中包含重量级冻结编码器（如视频 VAE、T5 文本编码器），且连续训练窗口高度重叠，每步反复解码同一段视频帧并重复执行相同冻结前向。
- **After 手法**：
  - 拆为两阶段工作流：Step 1 多卡按 episode 分片一次性解码与编码，通过**临时文件 + 原子重命名**写盘（`.npy` + 汇总 `index.json`），支持中断续跑；Step 2 训练时按索引直接读缓存并将冻结编码器保留在 CPU。
- **强前提与不适用（必须逐条核实）**：
  1. 编码器必须**严格冻结且处于 `eval()` 模式**；
  2. 编码前**不存在在线随机数据增强**（随机裁剪、翻转、色彩抖动等会让每 epoch 输入不同，绝对不可静态离线缓存）；
  3. 需提前核算磁盘容量、IO 带宽与缓存一致性键（模型权重/预处理参数变更必须隔离缓存目录）。
- **最小验证**：抽样对比在线编码输出与落盘再读回的特征逐位/容差内一致；验证中断后重跑可正确跳过已完成文件且不留半写损坏文件。

---

## D7 冻结编码器单流前瞻在线编码（受限 / 单模型参考）

- **机制说明**：在 DeepSpeed ZeRO-2 `optimizer.step` 异步发起 `all_gather_dp_groups` 后，在同一训练流上用冻结的 VAE/T5 预先编码下一 batch，再等待 AllGather 完成；单流执行无需副流、`Event` 或 `record_stream`。
- **强前提与限制**：
  - 强绑定特定模型结构与 DeepSpeed ZeRO-2 内部 `all_gather_dp_groups` 调用形态；
  - 冻结编码器参数**绝对不能**属于 ZeRO 优化器或 AllGather 分组（否则读写冲突）；
  - 要求 `gradient_accumulation_steps == 1`、`num_workers >= 1` 且预处理无未对齐随机状态；
  - 迁移到新项目前必须按「受限/单模型参考」重新核实全部前提，并与 `optimize-distributed-comm` 协同验收。

---

## 通用不适用场景

- Trace 表明 GPU 已经持续打满（步首无明显空闲、H2D 占比极低且未暴露）：优化数据管线不会带来端到端提速。
- 瓶颈实际在训练步内的 `.item()`/`nonzero` 同步或集合通信未重叠：分别交由 `optimize-sync-elimination` 或 `optimize-distributed-comm` 主导。
- 主机内存或 `/dev/shm` 已接近上限：盲目叠加 `num_workers`、`prefetch_factor` 与 `pin_memory` 反而会触发主机 OOM Killer 或 swap 抖动。
