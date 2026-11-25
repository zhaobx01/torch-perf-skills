---
name: optimize-data-pipeline
description: Analyze or implement PyTorch training data-pipeline and host-to-device (H2D) feeding optimizations: DataLoader worker/pin_memory/persistent_workers/non_blocking tuning, custom batch container pinning, spawn-safe workers for thread-pool datasets (e.g. TF/RLDS), copy-stream async H2D prefetching with stream/lifetime guards, dataset column pruning and CPU sampler/geometry vectorization, and frozen-encoder offline feature caching. Use when profiler traces show step-start GPU idle gaps, slow or blocking H2D copies, or CPU DataLoader/decoding bottlenecks. Hand off step-internal D2H scalar syncs (`.item()`, `nonzero`) to `optimize-sync-elimination`, collective comm overlap to `optimize-distributed-comm`, and AC/memory/GC jitter to `optimize-memory-and-recompute`.
---

# Optimize Data Pipeline

以目标项目的真实 Trace 步首空闲、H2D 搬运证据与数据加载调用链为依据，消除 DataLoader 供给饥饿、阻塞式 H2D 拷贝与低效 CPU 数据解码/采样瓶颈，同时严守数据顺序、断点恢复、多流内存安全与防进程死锁边界。

## 1. 定位与边界

- **目标领域**：从磁盘/数据集读取到送入计算流之前的数据供给全链路，按改动成本与风险由低到高分层覆盖：
  1. **DataLoader 基础并发与固定内存（`D1`）**：`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor`、`non_blocking=True` 与 `TOKENIZERS_PARALLELISM=false`；
  2. **自定义容器 `pin_memory` 适配（`D2`）**：修复 MMCV `DataContainer` 等自定义批结构绕过标准 pin 线程导致 `non_blocking` 失效的问题；
  3. **多进程 `spawn` 与可序列化重建（`D3`）**：为底层含活跃线程池（如 TF/RLDS）的数据集安全启用多 worker，避开 `fork` 死锁；
  4. **独立 Copy-Stream 异步 H2D 预取（`D4`）**：在 pinned + non-blocking 前提下用独立拷贝流与 `Event` 做单步前瞻搬运，并严格执行 `record_stream` 与主机源保活；
  5. **按需列解码与 CPU 采样/几何预处理向量化（`D5`）**：消除 HF Dataset 无关图像列解码、Python 循环采样/重采样与跨 epoch 分片重建停顿；
  6. **冻结编码器离线缓存 / 前瞻在线编码（`D6` / `D7`）**：仅在编码器严格冻结、无在线随机增强（或满足强并发前提）时评估的受限工作流改造。
- **核心作用**：缩短 step 头部数据等待与 H2D 暴露时间，在不改变有效训练样本分布、resume 顺序和设备内存安全的前提下提升 GPU 供给连续性。
- **何时使用**：
  - `trace-bottleneck-report` §3 显示步首/数据加载大空闲，或 §5 显示大块 H2D 拷贝暴露、拷贝前 CPU `copy_` 阻塞或带宽偏低；
  - 源码中出现 `DataLoader(..., num_workers=0)`、`pin_memory=False`、`persistent_workers=False`、Accelerate 默认 `non_blocking=False`、`hf_dataset[idx]` 整行读取或纯 Python 循环采样/几何重采样；
  - 多卡训练因某 rank 数据读取/解码变慢导致慢 rank 偏斜。
- **明确不做什么与多 Skill 分流规则**：
  - **训练步内 D2H 同步（`optimize-sync-elimination`）**：等待下一个 batch 或阻塞式 H2D 拷贝归本 Skill 主导；训练步计算期间因 `.item()` / `.cpu()` / `.tolist()` / 布尔索引触发的 D2H 读回归 `optimize-sync-elimination` 主导。
  - **分布式通信重叠（`optimize-distributed-comm`）**：AllReduce/AllGather/FSDP/ZeRO-2 通信重叠归通信专项主导；涉及冻结编码器与 ZeRO-2 AllGather 重叠（`D7`）时须协同核对。
  - **显存、AC 与 GC 抖动（`optimize-memory-and-recompute`）**：激活重算、峰值显存与多卡 `gc.freeze()` 归显存专项主导；但本 Skill 启用大 `prefetch_factor`、`pin_memory` 或双缓冲预取流时须协同评估主机 RSS 与显存增量。
  - **图编译与定长填充契约（`torch-compile-integration`）**：为稳住编译 shape 而引入的 padding/分桶契约（如 `pad_156`、`text_len_bucket`）归编译专项主导。

## 2. 工作模式与交付契约

按**本次用户授权**判定模式，不以调用次数机械限制：

### 2.1 只读分析模式（默认）

未明确授权修改源码或启动训练时，**不修改项目源码、不启动训练或采集**。

- **必填交付要素**：涉及位置与符号、建议改法（模式编号）、证据级别（`[实测]` / `[静态]` / `[推断]`）、关键前提与风险（内存/死锁/数据顺序/多流安全）、最小验证项、缺失证据。
- **默认版式（多候选评审）**：使用 **4 列主表 + Top 1–2 短卡片**——

  | 优先级 | 位置与符号 | 建议改法 | 证据级别 |
  | --- | --- | --- | --- |
  | P0 / P1 / 暂缓 | `file:line` + 类/函数或配置键 | 一句话说明改写方向与模式（如 `D1`） | `[实测]` / `[静态]` / `[推断]` |

  仅对最值得先动的 1–2 项补充 3 行短要点：**关键前提与风险**、**最小验证项**、**缺失证据/待核实项**；低优项不展开，不默认刷屏大段 diff。单点咨询可直接简洁作答；无可靠候选时只列排查范围与缺失证据。

### 2.2 实施模式（授权后修改与验收）

面向通用 PyTorch 项目直接最小侵入修改（若目标本身是 TurboPhysAI 项目，则顺从其 `OptimizationGroup` / recipe 架构）：

- **工程硬约束**：提供默认安全的可关闭开关与明确回退路径；不混入同步消除、编译或算子替换收益；严格拆分四层结论——**配置/开关已开启 ≠ 数据搬运/预取路径实际命中 ≠ 数据顺序与数值/多流安全等价 ≠ 端到端稳态提速**。
- **固定 4 项交付清单**：
  1. **实际改动与启停开关**（文件、默认状态与启停方式）；
  2. **已跑检查与结果**（数据等价/顺序/resume 测试、多流与早退测试、Trace 步首空闲/H2D 变化与无 Profiler 稳态测速）；
  3. **未验证项与限制**（无 GPU 或缺真实数据集时明确标“设备 H2D 吞吐、主机 RSS 与端到端提速未验证”）；
  4. **回退方式（按生命周期说明）**：涉及 DataLoader/worker 池、`multiprocessing_context`、已安装全局 `pin_memory` 包装或预取流/线程状态时，须重启进程或重建 DataLoader 实例。

## 3. 诊断与实施流程

1. **先确认瓶颈在供给侧**：有 Trace 时优先看 `trace-bottleneck-report` §3（步首/数据空闲）与 §5（H2D 搬运）；无 Trace 时可先运行 [scripts/scan_data_pipeline_candidates.py](scripts/scan_data_pipeline_candidates.py) 产出静态候选，并明确标为 `[静态]`。
2. **按风险由低到高逐级筛查**（详见 [references/patterns.md](references/patterns.md)）：
   - 先查 `D1`（DataLoader 并发/`pin_memory`/`persistent_workers`/`non_blocking`）与 `D2`（自定义容器是否绕过 pin 线程）；
   - 再查 `D3`（TF/RLDS 是否因防 `fork` 死锁被迫 `num_workers=0`）与 `D5`（HF 整行误解码、慢采样器、Python 几何循环）；
   - 仅在 `pin_memory + non_blocking` 已生效但 H2D 仍显著暴露时，再评估 `D4`（独立 Copy-Stream 异步预取，强制配齐 `Event` + `record_stream` + 主机源保活 + `finally` 同步）；
   - 仅当冻结编码器本身成为主要供给瓶颈且确认无在线随机增强时，再讨论 `D6`（离线缓存）或 `D7`（前瞻在线编码）。
3. **验证与回退**：按 [references/validation.md](references/validation.md) 执行样本内容/顺序/断点恢复核对、主机 RSS 与 `/dev/shm` 防线检查、早退清理验证及无 Profiler 稳态测速；若主机内存暴涨、worker 僵死或端到端无收益，按生命周期关闭回退。

## 4. 参考与脚本

- [references/patterns.md](references/patterns.md)：`D1`–`D7` 通用改写模式（Before 坏味道 → After 手法 → 前提/反例 → 最小验证）
- [references/validation.md](references/validation.md)：样本与顺序等价、多流/早退安全、主机内存与 `/dev/shm` 防线及性能验收
- [references/turbophysai-data-evidence.md](references/turbophysai-data-evidence.md)：TurboPhysAI 16 模型跨模型数据管线证据与陷阱
- [scripts/scan_data_pipeline_candidates.py](scripts/scan_data_pipeline_candidates.py)：纯标准库静态扫描脚本（`--json` / `--min-severity` / `--no-configs`，仅产出静态候选）
- [scripts/selftest.py](scripts/selftest.py)：扫描脚本正/负例与边界自测
