---
name: optimize-memory-and-recompute
description: Analyze or implement PyTorch training memory and recomputation optimizations: bidirectional activation checkpointing tuning (disabling redundant AC, layer/stride reduction, selective AC/SAC), disabling training-time KV cache, trimming intermediate activation views and unused forward branches, freezing Python GC after warmup to eliminate multi-rank jitter, and allocator fragmentation tuning. Use when memory headroom allows trading memory for less recompute, when facing OOM or allocator fragmentation, or when multi-rank training shows periodic step-time spikes. Hand off offline encoder pre-caching to `optimize-data-pipeline`, compile boundaries to `torch-compile-integration`, and FSDP/ZeRO comm overlap to `optimize-distributed-comm`.
---

# Optimize Memory And Recompute

以目标项目的实测峰值显存、重算占比与步长抖动证据为依据，在“**显存富余减重算换吞吐**”“**显存紧绷压激活/防碎片/上 SAC**”“**消除周期性 GC 长尾抖动**”三个方向上安全调优，不以隐藏 OOM 风险或主机内存泄漏为代价换取短步提速。

## 1. 定位与边界

- **目标领域**：
  1. **激活检查点（AC / SAC）双向调优**：关冗余 AC、按层/步长削减 AC，或用选择性重算（SAC）仅保存高代价算子输出；
  2. **训练期激活显存与隐式缓存瘦身**：关训练 prefill 隐式 KV cache、减少临时 cast 副本、断开切片 `_base` 视图防大张量滞留、分配器防碎片配置；
  3. **训练期未消费死分支裁剪**：跳过下游不参与 loss 的冗余编码/head 分支；
  4. **Python GC 冻结与清缓存抖动治理**：warmup 后全卡对齐 `gc.collect(2)` + `gc.freeze()`，消除多卡不同步 GC 尖刺，排查热路径过度 `empty_cache()`。
- **核心作用**：把闲置显存安全转化为吞吐，或在显存受限时释放空间（同时为 FSDP 预取、`torch.compile` 腾出余量），并消除周期性步长尖刺。
- **何时使用（三向症状分流）**：

  | 观察到的主症状 | 核心判断 | 优先方向 |
  | --- | --- | --- |
  | 峰值显存明显低于卡容（有安全余量），且反向耗时占比高/已全层开 AC | 重算过重，可用显存换吞吐 | `M1`（关整模块 AC / 关前或尾 N 层 / `layer_stride`）或 `M2`（全量 AC 改 SAC） |
  | 峰值显存紧绷、OOM、或 `reserved >> allocated` 碎片严重，需为 FSDP/compile 腾显存 | 激活或碎片压力大 | `M3`（关训练 KV cache）、`M4`（压激活副本与断切片视图）、`M7`（`expandable_segments`）、`M2`（精细 SAC）、`M5`（剪死分支） |
  | 稳态步长正常，但每数十步出现周期性单步尖刺或多卡偶发慢 rank 等待 | 疑似 Python 2 代 GC 停顿或热路径 `empty_cache()` | `M6`（warmup 后多卡对齐 `gc.freeze()`，收缩过度 `empty_cache()`） |

- **明确不做什么与多 Skill 分流规则**：
  - **两阶段离线编码落盘缓存（`optimize-data-pipeline`）**：如离线预编码 VAE latent / T5 embedding 并改写数据集读取，交由 `optimize-data-pipeline` 主导，不在本 Skill 展开。
  - **分布式通信重叠（`optimize-distributed-comm`）**：FSDP `reshard_after_forward`、prefetch 与 ZeRO-2 缓冲调整归通信专项主导；本 Skill 负责为其腾出显存或协同核算总显存预算。
  - **图编译（`torch-compile-integration`）**：编译范围与图断治理归编译专项主导；但涉及 AC 与 `torch.compile` 交互（如按层号循环触发多图变体、checkpoint 放编译外层）须协同核对。
  - **步内同步消除（`optimize-sync-elimination`）**：`.item()`、`.tolist()`、微步 `dist.barrier()` 归同步消除主导。

## 2. 工作模式与交付契约

按**本次用户授权**判定模式，不以调用次数机械限制：

### 2.1 只读分析模式（默认）

未明确授权修改源码或启动训练时，**不修改项目源码、不启动训练或采集**。

- **必填交付要素**：涉及位置与符号、建议改法（模式编号）、证据级别（`[实测]` / `[静态]` / `[推断]`）、关键前提与显存/泄漏风险、最小验证项、缺失证据。
- **默认版式（多候选评审）**：使用 **4 列主表 + Top 1–2 短卡片**——

  | 优先级 | 位置与符号 | 建议改法 | 证据级别 |
  | --- | --- | --- | --- |
  | P0 / P1 / 暂缓 | `file:line` + 类/函数名 | 一句话说明改写方向与模式（如 `M1`） | `[实测]` / `[静态]` / `[推断]` |

  仅对最值得先动的 1–2 项补充 3 行短要点：**关键前提与风险**、**最小验证项**、**缺失证据/待核实项**；低优项不展开，不默认刷屏大段 diff。单点咨询可直接简洁作答；无可靠候选时只列排查范围与缺失证据。
- **缺证据取证降级协议（含两项硬前置门槛）**：
  - **减 AC / 改 SAC 前置门槛**：缺少目标配置（尤其最坏序列长度/最大 batch）的基线峰值显存实测数据（`max_memory_allocated` 与 `max_memory_reserved`）时，**允许暂缓削减 AC**，不强凑 P0，先给出最小显存摸底方案（短步记录峰值显存或采集显存快照交 `analyze-memory-snapshot`）。
  - **GC 冻结前置判断**：缺少多步步长分布或慢 rank 日志时，将 `gc.freeze()` 标为 `[静态]` 低优候选，不宣称存在 GC 抖动。
  - 查本地版本与读已有日志/快照可直接执行；启动训练或采集快照须先获用户授权。
- **特殊标注**：存在联动契约的改动（如裁剪死分支与上游返回解包、AC 调整与 `torch.compile` 包装顺序）必须**成组评估**；为防 OOM 保留的 AC 或 checkpoint 前同步不得误标为冗余删掉。

### 2.2 实施模式（经明确授权后）

- **目标代码形态**：默认面向**通用 PyTorch 项目**（最小侵入修改并接入可关闭开关）；若目标本身是 TurboPhysAI 接入任务，顺从其 `OptimizationGroup` / recipe 架构。
- **工程硬约束与两项安全防线**：
  1. **显存余量防线**：任何削减 AC / 放宽 SAC 的改动，必须在基线实测峰值显存与最坏输入形状下核对安全余量，并同时回报吞吐变化与峰值显存增量。
  2. **GC 泄漏防线**：实施 `gc.freeze()` 必须在模型/优化器初始化及 warmup 完成后执行（多卡先 `dist.barrier()` + `gc.collect(2)`），并将“长跑多步确认主机 RSS 内存无持续单调增长”列为必查验收项。
  3. **四层结论严格拆分**：`开关/配置存在` ≠ `重算/显存/GC 行为实际改变` ≠ `数值与显存/RSS 安全等价` ≠ `端到端稳态提速`。
  4. **无 GPU / 缺依赖诚实交付**：无目标硬件时仅完成静态接入与 CPU/mock 单测，明确标注峰值显存、OOM 边界与性能收益为“未验证”。
- **固定 4 项交付清单**：
  1. **实际改动与启停开关**（文件、默认状态与启停方式）；
  2. **已跑检查与结果**（数值/梯度对照、最坏形状峰值显存 `allocated/reserved`、RSS 趋势、无 Profiler 稳态测速）；
  3. **未验证项与限制**（明确缺口、形状/序列上界与叠加其他优化的显存余量）；
  4. **回退方式（按生命周期说明）**：关闭开关恢复原路径；凡涉及进程级分配器环境变量（`PYTORCH_CUDA_ALLOC_CONF`）、`gc.freeze()` 状态、模型层级 `checkpoint_wrapper` 包装或已捕获编译图时，**切换开关后必须重启进程并重新构建模型**。

## 3. 主干流程

1. **按三向症状定基线**：收集当前 `max_memory_allocated` / `max_memory_reserved`、是否启用 AC/SAC、步长长尾分布（及可选显存快照 / `trace-bottleneck-report` 慢 rank 数据）与 PyTorch/后端版本。
2. **静态初筛**：运行 [scripts/scan_memory_candidates.py](scripts/scan_memory_candidates.py) 快速定位 AC/checkpoint 入口、`use_cache=True` 与热路径 `empty_cache()`/`gc.collect()`（全部标为 `[静态]`）。
3. **匹配模式与核对耦合**：阅读 [references/patterns.md](references/patterns.md)（`M1`–`M7`）选择方案并核对与 `torch.compile`/FSDP 的耦合陷阱；需要对照 16 模型真实实现时查阅 [references/turbophysai-memory-evidence.md](references/turbophysai-memory-evidence.md)。
4. **实施与显存/数值/RSS 联合验收**：按 [references/validation.md](references/validation.md) 完成数值、最坏形状峰值显存、长跑 RSS 与多轮稳态性能验收；OOM 或 RSS 泄漏时立刻按生命周期回退。

## 4. 参考资料索引（按需加载）

| 文档 / 脚本 | 何时读取或运行 |
| --- | --- |
| [references/patterns.md](references/patterns.md) | 选型与设计方案，核对 `M1`–`M7` 的 Before 坏味道、前提、与 compile/FSDP 交互及最小验证时 |
| [references/validation.md](references/validation.md) | 实施改造、设计数值/最坏形状显存/RSS 泄漏防线/稳态测速与重启回退时 |
| [references/turbophysai-memory-evidence.md](references/turbophysai-memory-evidence.md) | 对照 16 模型快照中已核实的跨模型实现、PR/基线 SHA 与陷阱细节时 |
| [scripts/scan_memory_candidates.py](scripts/scan_memory_candidates.py) | 快速定位 AC/checkpoint、`use_cache=True` 与热路径 `empty_cache`/`gc.collect`（纯标准库，仅静态候选） |
| [scripts/selftest.py](scripts/selftest.py) | 校验 `scan_memory_candidates.py` 的真阳性与误报边界 |
