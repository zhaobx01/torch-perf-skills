---
name: optimize-memory-and-recompute
description: >-
  Analyze or implement PyTorch training memory and recomputation optimizations.
  Use for reducing activation recompute when memory permits, diagnosing
  activation or allocator pressure, or addressing step jitter linked to GC or
  allocator activity. Route offline feature caches to optimize-data-pipeline,
  graph capture to torch-compile-integration, and collective overlap to
  optimize-distributed-comm.
---

# Optimize Memory And Recompute

在目标训练路径上平衡激活显存与重算开销，并处理已定位的 GC 或分配器抖动。选型以最坏输入形状、实际 checkpoint 路径和稳态步时为依据。

## 定位与分流

| 触发信号 | 优先方向 |
| --- | --- |
| 最坏输入形状有显存余量，且反向能观察到大量激活重算或源码明确启用全层 AC | M1 缩小 AC 范围，或 M2 用选择性重算（SAC）保留高代价算子输出 |
| 训练 OOM、激活峰值高、分配重试或快照显示碎片 | M1 增加必要的 AC，M2 收紧已有 SAC，M3–M5 核对缓存、副本和未消费计算，M7 核对分配器 |
| 周期性步时尖刺与 GC 收集事件或显存清缓存调用对齐 | M6 评估冻结时机或缓存回收频率 |

按主瓶颈决定主导 Skill。有多 rank Trace 且到达集合操作的时间不齐时，先用 `trace-bottleneck-report` 定位慢 rank，区分 GC、分配器、数据等待、步内同步和通信。只有尖刺形状时先列静态候选，有证据指向 GC 或分配器时本 Skill 再主导。

- 离线 VAE/T5 编码与训练数据切换、等待 batch 或 H2D 归 `optimize-data-pipeline`。步内 D2H 或多余同步归 `optimize-sync-elimination`。
- FSDP/ZeRO 分片拓扑与集合重叠归 `optimize-distributed-comm`。本 Skill 核算 AC、权重常驻和预取叠加时的显存预算。
- 捕获边界、图断与重编译归 `torch-compile-integration`。卷积布局归 `optimize-channels-last`，算子后端选择归 `optimize-kernel-replacement`。遇到 AC 与这些改动共用入口时，联合核对并做消融。

## 按当轮授权工作

### 只读分析

未授权改源码时只读分析。启动训练、补采 Trace 或快照、安装依赖各自以当轮用户授权为准。可直接查本地版本、源码、已有日志与已有 Trace。

多候选默认用 **4 列主表 + Top 1–2 短卡片**。单点直接回答，无可靠候选时说明排查范围和证据缺口。

| 优先级 | 位置与符号 | 建议改法 | 证据级别 |
| --- | --- | --- | --- |
| P0 / P1 / 暂缓 | 文件、行号与类/函数 | 方向及对应 M1–M7 | [实测] / [静态] / [推断] |

高优候选补关键前提和风险、最小验证项、缺失证据。静态源码和语料注释对应静态证据，运行结论以目标配置的对照为准。

- **缺显存证据**：暂缓削减 AC 或放宽 SAC。说明最坏序列、完整 batch 和累积窗口缺哪些数据。先核对已有峰值与重试记录，获得授权后再在目标配置下做短步峰值摸底。余量不足时保留原 AC，并评估省显存方向。
- **缺 GC 或分配器归因**：把冻结、清缓存频率和分配器配置变体列为低优静态候选。给出对齐慢 rank 步号与 GC 事件、清缓存调用和分配重试的最小取证方案，拿到归因后再选型。
- **缺目标依赖或后端版本**：先查本地；如已有快照且已安装 `analyze-memory-snapshot`，可协同分析。否则按包内峰值、重试与快照取证步骤继续，标注缺口。
- **强耦合与保护路径**：写清原子组的成员、调用契约及启停和回退方式。对防 OOM、保数值、保导出或 checkpoint 一致性的路径单标「正确性保护（非提速）」。

### 实施

按用户授权实施。通用 PyTorch 项目复用现有配置入口，提供最小改动、可关闭开关、命中检查和明确回退。TurboPhysAI 接入遵循其 `OptimizationGroup` / recipe。调用/返回值、状态初始化、checkpoint 包装存在依赖时成组启用、验收及回退。

减 AC 或放宽 SAC 前先确认最坏形状的基线显存余量。实施 GC 冻结时验证各 rank 到达同一安全点及长跑 RSS。依次给出四层结论 **开关已开启 ≠ 优化路径实际命中 ≠ 数值与功能等价 ≠ 端到端稳态提速**。显存、OOM 与 RSS 作为独立验收红线。缺目标 GPU 或依赖时，做获准的静态接入和可执行的 CPU/mock 检查，设备数值、峰值显存、OOM 边界及吞吐标为未验证，RSS 仅报告已跑场景。

1. **改动与开关**：列出文件、默认状态、启停方式及原子组。
2. **已跑检查与结果**：包括路径命中、数值、功能、显存或 RSS、关闭开关的回退验证及无 Profiler 稳态性能。说明预热和样本。
3. **未验证项与限制**：注明输入上界、后端版本、依赖及叠加优化的显存余量。
4. **按生命周期回退**：局部无状态分支可直接切回。涉及分配器环境和已冻结 GC 时用新进程，涉及已安装 checkpoint 包装、被替换的前向或编译缓存时恢复原入口并按影响重建对象或重启进程。

## 工作流

1. 定位调用链、AC/SAC 状态和 PyTorch/后端版本，收集已有峰值显存、分配重试、步时分布及 rank 数据。
2. 用 [静态扫描器](scripts/scan_memory_candidates.py) 定位 AC、KV cache、GC 和清缓存候选。用可解析目标源码的 Python 获得 AST 命中；正则回退会向 `stderr` 告警，且正则回退及 YAML 只覆盖有限字面写法。`--json` 输出包含 `disclaimer`、`files_scanned`、`files_skipped`、`skipped_files` 与 `findings`，部分跳过或无文件可扫返回非零。候选语境和调用频率由实际调用链与运行证据确认。
3. 读 [模式](references/patterns.md) 选 M1–M7，按需查 [TurboPhysAI 证据](references/turbophysai-memory-evidence.md) 核对 Before/After、启用条件及受限变体。
4. 依 [验证与回退](references/validation.md) 执行授权范围内的路径、正确性、显存或 RSS、回退和稳态性能检查。用 [扫描器自测](scripts/selftest.py) 验证 CLI、命中和误报边界。
