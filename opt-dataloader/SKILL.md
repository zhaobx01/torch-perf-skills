---
name: opt-dataloader
description: >-
  Analyze or optimize PyTorch training data supply when batch loading, CPU
  preprocessing, or host-to-device copies delay a step. Use code or traces
  to locate DataLoader, decoding, sampling, or batch-transfer bottlenecks.
  Route step-internal D2H synchronization to opt-sync,
  collective overlap to opt-distributed, and compile shape
  contracts to opt-compile.
metadata:
  version: "1.0.0"
---

# Optimize Data Pipeline

## 1. 定位与分流

处理从数据集读取、解码、采样和 collate 到 batch 进入计算流的供给瓶颈。触发信号包括 `next(loader)` 等待、步首 GPU 空档、暴露的大块 H2D，以及源码中可疑的 worker、整行解码或拷贝配置。源码信号用于定位候选。

- **同步消除**处理 forward 或训练步内由 `.item()`、`.cpu()`、`.tolist()`、布尔索引等触发的 D2H。批量 H2D 和下一批等待由本 Skill 主导。
- **分布式通信**处理集合通信重叠。各 rank 到达集合通信的时间不齐时，先定位慢 rank 的数据等待、GC 或负载偏斜，再决定主导方。OpenWAM ZeRO-2 前瞻编码由通信专项主导，本 Skill 协同验收数据顺序和预取生命周期。
- **编译集成**处理为稳定图形状而引入的 padding、分桶及其 mask 契约。**显存专项**处理激活重算、GC 抖动和显存回收，本 Skill 验收多 worker、锁页和预取产生的资源增量。

## 2. 按本次用户授权工作

**只读分析**。仅询问方法或优化点、未授权修改时，读取源码、已有日志和 Trace，查询本地版本，或运行包内静态扫描。启动训练、补采 Trace 或安装依赖需要相应明确授权。

| 优先级 | 位置与符号 | 建议改法 | 证据级别 |
| --- | --- | --- | --- |
| P0 / P1 / 暂缓 | 文件、行号、函数或配置项 | 一句话说明改法和模式 | `[实测]` / `[静态]` / `[推断]` |

多候选评审只对优先的 1–2 项展开关键前提与风险、最小验证、缺失证据。单点可直接回答，无可靠候选时交付排查范围与缺口。

**缺证据时允许暂缓**。先核对 loader 的真实调用链、框架版本、已有运行记录和各 rank 时间线。若仍需区分读批等待、H2D 暴露与步内同步，给出最小取证范围及决策分支，例如经授权短采若干预热后稳态步骤，关联 `next(loader)`、CPU `copy_` 与设备流。证实供给暴露后再选模式，证实其他瓶颈则交接，信号不足则保留静态候选。

**实施与验收**。按授权在通用 PyTorch 项目作最小侵入修改，TurboPhysAI 项目沿用 `OptimizationGroup` / recipe。提供默认安全的开关。对强耦合或互斥路径写明依赖，成组评估、启停与回退。为避免死锁、样本错序或内存复用错误而保留的慢路径标为「正确性保护（非提速）」。

始终拆分 **配置已开启 ≠ 优化路径实际命中 ≠ 功能与数值等价 ≠ 端到端稳态提速**，并交付四项结果。

1. **实际改动与启停**。文件、默认状态及操作方式。
2. **已执行的检查与结果**。命中、样本顺序与数值、断点恢复、边界和性能，独立归因编译、通信和其他改动。
3. **限制与未验证项**。缺 GPU 或真实数据时完成可运行的 CPU/mock 检查，分别列出设备数值、多流安全、H2D、真实负载下的内存峰值和稳态收益的证据状态。
4. **按生命周期回退**。局部无状态逻辑可关闭后恢复。涉及预取先显式结束迭代器、join 线程并完成拷贝，改变 loader 构造参数时重建 loader，涉及进程级环境或全局包装时重启进程。缓存切换时重建 reader 并恢复对应编码路径。

## 3. 诊断与实施

1. 有 Trace 时先读 `trace-triage` §3 空闲归因和 §5 搬运表。没有 Trace 时执行 [静态扫描器](scripts/scan_data_pipeline_candidates.py)，确认候选在目标训练路径上。脚本可用 `--json`、`--min-severity` 和 `--no-configs`。
2. 先查 worker、固定内存与非阻塞配置（D1）、自定义 batch pinning（D2）、进程启动约束（D3）及解码和采样（D5）。仅在 H2D 仍暴露且源可安全固定时评估独立拷贝流（D4）。冻结编码反复成为供给瓶颈时，按缓存契约评估离线特征（D6）。
3. 依据 [验收与回退](references/validation.md) 核对真实命中、数据与恢复语义、内存和早退清理，再做无 Profiler 的稳态 A/B。

## 4. 按需读取

- 挑选候选或实现改法时读 [D1–D6 通用模式](references/patterns.md)。
- 设计验证、测速和回退时读 [验收与回退](references/validation.md)。
- 核实版本敏感实现、强耦合反例或历史来源时读 [TurboPhysAI 证据](references/turbophysai-data-evidence.md)。
- 变更扫描规则或排查脚本边界时运行 [扫描器自测](scripts/selftest.py)。
