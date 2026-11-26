---
name: opt-distributed
description: Analyze or implement PyTorch multi-GPU collective overlap, sharding, and RCCL ordering. Use for exposed collectives, wrapper/config review, or RCCL ordering failures. Triage slow ranks first. Route batch/H2D stalls to opt-dataloader, step-internal sync to opt-sync, and graph capture to opt-compile.
---

# Optimize Distributed Comm

降低多卡训练关键路径上的集合通信暴露与冗余流量。核对实际并行架构、显存与多卡正确性。使用信号包括 T5 暴露通信高、rank 到达齐整但重叠不足、并行配置评审，以及 HIP/RCCL 集合提交乱序或 hang。

## 1. 定位与交接

- 由当前主瓶颈决定主导 Skill。先用 `trace-triage` 的 §1、§3 和 §6 T5 分流慢 rank，区分集合等待和计算重叠不足。已有 Trace 时用该只读上游做统计，通信方案在本 Skill 选型和验收。
- batch、DataLoader 或 H2D 等待交 `opt-dataloader`。OpenWAM 冻结编码器的输入准备与调度也由数据管线主导。涉及改写 ZeRO-2 收集或其与编码的重叠时，通信侧主导 gather 的依赖、发射点和进程组顺序，数据管线协同核对 batch 顺序。
- GC 周期尖刺、热路径分配器抖动或需要腾显存交 `opt-memory`。通信侧在拟用策略下实测缓冲和权重常驻峰值。
- 步内 `.item()`、`.tolist()` 和多余微步 `dist.barrier()` 交 `opt-sync`。集合序列或权重一致性所必需的同步由通信侧核对。
- 图捕获、编译范围与 DDP bucket 图切分交 `opt-compile`，通信侧负责包装生效、反向重叠与异步工作边界。单卡 kernel 和布局分别交 `opt-kernel` 与 `opt-layout`，通信侧负责交换协议和参数分片视图。

## 2. 按当轮授权决定工作模式

### 分析模式

用户只问方法、候选、原因或未授权修改时，做只读分析。读现有 Trace、日志、源码和版本信息可直接执行。修改项目或安装依赖、启动训练、补采 Trace 分别以当轮用户授权为限。

分析交付包括位置与符号、改法、证据级别（`[实测]` / `[静态]` / `[推断]`）、前提和风险、最小验证项及缺失证据。多候选默认用 4 列主表，再对前 1–2 项补充前提与风险、最小验证和缺口。单点咨询可直接回答。

| 优先级 | 位置与符号 | 建议改法 | 证据级别 |
| --- | --- | --- | --- |
| P0 / P1 / 暂缓 | 文件、行号和符号或配置键 | 模式及改动方向 | 实测 / 静态 / 推断 |

缺多 rank Trace、目标版本或显存数据时允许暂缓。写清缺失证据、待验证假设、最小取证办法与对应决策分支。例如先查实际并行配置和已安装包，或在授权后采短步数的多 rank Trace，再对照 T5。

把 Muon 收发与调用点、ZeRO-2 依赖组等作为原子改动。R1–R3、编译旁路的提前排空等单标「正确性保护（非提速）」。

### 实施模式

用户授权实施时，默认按通用 PyTorch 项目的入口加可关闭开关。TurboPhysAI 接入用其 `OptimizationGroup` / recipe。先验证相关且适用的 C1–C5 公开 API、配置或业务集合路径，并保留必要的 R1–R3 正确性保护。

C6–C8 涉及私有框架方法。只有相关一级路径已验证、多 rank Trace 仍显示可回收的暴露、且用户二次明确授权才实施。检查目标框架构建、关键方法签名与源码 AST hash。全 rank 一致后才启用，版本、选项或布局不符时明确拒装或成组回退。在途工作已经发射时走专门的排空/失败路径。

分别报告 **开关开启 ≠ 路径命中 ≠ 数值与功能等价 ≠ 端到端稳态提速**。缺目标多卡或依赖时，只做授权范围的静态接入和可跑的 CPU/mock 检查，多卡数值、死锁稳定性、峰值显存和性能标为未验证。

实施交付按以下四项组织。

1. 改动与开关，含默认值、受影响的文件和版本前提。
2. 已跑检查与结果，含命中、数值、checkpoint、峰值、T5 和稳态测速。
3. 未验证项与适用边界。
4. 按生命周期回退，关闭开关或原子组。已安装包装、进程组、缓冲池、优化器分片或编译图须按需清理、重启进程并重建对象，续训使用兼容 checkpoint。

## 3. 主干流程与参考

1. [分流与 RCCL 防护](references/triage-and-rccl.md) 用于诊断慢 rank、解释 T5 及核对 R1–R3。
2. 盘点 PyTorch、NCCL/RCCL、并行框架版本、卡数与拓扑。无 Trace 或调用点未知时，用 [静态扫描器](scripts/scan_comm_candidates.py) 初筛。优先用可解析目标源码的 Python 版本。已识别导入和常量配置仍只作为候选，动态选项与超出子集的 YAML 查实际配置。
3. [模式 C1–C8](references/patterns.md) 用于选型、核对前提和回退。[TurboPhysAI 证据](references/turbophysai-comm-evidence.md) 给出静态对照和受限案例。
4. [多卡验收规范](references/validation.md) 用于数值、显存、通信命中、checkpoint 与性能验收。
5. [扫描器自测](scripts/selftest.py) 用于验证候选命中、误报边界、CLI 和静态免责。
