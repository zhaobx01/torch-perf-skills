---
name: optimize-distributed-comm
description: Analyze or implement distributed communication overlap, sharding granularity, and collective reduction optimizations for PyTorch multi-GPU training (DDP, FSDP1/FSDP2, DeepSpeed ZeRO-2, and custom distributed collectives, plus HIP/RCCL collective ordering guards). Use when profiler traces show high exposed collective communication or low comm-compute overlap, or when reviewing distributed wrapper and ZeRO-2 configs. Always triage slow-rank skew before tuning overlap; hand off DataLoader/H2D stalls to data-pipeline optimization, step-internal D2H syncs to `optimize-sync-elimination`, and compile scope/graph breaks to `torch-compile-integration`.
---

# Optimize Distributed Comm

以目标项目的真实多卡 Trace 与并行架构为依据，排查并优化 PyTorch 分布式训练中的暴露通信与冗余集合通信，同时严守多卡数值等价与防死锁边界（含 HIP/RCCL 有序通信保护）。

## 1. 定位与边界

- **目标领域**：多卡分布式训练的通信重叠、分片粒度与集合通信效率优化，按风险分层覆盖：
  1. **先排慢 rank**（区分真通信暴露 vs 快卡等慢卡）；
  2. **公开 API / 配置级调优**（DDP、FSDP1/FSDP2、DeepSpeed ZeRO-2 标准配置）；
  3. **集合通信重构**（按 owner 分片计算 + `all_to_all` 替代全量 `all_gather`）；
  4. **深度框架内部改写**（DeepSpeed ZeRO-2 双缓冲事件调度、owner 分流归约与延迟 AllGather/Broadcast，属受限/强版本绑定参考）；
  5. **HIP/RCCL 通信有序性与死锁防护**（归类为「正确性保护（非提速）」）。
- **核心作用**：降低暴露通信时间、提高通信与计算重叠率、削减冗余通信字节数，避免盲目改写引发显存暴涨或分布式死锁。
- **何时使用**：
  - `trace-bottleneck-report` §6（T5 通信表）显示暴露通信占 step 比例高或重叠率低；
  - 源码或 DeepSpeed 配置中存在保守并行默认值（如 `find_unused_parameters=True`、`limit_all_gathers=True`、`reshard_after_forward=True`、`overlap_comm=false`）或全卡重复 `all_gather`；
  - 在 ROCm/HIP（RCCL）多卡环境下排查或预防多 stream 集合提交乱序死锁。
- **明确不做什么与多 Skill 分流规则**：
  - **慢 rank 根因优先**：若 Trace 显示暴露通信高，但 T5 起始极差大或最慢 rank 空闲差显著，**先排查慢 rank 根因**（数据加载、单卡 GC 抖动、负载不均或外部干扰），不要直接做通信重叠。
  - **步内 host-device 同步（`optimize-sync-elimination`）**：`.item()`、`.tolist()`、微步 `dist.barrier()` 等主机同步归同步消除主导（不要把消除多余 `barrier()` 与通信重叠混为一谈）。
  - **数据管线与 batch 搬运**：DataLoader 阻塞、pin_memory、batch 级异步 H2D 预取流归数据管线主导。
  - **图编译与图中断（`torch-compile-integration`）**：编译范围选择与图中断治理归编译专项主导；涉及 DDP/FSDP 与 `torch.compile` 包装顺序或 `optimize_ddp` 交互时协同确认。
  - **单卡算子/布局（`optimize-kernel-replacement`、`optimize-channels-last`）**：单卡计算热点与 NHWC/NDHWC 布局归对应专项；仅在布局转换影响 FSDP/DDP 参数视图或权重梯度布局时协同核对。

## 2. 工作模式与交付契约

按**本次用户授权**判定模式，不以调用次数机械限制：

### 2.1 只读分析模式（默认）

当用户仅询问方法、排查候选，或未明确授权修改源码/启动训练时，**不修改项目源码、不启动训练或采集**。

- **必填交付要素**：涉及位置与符号、建议改法（模式编号与层级）、证据级别（`[实测]` / `[静态]` / `[推断]`）、关键前提与风险（显存/死锁/版本依赖）、最小验证项、缺失证据。
- **默认版式（多候选评审）**：使用 **4 列主表 + Top 1–2 短卡片**——

  | 优先级 | 位置与符号 | 建议改法 | 证据级别 |
  | --- | --- | --- | --- |
  | P0 / P1 / 暂缓 | `file:line` + 类/函数或配置键 | 一句话说明改写方向与模式（如 `C2`） | `[实测]` / `[静态]` / `[推断]` |

  仅对最值得先动的 1–2 项补充 3 行短要点：**关键前提与风险**、**最小验证项**、**缺失证据/待核实项**；低优项不展开，不默认刷屏大段 diff。单点咨询可直接简洁作答；无可靠候选时只列排查范围与缺失证据。
- **缺证据取证降级协议**：缺少多 rank Trace（无法确认是真暴露通信还是慢 rank 偏斜）或缺少显存余量数据时，**允许暂缓优化**，不强凑 P0、不盲改并行配置；写清**缺什么证据**、**验证什么假设**、**最小开销获取方式**（优先用 `trace-bottleneck-report` 统计多 rank T5 表，或查当前并行框架版本/配置）及拿到后的决策分支。查本地版本、读已有日志/Trace 可直接执行；启动多卡训练或补采 Trace 须先获用户授权。
- **特殊标注**：存在联动契约的改动（如 `muon_gather` + `muon_owner_ns`、`zero2_owner_reduce` 依赖 `zero2_overlap`）必须**成组评估**；HIP/RCCL 有序通信、Gloo 元数据组与分支次数对齐单标为 **「正确性保护（非提速）」**，不混入提速候选。

### 2.2 实施模式（两级门禁 + 分层验收）

- **目标代码形态**：默认面向**通用 PyTorch 项目**（最小侵入修改并行构造与配置并接入可关闭开关）；若目标本身是 TurboPhysAI 接入任务，顺从其 `OptimizationGroup` / recipe 架构。
- **两级实施门禁**：
  1. **一级（公开 API / 配置级与自研集合重构，常规实施授权即可）**：先穷尽 `C1`–`C5`（DDP 参数、FSDP 预取/常驻/分片粒度、ZeRO-2 标准配置、业务层 `all_to_all` 重构）及必要的 RCCL 正确性保护（`R1`–`R3`）。
  2. **二级（高风险框架私有内部改写，必须二次明确授权）**：针对 `C6`–`C8`（侵入 DeepSpeed ZeRO-2 等框架私有方法的双缓冲/owner 归约/延迟 AllGather），**仅当一级调优已用尽、明确展示暴露通信残留证据，并获用户二次明确授权后才可实施**；且必须内置**严格版本/源码 AST hash 门禁**，版本不匹配时拒绝安装或显式回退，严禁静默错跑。
- **硬约束**：
  - **可关闭开关 + 明确回退**：强耦合改动成组启停与回退；不顺带混入 compile、同步消除或算子替换。
  - **四层结论严格拆分**：`配置/开关已开启` ≠ `通信重叠或流量削减真实命中` ≠ `多卡数值与无死锁等价` ≠ `端到端稳态提速`。
  - **无多卡 GPU / 缺依赖诚实交付**：无多卡环境时仅完成静态接入与单进程/mock 逻辑单测，明确标注多卡数值、死锁稳定性、峰值显存与性能为“未验证”。
- **固定 4 项交付清单**：
  1. **实际改动与启停开关**（文件、默认状态与启停方式，二级改写需注明锁定的框架版本）；
  2. **已跑检查与结果**（多卡数值/梯度对照、checkpoint 往返、峰值显存变化、T5 暴露通信与重叠率变化、无 Profiler 多轮稳态测速）；
  3. **未验证项与限制**（明确缺口、卡数/拓扑与前提边界）；
  4. **回退方式（按生命周期说明）**：并行包装（DDP/FSDP/ZeRO）、进程组（`ProcessGroup`）、通信缓冲与优化器分片状态均属进程级构造对象——**切换开关后必须重启训练进程并重新构建模型、并行包装与优化器**。

## 3. 主干流程

1. **先分流慢 rank，再确认真通信瓶颈**：已有 Trace 时，优先对照 `trace-bottleneck-report` 的 T5（暴露通信、重叠率、起始极差）与最慢 rank 空闲差，按 [references/triage-and-rccl.md](references/triage-and-rccl.md) 排除慢 rank 假象。
2. **盘点并行栈与静态初筛**：记录 PyTorch、NCCL/RCCL、DeepSpeed/Accelerate 版本与拓扑（单机/多机、卡数）；尚无 Trace 或面对大型陌生仓库时，运行 [scripts/scan_comm_candidates.py](scripts/scan_comm_candidates.py) 初筛 DDP/FSDP/DeepSpeed 配置与显式 `dist.*` 调用点（标为 `[静态]`）。
3. **按层级匹配模式与核对陷阱**：阅读 [references/patterns.md](references/patterns.md)（`C1`–`C8`）选择最低侵入且满足显存/动态图前提的方案；在 ROCm/HIP 环境下同步核对 [references/triage-and-rccl.md](references/triage-and-rccl.md) 中的 `R1`–`R3` 正确性保护；需要对照 16 模型真实实现时查阅 [references/turbophysai-comm-evidence.md](references/turbophysai-comm-evidence.md)。
4. **分层实施与多卡验收**：遵守两级门禁实施改造，按 [references/validation.md](references/validation.md) 验证多卡数值、梯度、checkpoint 往返、峰值显存、多步无死锁与 T5 暴露通信下降，再做无 Profiler 稳态测速。

## 4. 参考资料索引（按需加载）

| 文档 / 脚本 | 何时读取或运行 |
| --- | --- |
| [references/patterns.md](references/patterns.md) | 选型与设计改写方案，核对 `C1`–`C8` 的 Before 坏味道、前提、显存/版本风险与最小验证时 |
| [references/triage-and-rccl.md](references/triage-and-rccl.md) | 区分暴露通信与慢 rank 偏斜、解读 T5 指标，或在 ROCm/HIP（RCCL）下核对 `R1`–`R3` 防死锁保护时 |
| [references/validation.md](references/validation.md) | 实施改造、设计多卡数值/显存/checkpoint/无死锁验收与按生命周期回退时 |
| [references/turbophysai-comm-evidence.md](references/turbophysai-comm-evidence.md) | 对照 16 模型快照中已核实的跨模型实现、PR/基线 SHA 与陷阱细节时 |
| [scripts/scan_comm_candidates.py](scripts/scan_comm_candidates.py) | 陌生仓库或无 Trace 时初筛 DDP/FSDP/DeepSpeed 配置与显式 `dist.*` 调用点（纯标准库，仅产出静态候选） |
| [scripts/selftest.py](scripts/selftest.py) | 校验 `scan_comm_candidates.py` 的候选命中与误报边界 |
