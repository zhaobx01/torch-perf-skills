---
name: optimize-sync-elimination
description: Analyze or implement removal of implicit host-device synchronization and step-internal D2H stalls in PyTorch training/inference (scalar reads like `.item()`/`.tolist()`, data-dependent dynamic shapes like boolean indexing/`nonzero`/redundant `unique`, per-step logging/grad-accum sync, explicit `synchronize()`/`barrier()`). Use when traces show GPU idle starvation around host syncs or code scans flag hot-path scalar readbacks. Hand off DataLoader/batch H2D prefetch pipelines to `optimize-data-pipeline`, compile-region graph breaks to `torch-compile-integration`, and collective overlap/slow-rank issues to `optimize-distributed-comm`.
---

# Optimize Sync Elimination

以目标项目的真实调用链和 Trace 证据为依据，识别并消除训练/推理步内不必要的 host-device 同步（阻塞 D2H 读回、动态形状同步、多余的显式同步与同步日志），减少 GPU 队列排空造成的空闲。

## 1. 定位与边界

- **目标领域**：训练/推理步内的 host-device 同步消除（标量读回、CPU 元数据往返、数据相关动态形状与冗余去重、梯度累积期/每步日志同步、库或循环内多余的显式同步、以及步内小张量/延迟指标的非阻塞拷贝安全条件）。
- **核心作用**：消除导致 GPU 队列排空与空闲饥饿的阻塞点，提高设备利用率。
- **何时使用**：
  - Profiler Trace（如 `trace-bottleneck-report` 的 §4）显示同步调用期间出现明显 GPU 空闲饥饿；
  - 源码热路径存在 `.item()`、`.tolist()`、`.cpu()`、`tensor[bool_mask]`、`nonzero`/`unique`、微步内 `torch.cuda.synchronize()`/`dist.barrier()` 等可疑同步点；
  - 用户明确要求排查或消除 PyTorch 同步开销。
- **明确不做什么与多 Skill 分流规则**：
  - **数据管线（`optimize-data-pipeline`）**：等待下个 batch、DataLoader `pin_memory`/`persistent_workers`、独立 H2D 预取流归 `optimize-data-pipeline` 主导；本 Skill 仅保留步内小张量与延迟指标拷贝所需的非阻塞安全条件（见 `P8`）。
  - **图编译与图中断（`torch-compile-integration`）**：已经位于 `torch.compile` 捕获区域内、因 `.item()`/动态分支导致的 graph break 或重编译（如 `max_seqlen` 分桶稳 shape），由编译专项主导；本 Skill 聚焦 eager 路径与编译边界之外的同步。
  - **分布式通信重叠与慢 rank（`optimize-distributed-comm`）**：集合通信暴露、FSDP/ZeRO 通信重叠（如 `deferred_gather`、`ordered_all_reduce`）或 rank 到达不齐导致的等待，归 `optimize-distributed-comm` / 慢 rank 排查主导（不要把通信有序调度误当同步消除）。
  - **布局转换与算子替换（`optimize-channels-last`、`optimize-kernel-replacement`）**：设备内 NHWC/NDHWC 布局重排或融合 kernel 替换不属于本 Skill。

## 2. 工作模式与交付契约

按**本次用户授权**判定模式，不以调用次数机械限制：

### 2.1 只读分析模式（默认）

当用户仅询问方法、排查候选，或未明确授权修改源码/启动训练时，**不修改项目源码、不启动训练或采集**。

- **必填交付要素**：涉及位置与符号、建议改法（对应模式编号）、证据级别（`[实测]` / `[静态]` / `[推断]`）、关键前提与风险、最小验证项、缺失证据。
- **默认版式（多候选评审）**：使用 **4 列主表 + Top 1–2 短卡片**——

  | 优先级 | 位置与符号 | 建议改法 | 证据级别 |
  | --- | --- | --- | --- |
  | P0 / P1 / 暂缓 | `file:line` + 类/函数名 | 一句话说明改写方向与适用模式（如 `P5`） | `[实测]` / `[静态]` / `[推断]` |

  仅对最值得先动的 1–2 项补充 3 行短要点：**关键前提与风险**、**最小验证项**、**缺失证据/待核实项**；低优项不展开，不默认刷屏大段 diff。单点咨询可直接简洁作答；无可靠候选时只列排查范围与缺失证据。
- **缺证据取证降级协议**：缺少 Trace 无法证实热路径执行或 GPU 饥饿时，**允许暂缓优化**，不强凑 P0、不盲改源码；写清**缺什么证据**、**验证什么假设**、**最小开销获取方式**（如跑 `trace-bottleneck-report` 或短步数 Profiler）及拿到后的决策分支。查本地版本、读已有日志/Trace 可直接执行；启动训练或补采 Trace 须先获用户授权。
- **特殊标注**：存在联动契约的改动（如前向延迟 `.detach()` + 日志边界 `.item()`）必须**成组评估**并显式标出依赖；为防 NaN/错误或保证 checkpoint 一致性保留的同步单标为 **「正确性保护（非提速）」**，不混入提速候选。

### 2.2 实施模式（经明确授权后）

- **目标代码形态**：默认面向**通用 PyTorch 项目**（最小侵入修改并接入项目配置或环境变量开关）；若目标本身是 TurboPhysAI 接入任务，顺从其 `OptimizationGroup` / recipe 架构。
- **硬约束**：
  - **可关闭开关 + 明确回退**：不静默吞错；强耦合改动成组启停与回退；不顺带混入 compile、布局或算子替换。
  - **四层结论严格拆分**：`配置/开关存在` ≠ `静态命中或同步真实消除` ≠ `数值与功能等价` ≠ `端到端稳态提速`。
  - **无 GPU / 缺依赖诚实交付**：无目标设备时仅完成静态接入与可运行的 CPU/边界单测，明确标注设备数值与性能为“未验证”。
- **固定 4 项交付清单**：
  1. **实际改动与启停开关**（文件、默认状态与启停方式）；
  2. **已跑检查与结果**（数值对照、边界/回退测试、Trace 同步事件与 GPU 空闲变化、稳态性能与预热说明）；
  3. **未验证项与限制**（明确缺口与前提边界）；
  4. **回退方式（按生命周期说明）**：关闭开关恢复原路径；若涉及进程级配置/环境变量、已安装模块包装、已捕获编译图或跨步缓存状态，说明需重启进程或重建对象。

## 3. 主干流程

1. **定范围与核版本**：沿调用链排查标量读回（日志、梯度累积微步、loss 归一化、动态长度/mask、位置编码元数据、优化器/AMP 状态、显式同步）。记录 PyTorch、CUDA/ROCm-HIP、分布式框架版本（同步行为随版本变化，以目标版本文档或实测为准）。
2. **分层取证**：
   - **已有 Trace**：优先复用 `trace-bottleneck-report` 核对同步调用期间的 GPU 空闲饥饿；单份 Trace 速查或未知 HIP/DTK 事件名摸排时，阅读 [references/profiling.md](references/profiling.md) 并运行 [scripts/analyze_trace_sync.py](scripts/analyze_trace_sync.py)。
   - **暂无 Trace**：运行 [scripts/scan_sync_points.py](scripts/scan_sync_points.py) 生成静态候选，逐点核对对象是否已在 CPU 上（过滤常见误报），并将所有未实测点标为 `[静态]`。
3. **匹配模式与设计方案**：按需阅读 [references/patterns.md](references/patterns.md)（P1–P8），确认改法、前提、强耦合依赖与反例；需要对照真实跨模型实现时，查阅 [references/turbophysai-sync-evidence.md](references/turbophysai-sync-evidence.md)。
4. **实施与分层验收**：按 [references/validation.md](references/validation.md) 完成数值、边界、分布式/checkpoint、Trace 同步下降确认与多轮稳态测速；无收益或回归时按生命周期回退并如实报告。

## 4. 参考资料索引（按需加载）

| 文档 / 脚本 | 何时读取或运行 |
| --- | --- |
| [references/patterns.md](references/patterns.md) | 设计改写方案、核对 P1–P8 适用前提、反例与最小验证时 |
| [references/profiling.md](references/profiling.md) | 已有 Trace、需区分 GPU 真实饥饿、摸排 HIP/DTK 事件名或排查静态误报时 |
| [references/validation.md](references/validation.md) | 实施改造、设计数值/边界/分布式验收与回退策略时 |
| [references/turbophysai-sync-evidence.md](references/turbophysai-sync-evidence.md) | 需要对照 16 模型快照中已核实的跨模型实现与陷阱细节时 |
| [scripts/scan_sync_points.py](scripts/scan_sync_points.py) | 缺 Trace 时对目标源码做静态候选初筛（纯标准库，仅产出静态候选） |
| [scripts/analyze_trace_sync.py](scripts/analyze_trace_sync.py) | 单份 Chrome Trace JSON 快速按事件名/栈汇总同步事件或 `--list-names` 摸排 |
| [scripts/selftest.py](scripts/selftest.py) | 校验上述两个脚本的候选命中、误报边界与 Trace 统计口径 |
