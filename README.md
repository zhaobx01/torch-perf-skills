# PyTorch 性能优化技能库

用 PyTorch Profiler Trace 定位训练或推理瓶颈，再用专项 Skill 分析、实施和验证优化。仓库内置 **1 个 Trace 诊断 Skill 和 8 个专项优化 Skill**。每个 Skill 的 `SKILL.md` 都是独立入口。

## 快速上手

1. **加载所需 Skill。** 按客户端支持的方式加载完整 Skill 目录，保留 `SKILL.md`、`references/` 和 `scripts/`。
2. **按已有证据选择入口。** 已有 Trace 且瓶颈未明时，用 `trace-triage` 排查，优先跟进证据充分的 1–2 个方向。热点已明确时，直接选下表中的专项。需要准备 Trace 时，先看[采集清单](trace-triage/references/collection.md)。
3. **明确任务范围。** 分析时提供目标项目路径、已有采集或日志，以及已知的 PyTorch、后端和硬件信息。进入实施后，按项目容差验证正确性、实际路径命中与回退，再用相同配置下的无 Profiler 基线比较稳态性能。

## Skill 导航

| Skill | 优化方向 | 适用信号 | 核心能力与验收 |
| --- | --- | --- | --- |
| [trace-triage](trace-triage/SKILL.md) | 瓶颈诊断与分流 | 已有 Profiler Trace，需定位时间去向（上游唯一诊断入口） | 解析多 rank Trace（可选 HTA 交叉校验），产出报告与 JSON，按可回收上限排序并分流。 |
| [opt-runtime-env](opt-runtime-env/SKILL.md) | 运行时环境变量调优 | 只改启动配置/环境变量（不动源码），或需评估 ROCm/CUDA 后端开关 | 覆盖 5 大类后端环境变量选型、覆盖优先级与高危绑定排查，新进程单变量对照（冷启动/稳态/容差）与回退。 |
| [opt-compile](opt-compile/SKILL.md) | 编译优化 | Launch/elementwise 占比高、图中断/重编译，或需评估编译边界 | 按 P1–P7 选边界、治重编译与全局 Inductor 调优；带开关接入并做 A/B/C 三组归因验收。 |
| [opt-sync](opt-sync/SKILL.md) | 同步消除 | `.item()`、`.cpu()`、动态分支或逐步日志引发 GPU 空闲 | 结合 Trace/静态扫描消除步内阻塞同步与读回，验收同步事件、GPU 空闲与稳态步时。 |
| [opt-layout](opt-layout/SKILL.md) | 内存布局优化 | 热点集中在 2D/3D 卷积，需评估 `channels_last` 布局 | 划定 2D/3D 卷积布局转换边界与 ROCm 布局配置，验收数值、布局转换开销、显存与吞吐。 |
| [opt-kernel](opt-kernel/SKILL.md) | 算子替换与融合 | 注意力、优化器、归一化或卷积等单算子已确认为热点 | 评估融合优化器、FA2/FlexAttention、Muon 或厂商 kernel，验收命中、数值、回退与性能。 |
| [opt-distributed](opt-distributed/SKILL.md) | 通信优化 | 暴露通信高、重叠率低，或需排查并行配置与 RCCL 顺序 | 先排慢 rank，再分级实施公开配置、集合重构与 ZeRO-2 改写，验收重叠、多卡等价与防死锁。 |
| [opt-memory](opt-memory/SKILL.md) | 显存与重算优化 | 显存富余想减重算、面临 OOM/碎片，或周期性 GC 步时尖刺 | 双向调优 AC/SAC、关隐式 KV cache、对齐 `gc.freeze()` 与分配器，验收峰值显存、RSS 与长尾。 |
| [opt-dataloader](opt-dataloader/SKILL.md) | 数据管线与异步 H2D | 步首数据等待、阻塞 H2D 暴露，或 CPU 解码/采样供给不足 | 调优 DataLoader、pin/异步 H2D 预取流、解码采样与受限特征缓存，验收样本/续训顺序与吞吐。 |

## Skill 协作方式

以一次多卡训练排查为例：Trace 显示 GPU 存在空闲，且各 rank 进入 AllReduce 的时间不齐。

1. **先定位空闲来源。** 用 `trace-triage` 对齐各 rank 的训练步，确认最慢 rank 在集合通信前是在等 batch、做 H2D 拷贝，还是在步内被 `.item()` 等读回阻塞。
2. **由主瓶颈对应的 Skill 接手。** 步首供给不足交 `opt-dataloader`，步内读回阻塞交 `opt-sync`；若步时尖刺呈周期性且已有 GC 线索，交 `opt-memory` 核查；若各 rank 到达齐整后通信仍暴露在关键路径上，再用 `opt-distributed` 优化通信与计算重叠。
3. **每次只验证一项改动。** 例如开启 H2D 预取后，复核步首等待、最慢 rank 稳态步时与峰值显存；若仍有明显瓶颈，再用同口径 Trace 决定下一步。

当瓶颈落在计算侧时，仅通过启动环境变量做零侵入调优交 `opt-runtime-env`；涉及改源码的编译范围与全局 Inductor 配置交 `opt-compile`，卷积子图布局交 `opt-layout`，单算子替换或后端选择交 `opt-kernel`；涉及显存余量或重算权衡时由 `opt-memory` 把关。涉及整模型 AMP 迁移、精度 API 调整或通用数值回归验收时，按 [共享精度与数值验收指南](docs/precision-validation-guide.md) 执行。

如需补充底层诊断，可按需配合 `profile-model` 采集 Trace、`debug-graph-breaks` 深挖图中断，或 `analyze-memory-snapshot` 解析显存快照。未安装外部工具或处于离线环境时，包内 9 个 Skill 仍可独立完成分析（运行验证需目标框架与硬件）。完整交接条件见 [Trace 分流规则](trace-triage/references/handoff.md)。

## 离线生成 Trace 报告

在仓库根目录使用兼容的 Python（例如 Python 3.10+）运行。输入目录应包含同一次运行的各 rank Trace（`trace_rank_*.json` 或 `.json.gz`），输出目录须为新路径。

```bash
python3 trace-triage/scripts/trace_report.py --trace-dir /path/to/traces -o ./trace-report
```

脚本仅依赖 Python 标准库，生成 `trace-report/report.md` 和 `trace-report/summary.json`。优先看排行及其引用的统计表。排行中的可回收时间上限用于决定排查顺序，实际收益由无 Profiler 的对照测量确定。[示例报告](trace-triage/references/sample-report.md)

## 仓库与维护入口

- **Skill 目录**中的 `SKILL.md` 描述触发条件和工作流程，`references/` 提供模式、证据和验证细节，`scripts/` 包含离线统计或静态扫描及对应的 `selftest.py`。
- **制作与发布规则**见 [SKILL_DESIGN_PRINCIPLES.md](SKILL_DESIGN_PRINCIPLES.md)，包括职责划分、证据分级、版本管理和发布自查。
- **共享精度验收与按需扫描器**见 [docs/precision-validation-guide.md](docs/precision-validation-guide.md) 与 [docs/scripts/scan_precision_candidates.py](docs/scripts/scan_precision_candidates.py)，用于整模型 AMP 迁移、HIP BF16 探测修复及跨专项数值容差验收。
- **备份元数据**在 [.skills-manager/](.skills-manager/)。按目录加载时以本页的 9 个 Skill 入口为准，通过 Skills Manager 恢复时以其登记记录为准。
