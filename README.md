# PyTorch 性能优化技能库

这套 Skill 用于分析 PyTorch 训练和推理的性能瓶颈，并在目标项目中实施和验证优化。仓库包含 1 个上游 Trace 诊断 Skill、7 个下游专项优化 Skill，以及面向 `torch.compile` 的分析与实施提示词。

## Skill 目录

| Skill | 优化方向 | 适用信号 | 核心能力与验收 |
| --- | --- | --- | --- |
| [trace-bottleneck-report](trace-bottleneck-report/SKILL.md) | 瓶颈诊断与分流 | 已有 Profiler Trace，需定位时间去向（上游唯一诊断入口） | 解析多 rank Trace（可选 HTA 交叉校验），产出报告与 JSON，按可回收上限排序并分流。 |
| [torch-compile-integration](torch-compile-integration/SKILL.md) | 编译优化 | Launch/elementwise 占比高、图中断/重编译，或需评估编译边界 | 按 P1–P7 选边界、治重编译与全局 Inductor 调优；带开关接入并做 A/B/C 三组归因验收。 |
| [optimize-sync-elimination](optimize-sync-elimination/SKILL.md) | 同步消除 | `.item()`、`.cpu()`、动态分支或逐步日志引发 GPU 空闲 | 结合 Trace/静态扫描消除步内阻塞同步与读回，验收同步事件、GPU 空闲与稳态步时。 |
| [optimize-channels-last](optimize-channels-last/SKILL.md) | 内存布局优化 | 热点集中在 2D/3D 卷积，需评估 `channels_last` 布局 | 划定 2D/3D 卷积布局转换边界与 ROCm 布局配置，验收数值、布局转换开销、显存与吞吐。 |
| [optimize-kernel-replacement](optimize-kernel-replacement/SKILL.md) | 算子替换与融合 | 注意力、优化器、归一化或卷积等单算子已确认为热点 | 评估融合优化器、FA2/FlexAttention、Muon 或厂商 kernel，验收命中、数值、回退与性能。 |
| [optimize-distributed-comm](optimize-distributed-comm/SKILL.md) | 通信优化 | 暴露通信高、重叠率低，或需排查并行配置与 RCCL 顺序 | 先排慢 rank，再分级实施公开配置、集合重构与 ZeRO-2 改写，验收重叠、多卡等价与防死锁。 |
| [optimize-memory-and-recompute](optimize-memory-and-recompute/SKILL.md) | 显存与重算优化 | 显存富余想减重算、面临 OOM/碎片，或周期性 GC 步时尖刺 | 双向调优 AC/SAC、关隐式 KV cache、对齐 `gc.freeze()` 与分配器，验收峰值显存、RSS 与长尾。 |
| [optimize-data-pipeline](optimize-data-pipeline/SKILL.md) | 数据管线与异步 H2D | 步首数据等待、阻塞 H2D 暴露，或 CPU 解码/采样供给不足 | 调优 DataLoader、pin/异步 H2D 预取流、解码采样与受限特征缓存，验收样本/续训顺序与吞吐。 |

## 使用流程

1. **加载 Skill**：按目标客户端方式加载对应的 Skill 目录，并保留随包的 `references/` 与 `scripts/`。
2. **定位瓶颈**：已有 Trace 时，先用 `trace-bottleneck-report` 生成统计报告；需采集 Trace 时参考[采集清单](trace-bottleneck-report/references/collection.md)。热点已明确时，可直接使用对应专项 Skill。
3. **分析或实施**：只做分析时交付证据与候选方案；进入实施后保留可关闭开关与按生命周期回退路径，先验数值与路径命中，再用无 Profiler 的同配置基线比较稳态性能。

### 离线生成 Trace 报告

从仓库根目录运行（需支持 `from __future__ import annotations` 的 Python，如 Python 3.10+）。脚本仅依赖标准库，输出目录须为新路径：

```bash
python3 trace-bottleneck-report/scripts/trace_report.py --trace-dir /path/to/traces -o ./trace-report
```

输出为 `trace-report/report.md` 和 `trace-report/summary.json`，可先查阅[示例报告](trace-bottleneck-report/references/sample-report.md)。

### `torch.compile` 分析与实施提示词

1. [训练瓶颈分析提示词](prompts/training_trace_compile_analysis_prompt.md)：用于离线分析，结合 Trace、源码与已有 HTA 数据生成瓶颈报告与编译建议。
2. [Compile 优化实施与验证提示词](prompts/training_compile_implementation_prompt.md)：用于接入与验证（默认 `max-autotune-no-cudagraphs`，使用前须核实目标 PyTorch 与后端支持）。

提示词优先使用本地 `trace-bottleneck-report` 与 `torch-compile-integration`；可选外部诊断 Skill（如 `debug-graph-breaks`、`analyze-memory-snapshot`）仅在本地已安装时协同，缺失则按包内规则降级并在报告中注明。

## 仓库结构

- 根目录下的 8 个 Skill 均以 `SKILL.md` 为入口，配套 `references/`（模式、单份跨模型证据与验证规范）及纯标准库 `scripts/`（离线 Trace 统计或静态候选扫描 + `selftest.py` 自测）。
- [SKILL_DESIGN_PRINCIPLES.md](SKILL_DESIGN_PRINCIPLES.md)：定义专用性能优化 Skill 的设计规范、双模式契约、语料证据要求与发布自查清单。
- [prompts/](prompts/)：存放跨 Skill 的端到端分析与实施提示词（唯一维护源）。
- [.skills-manager/](.skills-manager/)：存放 Skills Manager 的备份元数据。README 中的 Skill 目录以仓库内实际存在的 `SKILL.md` 为准。
