# PyTorch 性能优化技能库

这套 Skill 用于分析 PyTorch 训练和推理的性能瓶颈，并在目标项目中实施和验证优化。仓库包括 1 个 Profiler Trace 分析 Skill、7 个专项优化 Skill，以及面向 `torch.compile` 的分析和实施提示词。

## Skill 目录

| Skill | 优化方向 | 什么时候用 | 输入与产出 |
| --- | --- | --- | --- |
| [trace-bottleneck-report](trace-bottleneck-report/SKILL.md) | 瓶颈诊断与分流 | 已有 PyTorch Profiler Trace，想知道 GPU 时间花在哪里（上游唯一诊断入口） | 读取同一次运行的各 rank Trace，可结合运行记录、编译日志、算子统计与 HTA（只读 CSV 或在已有 HTA 环境中可选交叉校验）。生成 Markdown 报告和 JSON 汇总，量化计算、空闲、同步、搬运和通信，按可回收上限排序并分流给下游专项 Skill。 |
| [torch-compile-integration](torch-compile-integration/SKILL.md) | 编译优化 | 上游指向 launch/elementwise/重编译瓶颈，或想判断哪些模型/子模块适合 `torch.compile` | 消费上游瓶颈证据并结合源码与 P1–P7 选择编译边界、治重编译/图中断及全局 Inductor 调优。实施时接入可回退开关，做 A/B/C 三组归因验收。 |
| [optimize-sync-elimination](optimize-sync-elimination/SKILL.md) | 同步消除 | GPU 空闲与 `.item()`、`.cpu()`、数据相关分支或逐步日志有关 | 结合 Trace 和源码定位隐式同步及设备到主机拷贝。可先用静态扫描找候选，实施后验证同步事件、GPU 空闲和 step 时间。 |
| [optimize-channels-last](optimize-channels-last/SKILL.md) | 内存布局优化 | 热点集中在 2D 或 3D 卷积，想评估内存布局优化 | 根据卷积路径划定 `channels_last` 或 `channels_last_3d` 的转换边界，验证数值、布局转换开销、端到端性能和显存。 |
| [optimize-kernel-replacement](optimize-kernel-replacement/SKILL.md) | 算子替换与融合 | 注意力、归一化、优化器或卷积中的算子已确认是热点 | 评估 fused AdamW、FlashAttention、FlexAttention、Muon 或厂商 kernel 等替换方案，核对后端和输入条件，验证实际命中、数值等价、回退与性能。 |
| [optimize-distributed-comm](optimize-distributed-comm/SKILL.md) | 通信优化 | 多卡 Trace 显示暴露通信占比高、通信与计算重叠率低，或需排查 DDP/FSDP/ZeRO-2 并行配置与 RCCL 顺序保护 | 先排查慢 rank 偏斜，再分级评估公开并行参数调优、集合通信重构及 ZeRO-2 深度改写。可先用静态扫描找候选，实施后验证暴露通信占比、多卡数值一致与防死锁边界。 |
| [optimize-memory-and-recompute](optimize-memory-and-recompute/SKILL.md) | 显存与重算优化 | 显存富余想减激活重算换吞吐、面临 OOM/分配器碎片，或多卡出现周期性 GC 步长尖刺 | 评估 AC/SAC 双向调优、关闭训练期隐式 KV cache、压缩中间激活/死分支、warmup 后对齐 `gc.freeze()` 及分配器调优。实施后验证峰值显存、梯度数值与步长长尾分布。 |
| [optimize-data-pipeline](optimize-data-pipeline/SKILL.md) | 数据管线与异步 H2D | Trace 显示步首数据等待空闲、阻塞式 H2D 拷贝暴露，或 CPU 数据解码/采样成为供给瓶颈 | 评估 DataLoader 并发/`pin_memory`/`persistent_workers`/`non_blocking`、自定义容器 pin、TF/RLDS `spawn` worker、独立 Copy-Stream 异步 H2D 预取、按需列解码与向量化采样，以及受限场景下的冻结编码器缓存。实施后验证样本与 resume 顺序等价、多流/主机内存防线及稳态吞吐。 |

## 使用流程

1. **加载 Skill**。按目标客户端的方式安装或加载对应的 Skill 目录，并保留随目录提供的参考资料和脚本。
2. **定位瓶颈**。已有 Trace 时，用 `trace-bottleneck-report` 生成统计报告。需要准备 Trace 时，参考[采集清单](trace-bottleneck-report/references/collection.md)。已有明确热点时，可直接使用对应的专项 Skill。
3. **选择并实施优化**。根据报告和源码选择专项 Skill。只做分析时，交付证据和候选方案。进入实施后，保留回退路径，先验证正确性，再用相同配置下的无 Profiler 基线比较稳态吞吐或 step 时间。Trace 报告中的可回收时间用于确定排查优先级，实际收益以对照测量为准。

### 离线生成 Trace 报告

从仓库根目录运行。请使用支持 `from __future__ import annotations` 的 Python（如 Python 3.11）。脚本只依赖标准库，输出目录须是新路径。

```bash
python3 trace-bottleneck-report/scripts/trace_report.py --trace-dir /path/to/traces -o ./trace-report
```

输出为 `trace-report/report.md` 和 `trace-report/summary.json`。可以先看[示例报告](trace-bottleneck-report/references/sample-report.md)。

### `torch.compile` 分析与实施提示词

1. [训练瓶颈分析提示词](prompts/training_trace_compile_analysis_prompt.md)用于离线分析。填写项目路径、Trace、分析环境和优化目标，结合源码及已有 HTA 数据生成分析报告和编译建议。
2. [Compile 优化实施与验证提示词](prompts/training_compile_implementation_prompt.md)用于接入和验证。填写分析报告、运行环境、设备预算和验收要求，再按确认的范围修改和测试。提示词默认使用 `max-autotune-no-cudagraphs`，实际使用前核实目标 PyTorch 版本和后端是否支持。

提示词涉及上游 Trace 瓶颈统计时优先使用本地 `trace-bottleneck-report`（含可选 HTA 交叉校验），编译建议与实施使用本地 `torch-compile-integration`；可选外部诊断 Skill（如 `debug-graph-breaks`、`analyze-memory-snapshot`）仅在本地已安装时协同，缺失则按包内规则降级并在报告中注明未执行的分析。

## 仓库结构

- 根目录下的八个 Skill 目录各自以 `SKILL.md` 为入口。随包的 `references/` 提供模式、案例与验证方法，部分 Skill 的 `scripts/` 提供离线统计、源码扫描及自测工具。
- [SKILL_DESIGN_PRINCIPLES.md](SKILL_DESIGN_PRINCIPLES.md) 定义专用性能优化 Skill 的设计、双模式契约、语料证据与发布自查规范。
- [prompts/](prompts/) 存放跨 Skill 的端到端分析与实施提示词，作为唯一维护源（不再在单个 Skill 内部保留副本）。
- [.skills-manager/](.skills-manager/) 存放 Skills Manager 的备份元数据。README 中的 Skill 目录以仓库里实际存在的 `SKILL.md` 为准。
