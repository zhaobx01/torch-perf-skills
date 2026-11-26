---
name: trace-triage
description: Upstream read-only bottleneck profiler for PyTorch training/inference. Parses multi-rank PyTorch Profiler traces plus optional run records, compile logs, key_averages, and HTA outputs (or optionally runs HTA in an existing Python environment for cross-checking) to quantify GPU time breakdown, idle attribution, sync starvation, H2D/D2H transfers, exposed communication, and a Top-5 recoverable upper-bound ranking, then hands off to downstream optimization skills. Use whenever you have Profiler traces and want to find where GPU time goes or which optimization direction to prioritize; when traces are missing, output the collection checklist only. Do not use to edit project source code, run training/profiling, or design concrete compile/kernel/layout/sync rewrites (hand off to opt-compile, opt-kernel, opt-layout, or opt-sync).
metadata:
  version: "1.0.0"
---

# Trace 瓶颈速览（上游只读诊断入口）

## 定位与边界

- **目标领域**：PyTorch Profiler Trace 离线统计、瓶颈量化与优化方向分流（全仓唯一上游瓶颈诊断入口）。
- **核心作用**：把已采集的多 rank Profiler Trace（及可选旁证数据）变成可复算的统计报告——每类 GPU 开销占多少、空闲从哪来、同步值不值得改、搬运和通信暴露了多少，并给出 Top-5 可回收上限排行。
- **何时使用**：拿到 Profiler Trace 后先查瓶颈在哪、决定先做哪类优化；没有 Trace 时只交付采集清单。
- **明确不做什么**：**纯只读分析，不修改项目源码、不启动训练/推理、不替用户做采集、不自动安装/升级 HTA 或运行依赖、不写具体代码实施方案**；定位出主要瓶颈方向后，按 [references/handoff.md](references/handoff.md) 交接给对应下游专项 Skill。

## 输入

| 数据 | 必需 | 说明 |
| --- | --- | --- |
| Profiler trace | 是 | `trace_rank_<rank>_*.json`（可 `.gz`），所有 rank，同一次运行，含设备 kernel 事件和 `ProfilerStep` |
| 运行记录 | 否 | 通用格式见 [collection.md](references/collection.md) §2（`command.json`、`step_metrics/rank-*.json`，由用户训练脚本写出）；给出 batch/卡数/环境开关与不受 profiler 影响的真实 step 时间 |
| 编译日志 | 否 | `TORCH_LOGS=graph_breaks,recompiles` 的输出，统计图断/重编译次数及原因（仅粗计数；深度诊断交 `opt-compile`） |
| key_averages 文本 | 否 | rank 0 的算子表（常在 `train.log`），只作 top 算子交叉校验 |
| HTA 输出 / 已有 HTA 环境 | 否 | 已有的 `temporal/idle/overlap` CSV，或在用户已有 HTA Python 环境时按 [references/hta-analysis.md](references/hta-analysis.md) 可选运行 HTA 做交叉验证（缺环境则跳过，不安装） |

缺 Trace 时不要编造数字：先读 [references/collection.md](references/collection.md)，把采集清单交给用户，等采集完再统计。

## 流程

1. **核对环境与数据**：分列记录**采集时环境**与**当前分析环境**（采集环境未知时不得用当前环境代替）；确认 Trace 来自同一次运行、rank 齐全、含 `ProfilerStep` 与设备事件；记录是否带调用栈（`with_stack`）及 active step 数（仅 1 个有效 step 时标明“单步样本”）。
2. **分级解析并运行内置脚本**（纯 Python 标准库，不读写项目源码）：
   - 多卡大 Trace（单 rank 约 1.3 GB 内存）按可用内存分级解析：**先查文件元数据 → 单 rank 或 `--jobs 1` 试跑 → 再按内存余量调大 `--jobs`**，避免盲目全 rank 并行触发 OOM。
   ```bash
   python3 scripts/trace_report.py --trace-dir <trace 目录> -o <新输出目录> \
       [--run-dir <运行目录>] [--compile-log <日志>] [--key-averages <文本>] [--hta-dir <目录>] \
       [--project-hint mypkg/] [--env-prefix MYPROJ_] [--jobs 4]
   ```
   输出目录必须不存在，脚本拒绝覆盖。`--trace-dir` 的上级目录若含 `command.json`，自动当作运行记录。
3. **可选 HTA 交叉校验**：若已有 HTA CSV 则直接传 `--hta-dir`；若未提供 CSV 且用户环境已装 HTA、需要多卡重叠/空闲交叉核对，按 [references/hta-analysis.md](references/hta-analysis.md) 在已有环境中只读运行并逐项记录 API 状态、警告与失败降级。
4. **阅读与交付**：先看第 0 节排行，再按“证据”列编号读 §1~§7（§1 总览、§2 GPU 耗时、§3 空闲归因、§4 同步、§5 搬运、§6 通信、§7 旁证）。向用户交付：**排行 + 关键证据表 + 口径与限制**，并用 [references/handoff.md](references/handoff.md) 指引下游专项 Skill，不越界写实施方案。

## 常用选项

| 选项 | 默认 | 作用 |
| --- | --- | --- |
| `--step N` | 每个 rank 的中间 step | 指定 `ProfilerStep` 编号 |
| `--project-hint a/,b/` | 空 | 用路径片段识别项目帧，栈里的 `file:line` 才落在项目代码；不给则按内置库前缀过滤 |
| `--rules rules.json` | 内置 | 补充 kernel 分类（排在内置规则之前）；未分类占比 >5% 时报告会提示 |
| `--small-gap-us` / `--big-copy-mib` / `--big-gap-ms` | 50 / 1 / 5 | launch 小间隙、大拷贝、大间隙阈值 |
| `--starve-pct` / `--mask-pct` | 50 / 20 | 同步判定“GPU 饥饿”/“被掩盖”的空闲/CPU 比例 |
| `--min-impact-pct` | 0.5 | 调用点 GPU 空闲占 step 低于此值判“影响小”，不报饥饿、不进排行 |
| `--slow-rank-ms` | 10 | 最慢 rank 空闲比 rank 0 多于此值时补一份该 rank 的空闲归因 |
| `--peak-h2d-gbs` / `--peak-comm-gbs` | 无 | 提供硬件峰值才显示“带宽占峰值%”，trace 里没有峰值，不猜 |
| `--env-prefix P1_,P2_` | 空 | 运行记录里额外保留的环境变量前缀（内置保留 HIP/NCCL/RCCL/OMP/TORCH/PYTORCH/CUDA/GPU） |
| `--warmup-steps N` | 10 | 计算真实 step 中位数时跳过的前 N 步 |
| `--jobs N` | 1 | 并行解析 rank（单个 rank 约 1.3 GB 内存） |

## 输出

- `report.md`：第 0 节排行 + §1~§7 各表；每个百分比列表头写明分母。
- `summary.json`：全部数字、阈值、口径（`schema_version=1`），便于脚本比较。

## 解读边界（必须遵守）

- **单次采集、单步样本**：每个 rank 只取一个 step，各 rank 不独立；不报显著性，不把上限当收益。
- **上限 ≠ 收益**：排行 ms 与条件量化上界（见 [metrics.md](references/metrics.md)）都是理论上限；各候选可能覆盖同一段空闲，不能相加。
- **区分 profiler 开销**：trace step 时间含 profiler 开销（带 `with_stack` 更明显）；正式测速必须用无 profiler 的运行记录或独立测速。
- **区分标签与口径**：`[实测]` 来自 trace 事件，`[推断]` 来自结构规则，`[上限]` 为上限估计；占累计、占计算、占 step、占忙时的分母不同，不可跨列相加；不以 CPU 墙钟减累计 kernel 时间当 GPU idle，不把 kernel 占比当 SM 利用率；**HTA idle 与本报告空闲窗口规则不同，不能互换**，HTA 与原始 trace 属同批数据，不构成独立实验验证。
- **同步不看耗时看代价**：`.item()` CPU 耗时大不等于有代价，以“调用期间 GPU 空闲”判定；设备同步期间 GPU 忙 ≥90% 说明在等真实工作，不属于同步消除。
- **栈缺失不猜代码行**：无栈时调用点只到外层算子链，明确写“未定位到代码行”。

## 自测

```bash
python3 scripts/selftest.py   # 合成 trace，断言各表数字、分类总和与百分比定义（需 Python 3.7+）
```

## 参考

- [references/collection.md](references/collection.md)：手动采集清单（只写清单，不执行）
- [references/metrics.md](references/metrics.md)：各表母体、分母、单位、条件量化上界与归因规则
- [references/hta-analysis.md](references/hta-analysis.md)：已有 HTA 环境下可选交叉校验的操作与口径防错指南
- [references/handoff.md](references/handoff.md)：排行方向类别与下游专项优化 Skill 的衔接分流
- [references/sample-report.md](references/sample-report.md)：用合成数据生成的示例报告
