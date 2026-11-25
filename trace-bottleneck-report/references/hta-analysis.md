# HTA 交叉校验操作与口径防错指南（可选）

HTA（Holistic Trace Analysis）是分析已有 PyTorch Profiler/Kineto Trace 的 Python 库。在本 Skill 中，**优先运行内置零依赖的 `scripts/trace_report.py`**；当用户已提供 HTA CSV，或目标环境已安装 HTA 且需要对多卡时间分解/空闲/通信重叠做交叉核对时，按本文操作。本 Skill **绝不自动 `pip install`/升级 HTA 或修改用户环境**。

## 1. 选择已有环境与输入核对

- 指定了 HTA Python 时优先检查该解释器；不可用时说明原因，不默默切换环境。未指定时仅检查当前 Python 及环境管理器已列出的相关环境，不递归扫描整台设备。
- 分别核对 `sys.executable`、Python 与 `HolisticTraceAnalysis` 版本，以及 `from hta.trace_analysis import TraceAnalysis` 能否导入（包存在不等于导入成功）。官方要求 Linux/macOS 及 Python >= 3.10；离线读取已有 Trace 不需要 GPU，也不启动项目训练/推理。
- 核对 Trace 文件与 rank 映射、同一次运行、配置/源码对应、step 数量及预热/首次编译/带栈开销影响；不覆盖原始数据，未经允许不写 Parquet 持久缓存。

## 2. 可选调用方式与返回形式

仅在环境已安装 HTA 且用户需要补跑交叉校验时使用；每项 API 必须**分别捕获错误和警告**，一项失败不跳过其余独立项：

```python
from hta.trace_analysis import TraceAnalysis

analysis = TraceAnalysis(trace_files=trace_files, trace_dir=trace_dir)
temporal = analysis.get_temporal_breakdown(visualize=False)
kernel_types, kernels = analysis.get_gpu_kernel_breakdown(visualize=False)
idle, idle_stats = analysis.get_idle_time_breakdown(
    ranks=selected_ranks, visualize=False, visualize_pctg=False
)
# 仅在多 rank 且通信事件可识别时调用；无通信时标不适用
overlap = analysis.get_comm_comp_overlap(visualize=False)
```

| API | 返回形式与用途 | 必查覆盖项 |
| --- | --- | --- |
| `get_temporal_breakdown` | DataFrame；各 rank 计算/非计算/空闲汇总 | 按实际设备活动窗口核对分母；不直接等同 CPU step 时长 |
| `get_gpu_kernel_breakdown` | 两个 DataFrame；类型占比与 kernel 统计 | 记录 Top-K/筛选参数；累计时长不是设备并集 |
| `get_idle_time_breakdown` | 元组（首项分解表，次项可为 `None`） | 显式传 `selected_ranks`（默认可能只选 rank 0）；记录 stream 与启发式条件 |
| `get_comm_comp_overlap` | DataFrame；通信/计算重叠 | 记录实际输出 ranks、通信识别条件和分母；无通信时不得把 `NaN` 当 `0%` |

注意：`TraceAnalysis` 构造参数可能默认排除最后一个 `ProfilerStep`，需核对实际加载的文件、ranks 和 steps，不能只看目录文件数。

## 3. 每项单独记录状态与口径防错

- **逐项标状态**：对解析及每个 API 分别记“成功 / 部分成功 / 失败 / 跳过”，附错误、警告与实际覆盖；空表、`None`、`NaN`、缺 rank 需解释，不得补零。
- **不硬编码单位**：历史核对版本（`c023a08a`）中 `temporal` 返回列含 `(us)` 而文档出现 `ns` 描述；换算前必须先检查实际列名与原始 Trace 元信息。
- **不与内置报告混算口径**：
  - HTA `temporal` 空闲、每 stream `idle` 与本 Skill `trace_report.py` 的 T2 空闲归因规则不同，**不可互换或做差**；
  - `non_compute_time` 不等于 idle；
  - `overlap` 以通信区间并集为分母，不是整个 step；
  - HTA 与原始 Trace 属同一批采集数据，仅作交叉核对，不构成独立测速验证；关键矛盾回原始时间线抽查。

## 4. 失败降级

HTA 不可导入、Trace 不兼容或部分 API 失败时，完整保留 `trace_report.py` 的统计报告，逐项注明 HTA 未运行或部分失败的原因；**绝不将其他工具结果称为 HTA 结果**，也不为此安装依赖或启动模型采集。

参考（可选联网核对）：[HTA 仓库](https://github.com/facebookresearch/HolisticTraceAnalysis)、[TraceAnalysis API](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html)（口径核对基线 `c023a08a24ed44b4dd9f691d0e3d04464f1e76e9`）。
