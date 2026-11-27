---
name: opt-runtime-env
description: >-
  Analyze or tune PyTorch backend environment variables in ROCm/HIP or CUDA
  launch configuration. Use to choose a startup setting, audit overrides,
  or compare runtime flags. Route initial Trace triage to trace-triage and
  compiler, layout, or collective code changes to opt-compile, opt-layout, or opt-distributed.
metadata:
  version: "1.1.1"
---

# 运行时环境变量调优

在保持模型、训练与推理实现一致的条件下，评估 PyTorch 后端启动配置。五类变量覆盖精度与数学库、卷积搜索与布局、硬件队列/PCIe/分配器、多卡通信与线程、编译与缓存。

## 定位与分流

- **触发信号**：用户评估某个启动变量、排查覆盖顺序、希望只通过启动配置试点，或已有热点与启动线索（例如 GEMM 占时高，运行记录显式关闭 TF32）。已有 Trace 且瓶颈未归因时先由 `trace-triage` 主导。
- **纯配置实验**：本 Skill 主导变量选型、隔离和回退。涉及 NCCL/RCCL 拓扑与 rank 顺序、布局传播或编译产物时，分别与 `opt-distributed`、`opt-layout`、`opt-compile` 联合验收对应契约。
- **改动代码**：编译边界、算子实现、布局传播、步内同步、显存/重算、通信调度和数据管线分别交给对应专项。整模型 AMP、PyTorch 精度 API 或 HIP BF16 能力探测修复交项目代码流程，数值验收可参照整仓共享精度指南。

## 按本轮授权选择工作模式

- **只读分析**：检查源码入口、现有配置、构建和已有运行记录，或运行静态扫描。单点给前提、最小对照、回退和证据缺口；多候选用「优先级 / 变量与位置 / 建议动作及模式 / 证据级别」四列，Top 1–2 补关键前提与风险、最小验证、缺失证据。标记 `[实测]`、`[静态]` 或 `[推断]`。缺 Trace、代表性形状或构建信息时可暂缓推荐，列出最小补证及决策分支。
- **实施**：用户本轮授权后修改启动脚本、命令前缀、RuntimeConfig、字面 preset 或纯启动器中的配置。按授权范围执行对照；启动训练、推理或补采依本轮用户授权。缺目标 GPU 时做静态接入与可运行的 CPU/mock 边界检查，路径、精度与性能标为未验证。

## 主干流程

1. **追踪覆盖链。** 核对 shell 继承值、启动器、配方、命令行覆盖和目标程序加载的 preset 或条件默认值。TurboPhysAI 先处理 `environment.unset/set`，accuracy 模式再处理其 `unset/set`，最后叠加启动 CLI 的 `--set`，然后核对目标程序是否重写。
2. **挑选对照。** 用 [五类模式](references/patterns.md) 定位方案，用 [变量手册](references/env-catalog.md) 查取值和构建依赖，用 [语料证据](references/turbophysai-env-evidence.md) 核实受限案例。每次隔离一个变量；存在互斥、缓存或拓扑契约时，明示依赖并成组评估、启停与回退。已证实用于规避回归的设置单列为「正确性保护（非提速）」。
3. **以新进程验收。** 按 [分轨验收与回退](references/validation.md) 分开确认 **配置生效 ≠ 路径命中 ≠ 正确性与数值通过 ≠ 端到端稳态提速**。开关影响数学模式时用训练或推理容差验收。另记首轮搜索、编译或建环成本。
4. **交付四项。** 列出实际改动与基线/候选启停方式；已跑检查和结果；未验证项；生命周期回退。恢复基线值或未设置状态，配套恢复缓存命名空间，启动新进程核对路径和指标。

## 静态扫描

用 [scan_runtime_env.py](scripts/scan_runtime_env.py) 初筛 RuntimeConfig 的普通与 accuracy 层、字面 `.env`、简单 Shell 及 Python 环境操作，运行示例为 `python3 scripts/scan_runtime_env.py <路径> --json`。输出只说明可见配置线索；动态流程、跨文件和实际值由启动链核对。执行 [自测](scripts/selftest.py) 可检查 CLI 和回归样本。语料 After 配置与设备实测状态见 [证据索引](references/turbophysai-env-evidence.md)。
