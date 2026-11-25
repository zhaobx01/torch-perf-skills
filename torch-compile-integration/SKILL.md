---
name: torch-compile-integration
description: Analyze PyTorch source to choose torch.compile boundaries and insertion points, resolve recompilations/graph breaks/nested-layer reuse, tune global Inductor options (e.g. TORCHINDUCTOR_LAYOUT_OPTIMIZATION, HIP Triton heuristics), or implement and benchmark compilation with A/B/C attribution. Use when profiler bottlenecks point to launch overhead, elementwise/fusion opportunities, or compile-capture issues, or when adding/debugging torch.compile switches. Do not use for initial multi-rank Trace bottleneck discovery (use trace-bottleneck-report first), single-operator kernel/optimizer/FlexAttention replacement (use optimize-kernel-replacement), channels-last conv layout design (use optimize-channels-last), or host-device sync elimination (use optimize-sync-elimination).
metadata:
  version: "2.0.0"
---

# Torch Compile Integration

## 定位与边界

- **目标领域**：`torch.compile` 编译边界选择、图捕获与重编译治理、重复层复用、全局 Inductor 编译调优与三组对照验收。
- **核心作用**：以真实调用链与性能证据为依据，用最小可回退改动接入 `torch.compile`，保持算法语义、输入输出协议与数值精度；严格区分“开关开启 ≠ 成功包装 ≠ 实际捕获 ≠ 数值等价 ≠ 端到端稳态提速”。
- **何时使用**：
  - 上游 `trace-bottleneck-report` 报告显示 launch 小间隙或 elementwise 占比高、存在图断/重编译日志，或用户需要评估/接入/调试 `torch.compile` 边界、mode、dynamic、DDP/FSDP 组合与全局 Inductor 选项。
- **明确不做什么 & 多 Skill 主导/分流规则**：
  - **vs `trace-bottleneck-report`**：拿到原始多 rank Profiler Trace、尚未明确主要瓶颈时，**先由 `trace-bottleneck-report` 完成全量瓶颈统计与方向排行**，再把 `launch 开销/融合/编译` 方向交接给本 Skill，本 Skill 不重复承担全量 Trace/HTA 统计。
  - **vs `optimize-kernel-replacement`**：全局编译范围、图捕获、重编译治理及全局性 Inductor Triton 启发式（如 `hip_triton_configs`）由本 Skill 主导；单算子实现替换、融合优化器、FlexAttention 参数、外部 kernel 注册为 `custom_op`（K6）及单算子 `@torch.compiler.disable` 回退 ATen（K9）由 `optimize-kernel-replacement` 主导。
  - **vs `optimize-channels-last`**：全局 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION` 归本 Skill；2D/3D 卷积子图的 `channels_last` 边界与 ROCm 布局开关归 `optimize-channels-last`。
  - **vs `optimize-sync-elimination`**：训练循环内 `.item()`/`.cpu()`/`nonzero` 导致的 GPU 饥饿与同步消除归 `optimize-sync-elimination`；本 Skill 仅在标量提取直接阻断目标编译捕获时评估最小改写或局部 eager 边界。
- **自包含与离线边界**：全部核心规则、P1–P7 问题索引与历史案例均在包内，不依赖联网、原项目快照或外部 Skill。若本地已安装 `debug-graph-breaks`（图中断/`tlparse` 深挖）或 `analyze-memory-snapshot`（编译后 OOM 显存快照解析），可按需协同；未安装则直接按包内 P4 与显存排查要点降级，不自动联网下载外部 Skill 或安装 Python 依赖。

## 工作模式（按当次用户授权判定）

- **分析模式（只读）**：用户要求分析、审查、筛选候选或未明确授权修改源码与运行训练时进入。**不修改项目源码、不启动训练/推理、不额外采集**；可输出建议 diff（标注未应用）。
  - 必填要素：源码位置与限定符号、热点/调用链证据与证据级别（`[实测]` / `[静态]` / `[推断]`）、建议改法与分类（**A 直接试编译 / B 保持 eager / C 小重构后试编译 / D 证据不足待验证**）、关键前提与风险、最小验证项与缺失证据。
  - 多候选默认输出候选表，并只对最值得先动的 **Top 1–2 候选**展开前提、风险与验证短要点（单点咨询或证据不足暂缓 compile 时灵活简化）：

    | 优先级/分类 | 源码位置与符号 | 调用链/热点证据 | 编译边界、初始配置与风险 | 证据级别 |
    | --- | --- | --- | --- | --- |

- **实施模式（写代码 + 分层验收）**：本次请求已明确授权修改与验证时进入。
  - 默认面向通用 PyTorch 项目（接入项目配置/环境变量开关）；TurboPhysAI 任务顺从 `OptimizationGroup` / recipe 架构。
  - 从 `default` 模式建立基线，不盲选 `max-autotune`、`dynamic=True`、`fullgraph=True`，不默认开 `suppress_errors` 或静默回退。缺少目标 GPU/依赖时完成静态接入与 CPU/边界单测，并将设备数值与性能诚实标为“未验证”。
  - **固定 4 项交付清单**：① 实际改动与启停开关；② 已跑检查与结果（正确性、捕获/重编译、无 profiler 稳态性能与显存、A/B/C 归因）；③ 未验证项与限制；④ **按生命周期界定的回退方式**（纯调用期路由开关可直接关闭恢复；涉及进程级配置、已安装的模块/类包装、已捕获编译缓存或 `torch._dynamo.config` 变更时，须重启进程或重建模型）。

## 主干流程

1. **核实真实执行路径与基线**：确认入口、模型类型与继承覆写、权重加载、train/eval、AMP、DDP/FSDP、已有 compile 包装及 PyTorch/后端/硬件版本。有 Trace 时优先消费 `trace-bottleneck-report` 报告；无 Trace 时明确热点为 `[静态]` 假设，必要时允许建议“暂缓 compile”。切勿把“注册/装饰器存在”当作“运行命中”。
2. **按问题查阅 P1–P7 选择边界**：先明确当前核心问题，再按需查阅 [跨模型问题参考（P1–P7）](references/compile-problem-patterns.md) 与 [配置决策](references/decisions.md)：
   - **P1 编译范围**：比较整模型、训练步、重复子模块/子系统分区、局部 Tensor 函数，不机械按热点加装饰器；
   - **P2 复杂外层与状态契约**：保留外层调度/切块，抽最小 helper 并显式传出状态缓存（按需评估局部 AOT）；
   - **P3 重编译治理**：先看 `recompiles` guards，再权衡动态策略、参数分桶或定长 padding 契约（保真实有效长度与 dropout/mask 语义）；
   - **P4 图中断与局部 eager**：按“边界调整 / 保持语义的 Tensor 改写 / `@torch.compiler.disable` 局部 eager”三类择一最小处理，低影响项暂缓；外部 kernel 需进图时交 `optimize-kernel-replacement` K6；
   - **P5 低侵入接入与全局 Inductor 调优**：生命周期集中接入、明确真实关闭路径、`with torch._dynamo.config.patch(...)` 作用域隔离、全局 Inductor 环境变量与启发式候选集独立消融；
   - **P6 五层生效验证**：分层核对配置 → 匹配包装 → 运行捕获 → 正确性 → 稳态收益；
   - **P7 重复层复用与嵌套编译**：区分共享包装与共享图，核对闭包/缓存 key 与内外层分派。
3. **三组对照归因与分层验收**：按 [验证规范](references/validation.md) 执行验收。混合重构优先做 **A（原实现）/ B（重构后 eager）/ C（同重构 compiled）** 三组对照，禁止把计算重构、布局、同步、TF32 或通信优化的收益归因给 `torch.compile`。

## 参考索引（按需加载）

- **核心方法论**：[compile-problem-patterns.md](references/compile-problem-patterns.md)（P1–P7 问题路由）、[decisions.md](references/decisions.md)（mode/dynamic/fullgraph/DDP/FSDP/Inductor 配置）、[validation.md](references/validation.md)（正确性、捕获、三组对照与按生命周期回退）。
- **16 模型新语料总索引**：[turbophysai-compile-evidence.md](references/turbophysai-compile-evidence.md)（`zhaobx01/TurboPhysAI` 固定快照 `796d3f6` 永久链接索引与新模型机制要点）及 [turbophysai-compile-patterns.md](references/turbophysai-compile-patterns.md)（Dynamo-safe 改写与 FSDP1 包装顺序/`._orig_mod.` key 兼容）。
- **历史单模型深度案例（按需查阅，非每次必读）**：[maptrv2-case-study.md](references/maptrv2-case-study.md)、[maptrv2-turbophysai-compile-case-study.md](references/maptrv2-turbophysai-compile-case-study.md)、[openvla-compile-case-study.md](references/openvla-compile-case-study.md)、[cosmos3-compile-case-study.md](references/cosmos3-compile-case-study.md)、[bevformer-compile-case-study.md](references/bevformer-compile-case-study.md)、[bevfusion-compile-case-study.md](references/bevfusion-compile-case-study.md)、[openwam-compile-case-study.md](references/openwam-compile-case-study.md)。
- **版本与演进记录**：[sources.md](references/sources.md)。
