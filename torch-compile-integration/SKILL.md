---
name: torch-compile-integration
description: >-
  Analyze or implement PyTorch torch.compile boundaries, graph-break and
  recompilation fixes, repeated-layer reuse, and global Inductor tuning with
  A/B/C attribution. Use for launch overhead, elementwise fusion, or compile
  capture issues. Route initial multi-rank Trace triage to
  trace-bottleneck-report, single-op replacement to optimize-kernel-replacement,
  conv layouts to optimize-channels-last, and host-device stalls to
  optimize-sync-elimination.
metadata:
  version: "2.2.0"
---

# Torch Compile Integration

## 定位与边界

- **目标领域**：`torch.compile` 编译边界选择、图捕获与重编译治理、重复层复用、全局 Inductor 编译调优与 A/B/C 三组对照验收。
- **核心作用**：以真实调用链与性能证据为依据，用最小可回退改动接入 `torch.compile`，保持算法语义、输入输出协议与数值精度；严格区分五层结论（`开关开启` ≠ `成功包装` ≠ `运行实际捕获` ≠ `数值等价` ≠ `端到端稳态提速`）。
- **何时使用**：上游 `trace-bottleneck-report` 报告显示 launch 小间隙或 elementwise 占比高、存在图断/重编译日志，或用户需要评估/接入/调试 `torch.compile` 边界、mode、dynamic、DDP/FSDP 组合与全局 Inductor 选项。
- **按主瓶颈分流与跨 Skill 交接**：
  - **vs `trace-bottleneck-report`**：拿到原始多 rank Profiler Trace、尚未明确主要瓶颈时，先由 `trace-bottleneck-report` 完成全量瓶颈统计与方向排行，再把 `launch 开销/融合/编译` 交接给本 Skill。
  - **vs `optimize-kernel-replacement`**：全局编译范围、图捕获、重编译治理及全局 Inductor Triton 启发式（如 `hip_triton_configs`）由本 Skill 主导；单算子实现替换、融合优化器、FlexAttention 参数、外部 kernel 注册 `custom_op`（K6）及单算子 `@torch.compiler.disable` 回退 ATen（K9）由算子专项主导。
  - **vs `optimize-channels-last`**：全局 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION` 归本 Skill；2D/3D 卷积子图的 `channels_last` 边界与 ROCm 布局开关归布局专项。
  - **vs `optimize-sync-elimination`**：步内 `.item()`/`.cpu()`/`nonzero` 导致的 GPU 饥饿与同步消除归同步专项；本 Skill 仅在标量提取直接阻断目标编译捕获时评估最小改写或局部 eager 边界。
  - **vs `optimize-data-pipeline`**：batch 加载、DataLoader H2D 与离线特征缓存归数据管线专项；在数据预取侧做 CPU 分桶以稳住 shape 时与数据管线协同核对字段契约。
  - **vs `optimize-memory-and-recompute`**：激活检查点（AC/SAC）覆盖范围与 OOM 显存预算归显存专项；AC/SAC 与编译区域交叉或共用入口时联合核对包装顺序与重编译变体并做消融。
  - **vs `optimize-distributed-comm`**：DDP/FSDP/ZeRO 集合通信重叠与 RCCL 顺序归通信专项；启用 `optimize_ddp=False` 减少 DDP bucket 切图时，由通信侧独立验收反向重叠是否退化。
- **自包含与离线边界**：全部核心规则、P1–P7 问题索引与跨模型证据均在包内，不依赖联网或外部 Skill。若本地已安装 `debug-graph-breaks` 或 `analyze-memory-snapshot`，可按需协同；未安装则直接按包内 P4 与显存排查要点降级，不自动联网下载外部 Skill 或安装依赖。

## 工作模式（按当次用户授权判定）

- **分析模式（只读）**：用户只询问方案、候选评审或未授权修改源码时进入。只读查本地源码、版本与已有日志/Trace 可直接执行；修改源码、启动训练/推理、补采 Trace 或安装依赖均须以本轮明确授权为限。
  - **必填交付要素**：涉及位置与限定符号、建议改法与分类（**A 直接试编译 / B 保持 eager / C 小重构后试编译 / D 证据不足待验证**）、证据级别（`[实测]` / `[静态]` / `[推断]`）、关键前提与风险、最小验证项、缺失证据。
  - **默认展示模板（多候选评审）**：使用 **4 列候选主表 + Top 1–2 三行短卡片**，低优项不展开；单点咨询或暂缓 compile 时灵活简化。

    | 优先级 / 分类 | 位置与符号 | 建议改法 | 证据级别 |
    | --- | --- | --- | --- |
    | P0 (A/C) / P1 / 暂缓 (B/D) | `file:line` + 限定类/函数名 | 一句话说明编译边界、初始配置与适用模式（P1–P7） | `[实测]` / `[静态]` / `[推断]` |

    仅对 Top 1–2 候选补充 3 行短卡片：**关键前提与风险**、**最小验证项**、**缺失证据/待核实项**。
  - **缺证据取证降级协议**：缺少 Trace、`TORCH_LOGS=graph_breaks,recompiles` 或输入分布数据时，**允许建议暂缓 compile**，不盲凑 P0。写清**缺什么证据**、**待验证假设**、**最低开销取证办法**（如先查调用链/已安装包，或经授权采短步 Trace / `TORCH_LOGS`）与**拿到证据后的决策分支**。
  - **强耦合与正确性保护标注**：存在强耦合契约的改动（如 GR00T 定长 padding + `CollatorWidthDropout`、MapTRv2 静态 GT 补长 + DDP `static_graph`、FSDP1 compile + `._orig_mod.` 清洗）必须**成组评估、启停、验收与回退**；为规避版本/后端缺陷启用的兼容路径（如 `MixedPrecision(buffer_dtype=None)`、`fallback_random=True`、`alignment_asserts=False`、只读 PIL 拷可写）单标为 **「正确性保护（非提速）」**，不混入性能增益。
- **实施模式（写代码 + 分层验收）**：本次请求已明确授权修改与验证时进入。
  - **目标代码形态**：默认面向通用 PyTorch 项目（接入项目配置/环境变量开关）；TurboPhysAI 任务顺从 `OptimizationGroup` / recipe 架构。
  - **工程底线**：保留可关闭开关与明确回退；从 `default` 模式建立基线，不盲选 `max-autotune`、`dynamic=True`、`fullgraph=True`，不默认开 `suppress_errors` 或静默回退；缺少目标 GPU/依赖时完成静态接入与 CPU/边界单测，并将设备数值、捕获、显存与性能诚实标为「未验证」。
  - **固定 4 项交付清单**：
    1. **实际改动与启停开关**（文件、默认开关值、启停方式及强耦合原子组）；
    2. **已跑检查与结果**（正确性、捕获/重编译、无 Profiler 稳态性能与峰值显存、A/B/C 归因）；
    3. **未验证项与限制**（环境缺口、版本前提与适用边界）；
    4. **按生命周期界定的回退方式**：纯调用期路由开关可直接关闭恢复；涉及进程级环境变量、已安装模块/类包装、已捕获编译缓存或 `torch._dynamo.config`/Inductor 钩子变更时，须重启进程或按影响范围重建模型。

## 主干流程

1. **核实真实执行路径与基线**：确认入口、模型类型与继承覆写、权重加载、train/eval、AMP、DDP/FSDP、已有 compile 包装及 PyTorch/后端/硬件版本。有 Trace 时优先消费 `trace-bottleneck-report` 报告；无 Trace 或初次摸排陌生仓库时，可先运行 [scripts/scan_compile_candidates.py](scripts/scan_compile_candidates.py) 盘点已有 compile/disable 入口、checkpoint+FSDP/DDP 包装线索、全局 config 修改与重复层分布（结果标注为 `[静态]` 候选，证据不足时按降级协议暂缓）。切勿把“注册/装饰器存在”当作“运行命中”。
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

- [references/compile-problem-patterns.md](references/compile-problem-patterns.md)：P1–P7 问题驱动模式（含证据状态、适用/不适用条件与最小验证）。
- [references/decisions.md](references/decisions.md)：边界比较、子系统粒度、mode/dynamic/fullgraph、DDP/FSDP、局部 AOT 与 Inductor 环境决策参考。
- [references/validation.md](references/validation.md)：五层生效核对、正确性、捕获/重编译、A/B/C 三组对照、全局 Inductor 验收与按生命周期回退规范。
- [references/turbophysai-compile-evidence.md](references/turbophysai-compile-evidence.md)：TurboPhysAI 16 模型统一编译证据与 `corpus/all-models` 分支链接索引（保留核查快照 SHA `796d3f6`，静态读源码所得）。
- [scripts/scan_compile_candidates.py](scripts/scan_compile_candidates.py)（附自测 [scripts/selftest.py](scripts/selftest.py)）：纯 Python 标准库静态编译现状与风险线索扫描器。JSON 输出包含 `disclaimer`、`files_scanned`、`files_skipped`、`skipped_files` 与 `findings`，部分跳过或无文件可扫返回 exit 1，无效参数返回 exit 2。
