---
name: optimize-kernel-replacement
description: >-
  Analyze or implement local PyTorch operator replacement and kernel tuning on CUDA or ROCm/HIP.
  Use for a hot operator, an unfused optimizer, or attention backend selection and fallback.
  Route compile scope and global Inductor tuning to torch-compile-integration,
  layout propagation to optimize-channels-last, and host-device stalls to optimize-sync-elimination.
---

# Optimize Kernel Replacement

## 定位与边界

- **目标领域**：单算子/热点内核实现替换与后端调优（CUDA 与 ROCm/HIP）。
- **核心作用**：针对已确认或高度疑似的算子计算热点，用更快或更稳的融合/厂商/定制实现替换原算子，消除冗余打包与低效展开，并守住数值精度、输入契约与可回退边界。
- **何时使用**：
  - **Trace 信号**：GPU 计算时间集中在注意力（SDPA/FlexAttention/FA2 varlen）、优化器 step（逐参数 `foreach`/AdamW/Muon）、归一化（LayerNorm/RMSNorm）、RoPE/逐元素调制、特定卷积/反传池化或退化的 Inductor gather 展开。
  - **代码特征信号**：未启用融合的 `AdamW`、冗余 `BlockMask` 或等长 FA2 打包、使用复数表示的 RoPE、MHA 调用请求了未消费的权重、缺少运行时守卫的外部算子。看到 MHA 构造时先追查调用参数和消费者。
  - **用户诉求**：评估或接入融合优化器、切换注意力后端、调优 FlexAttention 参数、替换厂商 kernel 或排查算子替换引发的数值/NaN/回退问题。
- **按主要瓶颈分流**：
  - 原始多 rank Trace 尚未定位瓶颈时，先由 `trace-bottleneck-report` 统计。收到热点算子、调用位置和输入形态后，用本 Skill 选型与验收。
  - **vs `torch-compile-integration`**：本 Skill 主导单算子实现、后端、tile 选择与局部绕开低效展开，负责局部契约与验收。编译范围、图捕获、重编译和全局 Inductor 钩子／启发式（如 `hip_triton_configs`）由 `torch-compile-integration` 主导。通过全局钩子安装单算子特化时，分别验证安装作用域和算子结果。
  - **vs `optimize-sync-elimination`**：`.item()`/`.cpu()`/`nonzero` 导致的 GPU 空闲交给 `optimize-sync-elimination`。
  - **vs `optimize-channels-last`**：2D/3D 卷积子图的 `channels_last`/`channels_last_3d` 布局边界设计交给 `optimize-channels-last`（本 Skill 仅校验目标融合算子所需的输入布局契约）。
  - **vs `optimize-distributed-comm`**：ZeRO/FSDP 的集合调度、重叠和 rank 顺序由通信侧主导，本 Skill 验证局部打包、优化器计算与舍入契约。批加载/H2D 交 `optimize-data-pipeline`，激活重计算交 `optimize-memory-and-recompute`。通用算子库研发在本 Skill 范围之外。

## 工作模式（按当次用户授权判定）

- **分析模式（只读）**：用户只询问优化点或可行性，或当轮未授权修改时，读取源码、现有 Trace 与运行记录，按需运行只读静态扫描。
  - **必填交付要素**：涉及位置与符号、建议改法（关联模式编号 K1–K10）、证据级别（`[实测]` / `[静态]` / `[推断]`）、关键前提与风险、最小验证项、缺失证据。
  - **默认展示模板（多候选评审）**：使用 **4 列候选主表 + Top 1–2 短卡片**，低优项不展开，不默认刷屏大段 diff；单点咨询或无可靠候选时灵活简化，不强行套表。

    | 优先级 | 位置与符号 | 建议改法 | 证据级别 |
    | --- | --- | --- | --- |
    | P0 / P1 / 暂缓 | `file:line` + 类/函数名 | 一句话说明改写方向与适用模式（如 K3/K5/K8） | `[实测]` / `[静态]` / `[推断]` |

    仅对 Top 1–2 候选补充 3 行短卡片：**关键前提与风险**、**最小验证项**、**缺失证据/待核实项**。
  - **缺决定性证据时可暂缓**：写明缺失证据、待检验假设、最低开销的取证方式和不同结果下的决策。无 Trace 时先核对调用链、现有记录、包版本和设备信息。若需新采集，可建议经授权采少量预热后的稳态步。目标算子确为热点再选型，证据仍不足则保留静态候选。只读探测可直接运行，训练、补采和安装依赖须有当轮明确授权。
  - 在候选表和实施交付中标注**联动依赖、互斥项与正确性保护**。同一契约的生产端和消费端成组评估、启停、验收与回退。为规避 NaN 或后端故障启用的慢路径标作「正确性保护（非提速）」。
- **实施模式（写代码 + 分层验收）**：当本次请求已明确授权修改与验证时进入。
  - **目标代码形态**：默认面向**通用 PyTorch 项目**（直接修改源码并接入项目配置/环境变量开关）；若为 TurboPhysAI 接入任务，顺从其 `OptimizationGroup` / recipe 架构。
  - **工程底线**：保留**开关 + 可用性检查 + 回退三件套**；不混入其他领域优化；严格区分四层结论（`开关已开启` ≠ `优化路径真实命中` ≠ `数值与功能等价` ≠ `端到端稳态提速`）；涉及精度变更（如 FP32→TF32）单独按精度改动立项与验收。无目标 GPU/依赖时完成静态接入与 CPU/边界单测，并将设备数值与性能诚实标为"未验证"。
  - **固定 4 项交付清单**：
    1. **实际改动与启停开关**（修改文件、默认开关值与启停方式）；
    2. **已跑检查与结果**（数值对照、边界/回退单测、kernel 命中确认、稳态性能与预热/样本说明）；
    3. **未验证项与限制**（明确环境缺口与适用前提）；
    4. **回退方式（按生命周期说明）**：关闭开关恢复原路径；**仅当**改动涉及进程级配置/环境变量、已安装模块包装、已捕获编译缓存（含 `cache_key_tag`）或已构造优化器状态时，才要求重启进程或重建模型/优化器。

## 主干流程

1. **确认范围与基线**：从 Profiler Trace 定位热点和实际调用链。缺 Trace 时按取证协议降级，静态扫描结果标注「热点待实测」。记录 PyTorch、后端、GPU 架构与相关依赖版本，约定容差和测速基线。
2. **按模式选型**：查 [K1–K10](references/patterns.md) 的适用前提、反例与守卫契约。明确跨层联动项、原路径和失败策略，再判断是否具备实施条件。
3. **分层验收与交付**：按 [验证方法](references/validation.md) 检查路径命中、数值、边界/回退和性能。涉及状态或集合的替换增加 checkpoint 和分布式验收。

## 参考索引（按需加载）

- [references/patterns.md](references/patterns.md)：K1–K10 通用模式（证据状态、Before 坏味道 → After 手法、前提/不适用条件、最小验证）。
- [references/validation.md](references/validation.md)：数值正确性、边界/回退、分布式/checkpoint、性能验收与按生命周期回退细则。
- [references/turbophysai-kernel-evidence.md](references/turbophysai-kernel-evidence.md)：从 16 模型语料筛选的证据、受限案例、分支文件链接和核查 SHA。
- [scripts/scan_kernel_candidates.py](scripts/scan_kernel_candidates.py)（附自测 [scripts/selftest.py](scripts/selftest.py)）：缺 Trace 时用标准库扫描候选。使用支持被扫源码语法的 Python。JSON 输出包含 `disclaimer`、`files_scanned`、`files_skipped`、`skipped_files` 与 `findings`，扫描状态同时体现在 `stderr` 和退出码；动态参数和跨文件注册追调用链核实。
