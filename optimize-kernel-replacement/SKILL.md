---
name: optimize-kernel-replacement
description: Analyze or implement replacement and single-operator kernel-level tuning of PyTorch operators on CUDA or ROCm/HIP: fused optimizers (AdamW fused=True, Apex FusedAdam, batched/fused Muon), FlexAttention tuning (head-shared BlockMask, tile options, TF32 + cache_key_tag, arch-guarded templates), FlashAttention/SDPA backend and dense-vs-varlen routing, redundant attention op removal, real-valued RoPE and compiled pointwise ops, vendor LayerNorm/RMSNorm and fused conv+bias graphs, single-op @torch.compiler.disable fallback to ATen, and custom_op/targeted Triton/native extension wrapping with availability checks, fallback paths, numerical equivalence, and measured performance. Use when profiling shows hot operator/kernel time (attention, optimizer step, norm, RoPE/elementwise, conv, upsample/pool) or code review finds unfused/suboptimal operator backends. Do not use for global torch.compile scope/recompilation or global Inductor Triton heuristic tuning (use torch-compile-integration), host-device sync removal (use optimize-sync-elimination), channels-last layout design (use optimize-channels-last), distributed comm overlap, or writing general GEMM/attention kernel libraries from scratch.
---

# Optimize Kernel Replacement

## 定位与边界

- **目标领域**：单算子/热点内核实现替换与后端调优（CUDA 与 ROCm/HIP）。
- **核心作用**：针对已确认或高度疑似的算子计算热点，用更快或更稳的融合/厂商/定制实现替换原算子，消除冗余打包与低效展开，并守住数值精度、输入契约与可回退边界。
- **何时使用**：
  - **Trace 信号**：GPU 计算时间集中在注意力（SDPA/FlexAttention/FA2 varlen）、优化器 step（逐参数 `foreach`/AdamW/Muon）、归一化（LayerNorm/RMSNorm）、RoPE/逐元素调制、特定卷积/反传池化或退化的 Inductor gather 展开。
  - **代码特征信号**：`AdamW` 未开 `fused`、手写/复杂 `BlockMask` 或等长输入走 `varlen` FA2、复数 `torch.polar`/`view_as_complex` RoPE、`MultiheadAttention` 默认开启未消费的 `average_attn_weights`、无守卫的自定义 `autograd.Function`/`custom_op` 或 C++/HIP 扩展。
  - **用户诉求**：评估或接入融合优化器、切换注意力后端、调优 FlexAttention 参数、替换厂商 kernel 或排查算子替换引发的数值/NaN/回退问题。
- **明确不做什么 & 多 Skill 主导/分流规则**：
  - **vs `torch-compile-integration`**：涉及 Inductor 时，**单算子实现替换、后端/tile 选择、FlexAttention 配置及单算子 `@torch.compiler.disable` 绕开退化展开**由本 Skill 主导；**全局编译范围、图捕获、重编译治理以及全局性 Inductor Triton 启发式配置（如 `hip_triton_configs`）**交给 `torch-compile-integration` 或后端编译调优侧主导。
  - **vs `optimize-sync-elimination`**：`.item()`/`.cpu()`/`nonzero` 导致的 GPU 空闲交给 `optimize-sync-elimination`。
  - **vs `optimize-channels-last`**：2D/3D 卷积子图的 `channels_last`/`channels_last_3d` 布局边界设计交给 `optimize-channels-last`（本 Skill 仅校验目标融合算子所需的输入布局契约）。
  - **其他领域**：数据管线/异步 H2D、分布式通信重叠、激活重计算与从零研发通用 GEMM/FlashAttention 算子库均不在本 Skill 范围。

## 工作模式（按当次用户授权判定）

- **分析模式（只读）**：当用户仅询问优化点、可行性或未明确授权修改源码与运行训练时进入。**不修改项目源码、不启动训练或采集**；允许读取源码、Trace 及运行只读静态扫描脚本。
  - **必填交付要素**：涉及位置与符号、建议改法（关联模式编号 K1–K10）、证据级别（`[实测]` / `[静态]` / `[推断]`）、关键前提与风险、最小验证项、缺失证据。
  - **默认展示模板（多候选评审）**：使用 **4 列候选主表 + Top 1–2 短卡片**，低优项不展开，不默认刷屏大段 diff；单点咨询或无可靠候选时灵活简化，不强行套表。

    | 优先级 | 位置与符号 | 建议改法 | 证据级别 |
    | --- | --- | --- | --- |
    | P0 / P1 / 暂缓 | `file:line` + 类/函数名 | 一句话说明改写方向与适用模式（如 K3/K5/K8） | `[实测]` / `[静态]` / `[推断]` |

    仅对 Top 1–2 候选补充 3 行短卡片：**关键前提与风险**、**最小验证项**、**缺失证据/待核实项**。
- **实施模式（写代码 + 分层验收）**：当本次请求已明确授权修改与验证时进入。
  - **目标代码形态**：默认面向**通用 PyTorch 项目**（直接修改源码并接入项目配置/环境变量开关）；若为 TurboPhysAI 接入任务，顺从其 `OptimizationGroup` / recipe 架构。
  - **工程底线**：保留**开关 + 可用性检查 + 回退三件套**；不混入其他领域优化；严格区分四层结论（`开关已开启` ≠ `优化路径真实命中` ≠ `数值与功能等价` ≠ `端到端稳态提速`）；涉及精度变更（如 FP32→TF32）单独按精度改动立项与验收。无目标 GPU/依赖时完成静态接入与 CPU/边界单测，并将设备数值与性能诚实标为"未验证"。
  - **固定 4 项交付清单**：
    1. **实际改动与启停开关**（修改文件、默认开关值与启停方式）；
    2. **已跑检查与结果**（数值对照、边界/回退单测、kernel 命中确认、稳态性能与预热/样本说明）；
    3. **未验证项与限制**（明确环境缺口与适用前提）；
    4. **回退方式（按生命周期说明）**：关闭开关恢复原路径；**仅当**改动涉及进程级配置/环境变量、已安装模块包装、已捕获编译缓存（含 `cache_key_tag`）或已构造优化器状态时，才要求重启进程或重建模型/优化器。

## 主干流程

1. **确认范围与基线**：优先从 Profiler Trace 定位真实热点算子；缺 Trace 时运行 [scripts/scan_kernel_candidates.py](scripts/scan_kernel_candidates.py) 产出静态候选并明确标注"热点待实测"。记录 PyTorch、CUDA/HIP、GPU 架构（SM/`gcnArchName`）与厂商库版本（FlashAttention、Apex、`lightop`、`hipDNN`、Triton），约定数值容差与测速基线。
2. **按模式选型与设防**：对照 [通用模式（K1–K10）](references/patterns.md) 检查每项候选的适用前提、反例与守卫契约：
   - **基础防线**：K1 开关/检查/回退三件套；K2 私有接口/内部钩子契约校验与 `cache_key_tag` 缓存隔离。
   - **优化器与逐元素**：K3 融合优化器（原生 `AdamW(fused=True)`、Apex `FusedAdam` 双向 checkpoint 适配、Muon 多张量融合与动态 megabatch）；K4 逐元素编译融合、复数 RoPE 实数化（保原精度）与无参 RoPE 缓存。
   - **注意力与 FlexAttention**：K5 注意力后端正确性优先（如 HIP `SDPBackend.MATH` 属防 NaN 回退非提速）、uniform batch 走 dense FA2、去 `average_attn_weights` 与 `repeat == 1` 空转拷贝；K8 FlexAttention 调优（`H=1` 共享 `BlockMask`、tile 整除约束、TF32+缓存标签、架构白名单模板）。
   - **扩展封装与单算子绕开**：K6 `custom_op` + `register_fake` + `register_autograd` 与原生 C++/HIP 扩展防崩溃守卫；K7 融合卷积输入契约与排除层；K9 单算子 `@torch.compiler.disable` 绕开 Inductor 退化展开与局部低精度替换；K10 定点 Triton kernel 的最小元素数门槛、指针表缓存与 CUDA Graph 安全边界。
3. **分层验收与交付**：按 [验证方法](references/validation.md) 执行数值、边界/回退、分布式/checkpoint 与性能验收，按上述交付清单输出结论。

## 参考索引（按需加载）

- [references/patterns.md](references/patterns.md)：K1–K10 通用模式（证据状态、Before 坏味道 → After 手法、前提/不适用条件、最小验证）。
- [references/validation.md](references/validation.md)：数值正确性、边界/回退、分布式/checkpoint、性能验收与按生命周期回退细则。
- [references/turbophysai-kernel-evidence.md](references/turbophysai-kernel-evidence.md)：TurboPhysAI 16 模型语料跨模型证据与固定提交永久链接索引（静态读源码所得，不含实测收益）。
- [scripts/scan_kernel_candidates.py](scripts/scan_kernel_candidates.py)（附自测 [scripts/selftest.py](scripts/selftest.py)）：纯 Python 标准库静态候选扫描器（仅产出静态候选，不证明热点或收益）。
