---
name: opt-layout
description: Analyze or implement PyTorch channels_last and channels_last_3d memory layouts for convolutional subgraphs on CUDA or ROCm/HIP. Use for convolution hotspots, repeated layout conversions, or debugging layout propagation. Hand off global compilation and recompilation to opt-compile, and kernel selection to opt-kernel.
metadata:
  version: "1.0.1"
---

# Optimize Channels-Last

## 定位与边界

- **目标领域**：4D（`torch.channels_last`）与 5D（`torch.channels_last_3d`）卷积负载的物理内存布局优化（CUDA 与 ROCm/HIP）。
- **核心作用**：在不改变逻辑维度（`[N,C,H,W]` / `[N,C,D,H,W]`）与模型数学语义的前提下，为真实卷积热点划定低拷贝转换边界、对齐权重/激活/缓存/优化器状态布局，并验证 ROCm/CUDA 后端实际命中与端到端收益。
- **何时使用**：
  - **Trace 信号**：GPU 计算时间集中在 `Conv2d`、`Conv3d`、`ConvTranspose3d`、3D VAE、视觉 backbone 或支持 NHWC/NDHWC 分支的卷积类扩展（如 DCN、GridSampler），或 profiler 显示卷积前后存在反复隐式 NCHW↔NHWC 拷贝。
  - **代码特征信号**：包含密集 2D/3D 卷积子图、5D↔4D `rearrange`/相机维展平边界、卷积相关时序缓存、或已设置 `PYTORCH_MIOPEN_SUGGEST_NHWC/NDHWC` 却未核实张量与权重是否实际转成 channels-last。
  - **用户诉求**：评估是否值得启用 channels-last、选定整模型 vs 子模块转换范围、排查启用后 `view` 报错/checkpoint 不兼容/3D 反向变慢或未提速问题。
- **明确不做什么 & 多 Skill 主导/分流规则**：
  - **不改逻辑接口**：物理 `memory_format` 不是 `permute(0,2,3,1)` 后的逻辑 NHWC/NDHWC 接口；不按维数盲改 token、相机维、索引或元数据张量。
  - **vs `opt-compile`**：卷积子图的 `channels_last` 转换边界与局部 stride 保持由本 Skill 主导；全局 `torch.compile` 范围、重编译治理与 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION` 全局开关由 `opt-compile` 主导，消融时分开开关。
  - **vs `opt-kernel`**：融合卷积/厂商算子本身的替换选型（如 hipDNN/DCN 替换、`custom_op` 注册）由 `opt-kernel` 主导；本 Skill 仅负责其输入/权重/缓存的 channels-last 布局契约对齐。
  - **vs `opt-runtime-env` 与 `opt-sync`**：`PYTORCH_MIOPEN_SUGGEST_NHWC`/`NDHWC` 作为 ROCm 布局生效前提在改布局代码时由本 Skill 联合核查；仅修改启动环境变量（如 `MIOPEN_FIND_MODE`、`MIOPEN_FIND_ENFORCE` 或 TF32/GEMM 开关，不改模型布局源码）交 `opt-runtime-env`；`.item()`/`.cpu()` 同步问题交 `opt-sync`。

## 工作模式（按当次用户授权判定）

- **分析模式（只读）**：当次请求只询问方法、适用性或兼容性，或尚未授权改代码时进入。可读取源码和已有 Trace、查询本地版本、运行只读扫描。启动训练、新采集和安装依赖按各自授权判断。
  - **必填交付要素**：涉及位置与符号、建议改法（关联模式编号 L1–L7）、证据级别（`[实测]` / `[静态]` / `[推断]`）、关键前提与风险、最小验证项、缺失证据。
  - **默认展示模板（多候选评审）**：使用 **4 列候选主表 + Top 1–2 短卡片**，低优项不展开，不默认刷屏大段 diff；单点咨询或无可靠卷积热点时灵活简化，不强行套表。

    | 优先级 | 位置与符号 | 建议改法 | 证据级别 |
    | --- | --- | --- | --- |
    | P0 / P1 / 暂缓 | `file:line` + 类/函数名 | 一句话说明转换范围/边界处理与适用模式（如 L1/L2/L4） | `[实测]` / `[静态]` / `[推断]` |

    仅对 Top 1–2 候选补充 3 行短卡片：**关键前提与风险**、**最小验证项**、**缺失证据/待核实项**。
- **实施模式（写代码 + 分层验收）**：当次请求已授权修改时，在授权范围内接入和验证。运行训练或新采集需要相应授权。
  - **目标代码形态**：默认面向**通用 PyTorch 项目**（直接修改源码并接入项目配置/环境变量开关）；若为 TurboPhysAI 接入任务，顺从其 `OptimizationGroup` / recipe 架构。
  - **工程底线**：保留**可关闭开关 + 范围/维度检查 + 明确回退**；严格区分四层结论（`配置/开关已开启` ≠ `布局实际落到卷积/算子输入` ≠ `数值与状态等价` ≠ `含转换成本的端到端稳态提速`）；不把 `torch.compile`、`MIOPEN_FIND_MODE` 或精度变更的收益归因于布局。无目标 GPU/依赖时完成静态接入与 CPU/边界单测，并将设备数值与性能诚实标为"未验证"。
  - **固定 4 项交付清单**：
    1. **实际改动与启停开关**（修改文件、转换作用域、默认开关值与启停方式）；
    2. **已跑检查与结果**（边界 `stride()`/`is_contiguous(memory_format=...)` 核对、数值与 checkpoint 对照、含转换开销的稳态性能与预热/样本说明）；
    3. **未验证项与限制**（环境缺口、未验证的自定义算子/ROCm 开关等）；
    4. **回退方式（按生命周期说明）**：关闭开关并按影响清理状态。单独注册的输入 hook 可移除句柄。权重、优化器或编译缓存已改变时重建受影响的对象，进程级状态需要隔离时使用新进程。

## 主干流程

1. **确认卷积热点与基线**。查已有 Trace 中的目标卷积及其前后复制，核对 PyTorch、后端、硬件、实际 shape、编译和分布式配置。缺 Trace 时用静态扫描器定位卷积候选，手工追踪相机展平与 YAML 配置。证据不足时可暂缓，说明待验证假设和最小取证方式，例如在授权后采集目标子图附近的短 Trace。确认卷积或布局复制在关键路径上再优先试点。缺目标 GPU 时交付静态或可执行的 CPU 检查，并标注设备结论。
2. **用 L1/L2 划小范围试点**。选择权重转换的生命周期点及激活进出边界，检查退化维、5D↔4D 切换和已有 stride。涉及序列化看 L4，涉及算子或缓存契约看 L5，涉及局部编译布局看 L6，涉及 ROCm 布局提示看 L7。L3 只在目标反向确有瓶颈且输入梯度契约匹配时评估。
3. **分层验收与交付**。按 [验证方法](references/validation.md) 验证命中、数值和含转换成本的稳态性能。联动项成组启停和回退。只能测整组时报告组合效果。为了守住正确性而选择保守实现的行为标为「正确性保护（非提速）」。

## 参考索引（按需加载）

- [references/patterns.md](references/patterns.md)：L1–L7 通用模式（证据状态、Before 坏味道 → After 手法、不适用/风险、最小验证，含 ROCm/MIOpen 布局陷阱）。
- [references/validation.md](references/validation.md)：边界布局核对、退化维与数值/checkpoint 验证、变量隔离测速与按生命周期回退细则。
- [references/turbophysai-channels-last-evidence.md](references/turbophysai-channels-last-evidence.md)：需要核对语料来源、原始 PR 或模型特定约束时查看，包含快照 SHA 与聚合分支源码索引。
- [scripts/scan_channels_last_candidates.py](scripts/scan_channels_last_candidates.py)（附自测 [scripts/selftest.py](scripts/selftest.py)）：纯 Python 标准库静态卷积子图与布局陷阱扫描器（仅产出静态候选，不证明热点或收益）。
