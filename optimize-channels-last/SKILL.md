---
name: optimize-channels-last
description: Analyze or implement PyTorch channels-last (4D) and channels-last-3d (5D) memory-format optimization for convolutional workloads on CUDA or ROCm/HIP, including subgraph conversion boundaries, 5D/4D rearrange transitions, asymmetric 3D conv backward layout, optimizer-state checkpoint contiguity, custom-op/cache layout contracts, torch.compile stride guards, and ROCm PYTORCH_MIOPEN_SUGGEST_NHWC/NDHWC layout hints. Use when profiling shows Conv2d/Conv3d/ConvTranspose3d hotspots or when evaluating/debugging channels_last/channels_last_3d conversions. Do not use for arbitrary logical permute(0,2,3,1) NHWC API rewrites, global torch.compile scope or TORCHINDUCTOR_LAYOUT_OPTIMIZATION tuning (use torch-compile-integration), MIOPEN_FIND_MODE or general runtime env tuning, host-device sync removal (use optimize-sync-elimination), or non-layout operator kernel replacement (use optimize-kernel-replacement).
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
  - **vs `torch-compile-integration`**：卷积子图的 `channels_last` 转换边界与局部 stride 保持由本 Skill 主导；全局 `torch.compile` 范围、重编译治理与 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION` 全局开关由 `torch-compile-integration` 主导，消融时分开开关。
  - **vs `optimize-kernel-replacement`**：融合卷积/厂商算子本身的替换选型（如 hipDNN/DCN 替换、`custom_op` 注册）由 `optimize-kernel-replacement` 主导；本 Skill 仅负责其输入/权重/缓存的 channels-last 布局契约对齐。
  - **vs 运行时/后端调优与同步消除**：`PYTORCH_MIOPEN_SUGGEST_NHWC`/`NDHWC` 作为 ROCm 布局生效前提由本 Skill 主导核查；`MIOPEN_FIND_MODE`（算法搜索策略）、TF32/GEMM 精度开关仅作消融隔离与分流提示，不展开调参；`.item()`/`.cpu()` 同步问题交 `optimize-sync-elimination`。

## 工作模式（按当次用户授权判定）

- **分析模式（只读）**：当用户仅询问是否适合启用、转换边界、兼容性或未明确授权修改源码与运行训练时进入。**不修改项目源码、不启动训练或采集**；允许读取源码、Trace 及运行只读静态扫描脚本。
  - **必填交付要素**：涉及位置与符号、建议改法（关联模式编号 L1–L7）、证据级别（`[实测]` / `[静态]` / `[推断]`）、关键前提与风险、最小验证项、缺失证据。
  - **默认展示模板（多候选评审）**：使用 **4 列候选主表 + Top 1–2 短卡片**，低优项不展开，不默认刷屏大段 diff；单点咨询或无可靠卷积热点时灵活简化，不强行套表。

    | 优先级 | 位置与符号 | 建议改法 | 证据级别 |
    | --- | --- | --- | --- |
    | P0 / P1 / 暂缓 | `file:line` + 类/函数名 | 一句话说明转换范围/边界处理与适用模式（如 L1/L2/L4） | `[实测]` / `[静态]` / `[推断]` |

    仅对 Top 1–2 候选补充 3 行短卡片：**关键前提与风险**、**最小验证项**、**缺失证据/待核实项**。
- **实施模式（写代码 + 分层验收）**：当本次请求已明确授权修改与验证时进入。
  - **目标代码形态**：默认面向**通用 PyTorch 项目**（直接修改源码并接入项目配置/环境变量开关）；若为 TurboPhysAI 接入任务，顺从其 `OptimizationGroup` / recipe 架构。
  - **工程底线**：保留**可关闭开关 + 范围/维度检查 + 明确回退**；严格区分四层结论（`配置/开关已开启` ≠ `布局实际落到卷积/算子输入` ≠ `数值与状态等价` ≠ `含转换成本的端到端稳态提速`）；不把 `torch.compile`、`MIOPEN_FIND_MODE` 或精度变更的收益归因于布局。无目标 GPU/依赖时完成静态接入与 CPU/边界单测，并将设备数值与性能诚实标为"未验证"。
  - **固定 4 项交付清单**：
    1. **实际改动与启停开关**（修改文件、转换作用域、默认开关值与启停方式）；
    2. **已跑检查与结果**（边界 `stride()`/`is_contiguous(memory_format=...)` 核对、数值与 checkpoint 对照、含转换开销的稳态性能与预热/样本说明）；
    3. **未验证项与限制**（环境缺口、未验证的自定义算子/ROCm 开关等）；
    4. **回退方式（按生命周期说明）**：关闭开关恢复原路径；**仅当**改动涉及进程级环境变量（如 `PYTORCH_MIOPEN_SUGGEST_NHWC`）、已原地转换的模型权重/已构造优化器状态、已注册模块 hook 或已捕获编译缓存时，才要求重启进程或重新构建模型与优化器。

## 主干流程

1. **确认卷积热点与基线**：从 Profiler Trace 确认 Conv2d/Conv3d/卷积扩展是否为真实热点；缺 Trace 时运行 [scripts/scan_channels_last_candidates.py](scripts/scan_channels_last_candidates.py) 速览卷积分布、5D↔4D 边界、逻辑 `permute` 命名陷阱及已有布局入口（明确标注"热点待实测"）。记录 PyTorch、CUDA/ROCm、硬件版本、输入真实 shape、compile/分布式框架与默认环境变量。
2. **按模式划定边界与设防**：对照 [通用模式（L1–L7）](references/patterns.md) 逐项设计：
   - **范围与边界**：L1 按卷积子图选作用域与生命周期点（先小范围再扩整模型，混合 3D/2D 分维转换）；L2 显式 `to(memory_format=...)`、退化维歧义、5D↔4D `rearrange` 边界与防 `permute` 命名混淆。
   - **训练反向与状态兼容**：L3 3D 卷积前反向非对称布局（前向 `channels_last_3d` + 反向权重梯度回 NCDHW）；L4 保存前递归 `.contiguous()` 清洗非连续优化器状态。
   - **生态与后端对齐**：L5 自定义/融合算子与时序缓存同格式契约（防静默回退）；L6 与 `torch.compile` 的边界、stride guards 与图复用；L7 ROCm `PYTORCH_MIOPEN_SUGGEST_NHWC/NDHWC` 版本核对与独立消融（与 `MIOPEN_FIND_MODE`/`TORCHINDUCTOR_LAYOUT_OPTIMIZATION` 隔离）。
3. **分层验收与交付**：按 [验证方法](references/validation.md) 执行边界布局核对、退化维与训练/checkpoint 验证、变量隔离测速与按生命周期回退，按上述 4 项清单交付。

## 参考索引（按需加载）

- [references/patterns.md](references/patterns.md)：L1–L7 通用模式（证据状态、Before 坏味道 → After 手法、不适用/风险、最小验证，含 ROCm/MIOpen 布局陷阱）。
- [references/validation.md](references/validation.md)：边界布局核对、退化维与数值/checkpoint 验证、变量隔离测速与按生命周期回退细则。
- [references/turbophysai-channels-last-evidence.md](references/turbophysai-channels-last-evidence.md)：TurboPhysAI 16 模型语料跨模型证据与固定提交永久链接索引（静态读源码所得，不含实测收益）。
- [scripts/scan_channels_last_candidates.py](scripts/scan_channels_last_candidates.py)（附自测 [scripts/selftest.py](scripts/selftest.py)）：纯 Python 标准库静态卷积子图与布局陷阱扫描器（仅产出静态候选，不证明热点或收益）。
