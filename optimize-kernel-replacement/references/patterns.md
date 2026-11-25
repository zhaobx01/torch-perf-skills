# 算子替换的通用模式（K1–K10）

每条模式均提炼自 TurboPhysAI 16 模型语料库与对应上游基线源码的静态对照（完整溯源与固定快照链接见 [证据](turbophysai-kernel-evidence.md)），不含性能实测数字；实际收益与数值表现必须在目标项目重新测量。

每条模式均标注**证据状态**：
- **已核实改造案例**：Before 基线源码与 After 优化实现均已核实；
- **实现参考，基线待核实**：已核实 After 实现与契约逻辑，但对应 Before 位于外部运行库或暂未逐行核对。

---

## K1 替换的三件套：开关、可用性检查、回退

- **证据状态**：已核实改造案例（`openwam`、`groot`、`openpi`、`lerobot_act`）
- **Before 坏味道**：直接写死替换实现，或只在 import 时粗略判断库存在而不校验运行时 shape/dtype/连续性/分布式配置；对工厂函数传入参数生成器时，校验阶段意外耗尽迭代器。
- **After 手法**：新实现放在独立函数/包装层中由开关控制；安装期校验静态依赖（包括底层扩展符号如 `amp_C.multi_tensor_adam`），调用期逐项校验设备、dtype、维度对齐、连续性/布局、编译/hook 状态与分布式互斥项（如 DeepSpeed CPU offload），不满足则走原实现。对接收迭代器的工厂入口（如 `AdamW(params)`），先将生成器 materialize 一次再逐组校验，并尊重调用方显式参数。
- **不适用条件**：用户显式强选某加速路径、且缺失时应立即暴露错误而非静默降级的场景（例如已精确标记目标模块但输入违背契约时，应明确抛错）。
- **最小验证**：逐条构造不满足前提的输入（CPU 张量、非连续、不支持 dtype、带 hook、空/生成器参数），确认走原实现且输出一致；对强选路径确认违约时明确抛错。

## K2 私有接口、内部钩子与编译缓存隔离

- **证据状态**：已核实改造案例（`openwam`、`maptrv2`、`lingbot_vla_v2`、`lingbot_va`；`cosmos3` 属实现参考，基线待核实）
- **Before 坏味道**：无保护地调用库私有函数（如 FA2 `_wrapped_flash_attn_varlen_*`）、假设第三方优化器内部状态布局不变，或 hook Inductor 内部代码生成函数却不隔离 FX graph cache 导致复用旧产物。
- **After 手法**：
  - 调私有函数前校验源码 SHA256 白名单（OpenWAM）或断言符号存在并限定版本范围（Cosmos3），认不出就打日志回退公开入口。
  - 包装第三方优化器前用 AST 检查其内部假设（如 MapTRv2 验证 `FusedAdam.step` 确为 `group["step"]` 而非 `state["step"]`）。
  - Hook Inductor 内部函数改变生成代码（如 FlexAttention FP32 dot `ieee -> tf32`）时，**同步追加 `torch.compiler.config.cache_key_tag`** 隔离编译缓存。
- **不适用 / 风险**：与特定库版本强绑定，升级后白名单/钩子可能失效回退；源码形态匹配不证明数值等价。
- **最小验证**：mock 不匹配源码/版本确认触发回退或明确报错并记日志；验证修改编译产物的开关开启/关闭对应不同 `cache_key_tag`。

## K3 融合优化器：从 AdamW 扩展到非 AdamW 批量化更新

- **证据状态**：已核实改造案例（`openvla`、`openpi`、`bevformer`、`dit4dit`、`openwam`、`maptrv2`、`groot`、`lerobot_act`、`lingbot_vla_v2`；`sparse4d` 注册项属实现参考，基线待核实）
- **Before 坏味道**：`torch.optim.AdamW` 未启用 `fused=True` 导致逐 tensor 或多趟 `foreach` kernel；直接换 Apex `FusedAdam` 未处理 checkpoint 互通、`grad=None` 步数分叉或 DeepSpeed CPU offload 冲突；Muon 等矩阵优化器逐参数循环做 `lerp_`/`mul_`/`add_` 且用固定小块切分 Newton-Schulz。
- **After 手法**：
  - **原生 `AdamW(fused=True)`**：在未显式指定 `fused`/`foreach` 且所有参数为 CUDA 浮点且非 `differentiable` 时注入 `fused=True`（或 `setdefault`）；配置表框架先确认 `type == "AdamW"`。
  - **Apex `FusedAdam` 适配**：校验 `FusedAdam` + `multi_tensor_applier.available` + `amp_C.multi_tensor_adam`，排除 CPU offload，注意注册表导入顺序；拒绝不支持选项（`amsgrad`/`maximize`/`capturable`/`differentiable`/`master_weights`/非 FP32/稀疏梯度）；禁止同组部分 `grad=None` 致步数漂移且全空组不递增 step；实现 `state_dict`/`load_state_dict` 与标准 PyTorch AdamW 双向无损互转。
  - **非 AdamW 批量化与融合（Muon）**：同 shape 参数用 int64 指针表做 multi-tensor Triton 融合（单次 launch 完成整组 momentum/apply，小张量设融合门槛）；按中间矩阵（`A=[B,M,M]` bf16 字节预算）动态决定 megabatch 大小；FSDP 下所有 rank（含 `grad=None`）以相同排序参与 collective，仅对有梯度项 apply。
- **不适用 / 风险**：融合实现舍入可能与逐 tensor 略有差异；极小张量上 ATen 原生算子已近带宽上限，单张量强行融合反而更慢。
- **最小验证**：同初始权重与梯度跑多步对比参数与状态；做"融合保存→标准加载续训"和"标准保存→融合加载续训"双向往返；覆盖 `grad=None` 与分布式 FSDP/ZeRO 无死锁。

## K4 逐元素表达式融合、RoPE 实数化与常量缓存

- **证据状态**：已核实改造案例（`openwam`、`dreamzero`、`lingbot_va`、`lingbot_vla_v2`、`dit4dit`）
- **Before 坏味道**：无状态逐元素调制/门控残差/RMSNorm 零散执行；RoPE 用 `torch.polar` + `view_as_complex`/`view_as_real` 导致 Inductor 遇到复数算子回退；无参 RoPE 频率表每步/每层重复计算；特定 3D/ND `nn.Linear` 走通用路径开销高。
- **After 手法**：
  - 逐元素表达式用带缓存的 `torch.compile(..., dynamic=True, fullgraph=True)` 融合；外层已编译（`torch.compiler.is_compiling()`）时直接 eager，避免嵌套分派多份特化。
  - 复数 RoPE 改写为实数交错 cos/sin 乘加进图融合，**严格保留原计算精度**（如 `float64` 角度与旋转计算，不把实数化变成降精度）。
  - 无参 RoPE 频率按 `(shape, fps, device)` 或 `weakref(position_ids)` 在 `torch.no_grad()` 下 `.detach()` 有界缓存，不进入 `state_dict` 或 DDP/DeepSpeed 广播。
  - 选定 `nn.Linear` 展平走 2D `torch.addmm`/`torch.mm` 时，保留原模块与参数，有 hook/自定义 forward 时回退，并**显式排除 BF16 舍入敏感层**（如 `action_embedder`）。
- **不适用条件**：复数 RoPE 并非热点或不需要进编译图时不必改写；对 `addmm` 舍入敏感的嵌入/输出层禁止替换。
- **最小验证**：复数→实数 RoPE 确认近似逐位等价；检查重编译次数、缓存不进入 `state_dict()` 且变 shape 时有界释放；逐层核对排除层未被改写。

## K5 注意力路径与后端选择：先正确，再去冗余

- **证据状态**：已核实改造案例（`groot`、`dreamzero`、`lerobot_act`、`dit4dit`、`lingbot_vla_v2`、`lingbot_va`、`openwam`；`cosmos3` 属实现参考，基线待核实）
- **Before 坏味道**：HIP 等后端默认 SDPA 反向出现 NaN 却未设显式后端保护；仅包装模块方法却遗漏框架全局注意力注册表；等长 batch（`q_lens`/`k_lens` 为 `None`）仍 flatten + 构造 `cu_seqlens` 走 `flash_attn_varlen_func`；`nn.MultiheadAttention` 默认开启未被消费的 `average_attn_weights=True`；等头宽场景无条件执行 `repeat_interleave(1)` 产生多余拷贝。
- **After 手法**：
  - 后端选择做成显式可覆盖逻辑（环境变量 + 平台默认）；为规避反向 NaN 强制 `SDPBackend.MATH` 属于**正确性兜底**，严禁作为提速汇报。脱离 Dynamo 图时连同全局注册表（如 `AttentionInterface._global_mapping`）一并替换。
  - 等长 batch 且满足 FA2 契约（fp16/bf16、head_dim 合法、非导出模式）时直调 dense `flash_attn_func` 绕开 varlen 打包；固定有效段 cross-attention 在全量校验 shape/padding 后只对前 `valid` token 调 `flash_attn_func` 再 pad 回原长。
  - 未显式传参时默认关闭 `average_attn_weights=False`（或确认 `need_weights=False`）；GQA 扩展加 `if repeat != 1:` 保护跳过空转拷贝。
- **不适用条件**：存在真实变长 padding mask、调用方明确消费平均注意力权重、或目标平台 FA2 反向存在数值异常时不可启用对应捷径。
- **最小验证**：同输入对比**前向输出与反向梯度**（必测反向防 NaN），覆盖变长/等长、空 batch、全 mask、causal/非 causal 及 GQA（`repeat > 1`）与等头宽（`repeat == 1`）。

## K6 外部与原生扩展 Kernel 封装：`custom_op` 与崩溃守卫

- **证据状态**：已核实改造案例（`bevformer`、`sparse4d`、`lingbot_vla_v2`、`lerobot_act`；`operators/upsample_bilinear_2d.py` 与 `sparse_conv`/`voxelization` 属实现参考，基线待核实）
- **Before 坏味道**：在 `torch.compile` 区域内直接调用 `autograd.Function` 或外部库导致频繁断图；模块顶层硬导入可选厂商库导致缺库即崩溃；原生 C++/HIP 扩展缺少 Python 层 shape/dtype/连续性校验，越界传入直接触发 **coredump**。
- **After 手法**：
  - 要进图的外部/Triton kernel 注册为 `torch.library.custom_op`，补齐与真实输出 shape/dtype/device 一致的 `register_fake`；需反传时补 `register_autograd`（仅支持一阶梯度时在反向开头检查 `if torch.is_grad_enabled(): raise RuntimeError`）。
  - 可选厂商库延迟导入或捕获 `ImportError`，由运行时条件决定是否启用替换。
  - 原生 `_C` 扩展在 Python 包装入口逐条校验维度、支持 size、dtype、连续性与参数组合（如 `align_corners`、`antialias`、`reduce_type`），不满足即回退或抛 `ValueError`。
- **不适用条件**：冻结推理/纯前向算子无需强补 `register_autograd`；不在编译区域内的扩展可按需权衡是否改 `custom_op`。
- **最小验证**：eager 与 `torch.compile` 均走通前反向；核对 `register_fake` 元信息与真实输出一致；故意传越界/非连续/不支持 dtype 输入确认在进入底层扩展前即被拦截。

## K7 融合卷积类算子：记录输入契约与被排除的层

- **证据状态**：实现参考，基线待核实（`cosmos3` 3D conv+bias；`bevformer` DCN fusion graph 属已核实改造案例）
- **Before 坏味道**：普通卷积与 bias/concat 分离执行，但直接替换为 `hipDNN` 等融合图时未按通道/布局/dtype/梯度约束过滤，导致不兼容层报错或选到退化 tile。
- **After 手法**：将支持的 dtype（如仅 bf16）、通道数白名单、布局（`channels_last_3d` 连续）、stride/groups 与 `requires_grad` 契约逐条写成调用前检查，不满足回退参考实现；显式记录被排除的层及原因（如 Cosmos3 排除输入通道为 12 的 stem 层，源码注释自述避免退化 tile，属「来源自述，未复测」）。
- **不适用条件**：输入需反传但融合图仅支持前向、或不在支持通道/dtype 白名单内的层。
- **最小验证**：对白名单内包含层与被显式排除层各取样本，确认实际执行路径（命中 vs 回退）与数值均符合预期。

## K8 FlexAttention 调优：Head 共享 BlockMask、Tile、精度与模板守卫

- **证据状态**：已核实改造案例（`lingbot_vla_v2`、`lingbot_va`）
- **Before 坏味道**：对 head 无关的 3D mask 先 `.expand(-1, num_heads, -1, -1).contiguous()` 再调 `create_mask` 分配 4D 稠密 mask 才建 `BlockMask`；前反向 tile 使用默认值不适配目标架构；FP32 FlexAttention 改走 `tf32` 却未隔离编译缓存；定制 Triton 模板缺少架构/shape 守卫导致非目标输入出错或性能回退。
- **After 手法**：
  - Head 无关 3D mask 直接 `create_block_mask(..., H=1)` 广播并删除冗余 `create_mask` 4D 分配，按稀疏粒度核对 `block_size`（如 64 vs 128）。
  - 通过 `kernel_options` 覆盖前反向 tile（`BLOCK_M`/`BLOCK_N`/`bwd_BLOCK_*`、`num_warps`、`num_stages`），**确保 `SPARSE_*_BLOCK_SIZE` 能被对应 tile 整除**，并与调用方已有选项合并。
  - FP32 FlexAttention 切 `tf32` 时，按 K2 同步追加 `torch.compiler.config.cache_key_tag`，且作为精度变更单独验收。
  - 架构定制模板（如 `gfx936` 的 `waves_per_eu`/`kpack`）用 Inductor `ShapeEnv` 符号 guard 锁 `gcnArchName`、dtype、shape、stride 与预期 `kernel_options`，未命中退回默认配置；**将性能白名单与序列长度整除性正确性 guard 解耦**。
- **不适用条件**：不同 head 具有不同 mask 模式时不可设 `H=1`；对 FP32 严格 IEEE 精度有硬要求时不可切 `tf32`。
- **最小验证**：比较 `H=1` 与原 `H=num_heads` 的前向与反向梯度等价；覆盖序列长度整除与非整除 `block_size`/tile 边界；验证 `tf32` 缓存标签生效与精度容差；验证非白名单 shape/架构安全退回默认路径。

## K9 单算子绕开低效编译器展开与局部低精度替换

- **证据状态**：已核实改造案例（`lingbot_vla_v2` `upsample_bilinear_fast`；`lerobot_act` `qk_tf32` 属已核实改造案例）
- **Before 坏味道**：`torch.compile` 将个别简单算子（如 4D 2x bilinear `nn.Upsample`）展开为低效 gather kernel 反而慢于原生 ATen；FP32 注意力中 QK 矩阵乘成为主要耗时热点。
- **After 手法**：
  - **单算子脱离 Inductor 回原生 ATen**：对精确匹配参数签名的热算子（如 `Upsample(mode="bilinear", scale_factor=2, align_corners=False)` 且 4D/fp32/非空），用 `@torch.compiler.disable` 精确隔离直调 `torch.ops.aten.upsample_bilinear2d.default`，不满足条件的调用原样透传；同时确认引入的局部图断代价小于原退化 kernel 代价。
  - **局部低精度计算路径**：将 FP32 QK 矩阵乘替换为 TF32 输入 + FP32 累加的 `custom_op`（反向保持标准 `torch.bmm` 并显式拒绝高阶梯度），前置全量参数契约过滤，并将精度差异单独验收。
- **不适用 / 边界**：若退化来自全局性 Inductor Triton 启发式候选不足（如 `hip_triton_configs`）或大面积图断，**移交 `torch-compile-integration` / 编译后端调优侧**，不在本 Skill 内打全局编译器补丁；不可接受 TF32 精度误差时禁用局部低精度替换。
- **最小验证**：用 profiler 核对退化 kernel 消失且端到端耗时未因图断变慢；对局部 TF32 路径单独验证前反向梯度容差与任务指标收敛性。

## K10 定点 Triton / 原生扩展 Kernel 的门槛与安全边界

- **证据状态**：已核实改造案例（`lingbot_vla_v2` Muon 融合、`openwam` ZeRO 打包/范数；`lerobot_act` `pool_backward_gather` 属实现参考，基线为 `torchvision` ResNet `MaxPool2d`）
- **Before 坏味道**：对所有尺寸张量无差别走手写融合 kernel（小张量反而变慢）；multi-tensor 指针表未设缓存上限或在 CUDA Graph 捕获中做 pinned 异步 H2D 触发异常；定点 kernel 悄悄改变原通信前舍入或归约精度；未限定目标模块导致意外接管其他层。
- **After 手法**：
  1. **最小元素数门槛**：单张量融合设元素数阈值（如 Muon `_FUSE_MIN_ELEMS = 2 << 20`），小张量走原生 ATen，大批量同 shape 张量走 multi-tensor 指针表。
  2. **指针表缓存与 CUDA Graph 安全**：按 `data_ptr` 元组缓存 int64 指针表并设容量上限；热路径用 pinned memory + `non_blocking=True` 拷贝，遇到 `torch.cuda.is_current_stream_capturing()` 或非 CUDA 设备时改走安全路径。
  3. **保留原精度与舍入时机**：通信打包保留通信前的 bucket dtype 舍入顺序；范数归约保持原 `fp64` 累加精度。
  4. **精确作用范围与梯度声明**：仅对显式标记的目标模块（或严格 shape/layout 白名单）启用，并显式拒绝不支持的高阶梯度。
- **不适用条件**：通用逐元素表达式优先交 `torch.compile`（K4），非热点算子或通用 GEMM/Attention 从零研发不适用本模式。
- **最小验证**：分别测低于/高于元素数门槛、首次构建与缓存命中、CUDA Graph 捕获（如适用）、非目标模块回退、前反向数值等价及二阶求导报错。

---

## 不适用的场景汇总

- **热点不在这个算子上**：profiler 显示占比极低时，替换后端到端收益可忽略，反而增加兼容维护成本。
- **全局编译与后端启发式调优**：编译范围划分、重编译治理及全局 Inductor Triton 启发式（如 `hip_triton_configs`）交 `torch-compile-integration` / 后端调优侧。
- **从零开发通用高性能算子库**：本 Skill 覆盖算子替换、单算子后端调优及定点 Triton/原生扩展的接入守卫与验收，不替代通用 CUDA/HIP/Triton 算子库从零研发。
