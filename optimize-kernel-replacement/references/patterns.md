# 算子替换的通用模式（K1–K10）

模式及反例来自 TurboPhysAI 16 模型语料的静态核查。各机制的 Before、After 和核查 SHA 见 [证据](turbophysai-kernel-evidence.md)。目标项目的数值与性能须独立验证。

每条模式均标注**证据状态**：
- **已核实改造案例**：Before 基线源码与 After 优化实现均已核实；
- **实现参考，基线待核实**：已核实 After 实现与契约逻辑，对应 Before 位于外部运行库或尚未定位。
- **受限参考／版本兼容性待核对**：保留单模型或绑定特定平台、构建的案例。版本、形状和 ABI 约束见证据，通用模式只提炼可复用的检查。

---

## K1 替换的三件套：开关、可用性检查、回退

- **证据状态**：已核实改造案例（`openpi`、`groot`、`lerobot_act` 的优化器入口）。
- **Before 形态**：OpenPI 的训练入口直接构造 AdamW，GR00T 从 trainer 的配置选择优化器。新增检测时，粗略的 import 检查和提前遍历参数生成器是接入风险。
- **After 手法**：新实现放在独立函数/包装层中由开关控制；安装期校验静态依赖（包括底层扩展符号如 `amp_C.multi_tensor_adam`），调用期逐项校验设备、dtype、维度对齐、连续性/布局、编译/hook 状态与分布式互斥项（如 DeepSpeed CPU offload），不满足则走原实现。对接收迭代器的工厂入口（如 `AdamW(params)`），先将生成器 materialize 一次再逐组校验，并尊重调用方显式参数。
- **不适用条件**：用户显式强选某加速路径、且缺失时应立即暴露错误而非静默降级的场景（例如已精确标记目标模块但输入违背契约时，应明确抛错）。
- **最小验证**：逐条构造不满足前提的输入（CPU 张量、非连续、不支持 dtype、带 hook、空/生成器参数），确认走原实现且输出一致；对强选路径确认违约时明确抛错。

## K2 私有接口、内部钩子与编译缓存隔离

- **证据状态**：MapTRv2 优化器入口、Cosmos3 模型侧 FA2 路由为已核实改造。OpenWAM 私有 FA2 契约、LingBot-VLA-V2 TF32 hook 的运行库 Before 待核实。Cosmos3、LingBot-VA 的私有接口特化均为受限参考，版本兼容性待核对。
- **Before 形态**：Cosmos3 模型调用公开 FA2 入口，MapTRv2 由标准工厂构造优化器。引入私有函数、第三方状态约定或 Inductor hook 后，应考虑接口漂移和旧编译产物复用的风险。
- **After 手法**：
  - 调私有函数前校验实现指纹或已知版本及符号。明确不匹配时按契约回原入口或报告可操作的错误。OpenWAM 指纹不符时返回公开入口，Cosmos3 对选中的 HIP 路径断言必需符号存在。
  - 包装第三方优化器前用 AST 检查其内部假设（如 MapTRv2 验证 `FusedAdam.step` 确为 `group["step"]` 而非 `state["step"]`）。
  - Hook Inductor 内部函数改变生成代码（如 FlexAttention FP32 dot `ieee -> tf32`）时，**同步追加 `torch.compiler.config.cache_key_tag`** 隔离编译缓存。
- **不适用 / 风险**：与特定库版本强绑定，升级后白名单/钩子可能失效回退；源码形态匹配不证明数值等价。
- **最小验证**：mock 不匹配源码/版本确认触发回退或明确报错并记日志；验证修改编译产物的开关开启/关闭对应不同 `cache_key_tag`。

## K3 融合优化器：从 AdamW 扩展到非 AdamW 批量化更新

- **证据状态**：已核实改造案例（`openvla`、`openpi`、`bevformer`、`dit4dit`、`openwam`、`maptrv2`、`groot`、`lerobot_act`、`lingbot_vla_v2`）。`sparse4d` 已核实配置选择与注册，运行和状态转换契约待核实。
- **Before 形态**：多个模型的 AdamW 调用或配置未显式启用 `fused=True`，实际后端由目标版本决定。Muon 基线逐参数更新且用固定块切分 Newton-Schulz。换用 Apex 时另核对 checkpoint 互通、`grad=None` 步进与 CPU offload。
- **After 手法**：
  - **原生 `AdamW(fused=True)`**：优先参考 OpenPI 的守卫。只在调用与参数组都未显式选择 `fused`/`foreach`、`differentiable` 未启用且参数符合目标环境的融合契约时注入。配置表框架先确认 `type == "AdamW"`。较薄的 `setdefault` 包装只适用于已知调用契约。
  - **Apex `FusedAdam` 适配**：校验 `FusedAdam`、`multi_tensor_applier.available` 与 `amp_C.multi_tensor_adam`，排除不支持的 CPU offload 配置并核对注册顺序。MapTRv2 对特定 Apex 实现拒绝不支持的选项、dtype 和稀疏梯度，校验组内步数契约，在空梯度组不递增 step，支持规定格式的 AdamW 状态双向转换。按目标实现逐项重验。
  - **非 AdamW 批量化与融合（Muon）**：同 shape 参数用 int64 指针表做 multi-tensor Triton 融合（单次 launch 完成整组 momentum/apply，小张量设融合门槛）；按中间矩阵（`A=[B,M,M]` bf16 字节预算）动态决定 megabatch 大小；FSDP 下所有 rank（含 `grad=None`）以相同排序参与 collective，仅对有梯度项 apply。
- **不适用 / 风险**：融合实现可能改变舍入。小张量是否值得单独融合应在目标设备测量。
- **最小验证**：同初始权重与梯度跑多步对比参数与状态；做"融合保存→标准加载续训"和"标准保存→融合加载续训"双向往返；覆盖 `grad=None` 与分布式 FSDP/ZeRO 无死锁。

## K4 逐元素表达式融合、RoPE 实数化与常量缓存

- **证据状态**：已核实改造案例（`openwam`、`dreamzero`、`lingbot_va`、`lingbot_vla_v2`、`dit4dit`）
- **Before 形态**：基线中可见逐元素调制、门控残差、复数 RoPE、无参频率表计算和通用 `nn.Linear` 调用。是否存在 kernel 碎片、复数 lowering 限制、重复计算或线性层额外开销，由目标调用与版本决定。
- **After 手法**：
  - 逐元素表达式可用带缓存的 `torch.compile(..., dynamic=True, fullgraph=True)` 融合。存在外层编译时，评估直接调用张量实现能否减少重复特化。OpenWAM 在外层编译且开启 `global_compile` 时采用该分支。
  - 复数 RoPE 改写为实数交错 cos/sin 乘加进图融合，**严格保留原计算精度**（如 `float64` 角度与旋转计算，不把实数化变成降精度）。
  - 无参 RoPE 频率可按 `(shape, fps, device)` 缓存，或在张量身份和输入契约稳定时按 `weakref(position_ids)` 复用。逐例核对 key、失效条件与梯度边界，缓存保持有界且独立于持久化状态。改变频率表示时，读取方、初始化和缓存失效需联动。
  - 选定 `nn.Linear` 展平走 2D `torch.addmm`/`torch.mm` 时，保留原模块与参数，有 hook/自定义 forward 时回退，并**显式排除 BF16 舍入敏感层**（如 `action_embedder`）。
- **不适用条件**：复数 RoPE 并非热点或不需要进编译图时不必改写；对 `addmm` 舍入敏感的嵌入/输出层禁止替换。
- **最小验证**：复数→实数 RoPE 确认近似逐位等价；检查重编译次数、缓存不进入 `state_dict()` 且变 shape 时有界释放；逐层核对排除层未被改写。

## K5 注意力路径与后端选择：先正确，再去冗余

- **证据状态**：已核实改造案例（`groot`、`dreamzero`、`dit4dit`、`lingbot_va`、`cosmos3` 模型侧，以及 `lingbot_vla_v2` video teacher 路由）。LeRobot MHA 已核实模型调用方与替换包装，PyTorch 运行库基线待核实。OpenWAM 私有 FA2 和 `lingbot_vla_v2` depth teacher 的外部实现 Before 待核实。
- **Before 形态**：GR00T 已提供环境覆盖和 Spark 的 MATH 兜底。DreamZero 等长 batch 在 FA2 分支仍打包 `cu_seqlens`，独立状态块逐块调用注意力。DiT4DiT 对照的 Diffusers 在 `repeat == 1` 时也会执行拷贝。对 MHA 先查实际调用是否请求和消费权重。
- **After 手法**：
  - 后端选择保留显式覆盖。GR00T 增加 HIP 默认 MATH，标为「正确性保护（非提速）」。需要脱离 Dynamo 图时，沿实际派发路径和已注册别名定位入口。
  - 等长且满足 mask、平台、dtype 与 head_dim 等契约时可直接走 dense FA2。DreamZero 保留 FA3 优先级和 TensorRT 路由，dense 分支对部分参数使用断言。固定有效段 cross-attention 只对合法前缀计算，随后恢复原长及 mask 语义。
  - 独立且等长的注意力块可折入 batch 维合并调用，再恢复原形状。先核对各块的 mask、随机性及无跨块依赖，覆盖单块、空块和不整除边界。
  - 确认下游权重消费方式与返回契约后，再关闭 `average_attn_weights` 或 `need_weights`。`average_attn_weights=False` 会保留逐头权重并改变返回形状，`need_weights=False` 会取消权重返回。GQA 扩展只在 `repeat != 1` 时执行。
- **不适用条件**：存在真实变长 padding mask、调用方明确消费平均注意力权重、或目标平台 FA2 反向存在数值异常时不可启用对应捷径。
- **最小验证**：同输入对比**前向输出与反向梯度**（必测反向防 NaN），覆盖变长/等长、空 batch、全 mask、causal/非 causal 及 GQA（`repeat > 1`）与等头宽（`repeat == 1`）。

## K6 外部与原生扩展 Kernel 封装：`custom_op` 与崩溃守卫

- **证据状态**：已核实 BEVFormer 扩展封装、Sparse4D 原生替换、LingBot-VLA-V2 video teacher。LeRobot `qk_tf32`、公共原生上采样、`sparse_conv` 与 `voxelization` 属实现参考，运行库或公共算子 Before 待核实。
- **Before 形态与接入风险**：BEVFormer 基线通过 `autograd.Function` 调用扩展，断图情况由目标环境确认。可选库缺失和不合法输入可能造成导入失败或底层崩溃。公共原生上采样源码的崩溃警示为来源自述，未复测，其包装缺少尺寸白名单。
- **After 手法**：
  - 要进图的外部/Triton kernel 注册为 `torch.library.custom_op`，补齐与真实输出 shape/dtype/device 一致的 `register_fake`；需反传时补 `register_autograd`（仅支持一阶梯度时在反向开头检查 `if torch.is_grad_enabled(): raise RuntimeError`）。
  - 可选厂商库延迟导入或捕获 `ImportError`，由运行时条件决定是否启用替换。
  - 原生 `_C` 扩展在 Python 包装入口逐条校验维度、支持 size、dtype、连续性与参数组合（如 `align_corners`、`antialias`、`reduce_type`），不满足即回退或抛 `ValueError`。
- **不适用条件**：冻结推理/纯前向算子无需强补 `register_autograd`；不在编译区域内的扩展可按需权衡是否改 `custom_op`。
- **最小验证**：eager 与 `torch.compile` 均走通前反向；核对 `register_fake` 元信息与真实输出一致；故意传越界/非连续/不支持 dtype 输入确认在进入底层扩展前即被拦截。

## K7 融合卷积类算子：记录输入契约与被排除的层

- **证据状态**：已核实 `bevformer` DCN fusion graph 和 `cosmos3` 模型侧 Wan VAE 入口。后者受限于 HCU、厂商库和输入契约，版本兼容性待核对。
- **Before 形态**：Cosmos3 `CausalConv3d` 先处理 causal padding 和缓存，再调用 Conv3d。迁移到融合图时须保留该契约并核对梯度支持。
- **After 手法**：将 dtype、通道、布局、stride/groups 和梯度契约写成调用前检查。按原路径、参考实现或明确异常处理不支持项，并记录被排除的层及原因。Cosmos3 具体契约见证据。
- **不适用条件**：输入需反传但融合图仅支持前向、或不在支持通道/dtype 白名单内的层。
- **最小验证**：对白名单内包含层与被显式排除层各取样本，确认实际执行路径（命中 vs 回退）与数值均符合预期。

## K8 FlexAttention 调优：Head 共享 BlockMask、Tile、精度与模板守卫

- **证据状态**：已核实 `lingbot_vla_v2` 的 BlockMask 和 tile 参数对照。TF32 hook 的 Inductor Before 待核实。`lingbot_va` 厂商模板为受限参考，版本兼容性待核对。
- **Before 形态**：LingBot-VLA-V2 已显式传 `FLEX_KERNEL_OPTIONS`，并有对 head 无关的 3D mask 扩展及 `create_mask` 构造。切换精度或模板后须处理缓存复用和未命中守卫的风险。
- **After 手法**：
  - Head 无关 3D mask 直接 `create_block_mask(..., H=1)` 广播并删除冗余 `create_mask` 分配。按稀疏粒度和上下游 mask 契约核对 `block_size`。
  - 通过 `kernel_options` 覆盖前反向 tile（`BLOCK_M`/`BLOCK_N`/`bwd_BLOCK_*`、`num_warps`、`num_stages`），**确保 `SPARSE_*_BLOCK_SIZE` 能被对应 tile 整除**，并与调用方已有选项合并。
  - FP32 FlexAttention 切 `tf32` 时，按 K2 同步追加 `torch.compiler.config.cache_key_tag`，且作为精度变更单独验收。
  - 架构定制模板先校验软件构建、资产、架构、dtype、shape、stride 和选项。**将性能白名单与序列整除性等正确性守卫解耦**。分别说明未命中性能白名单和违反正确性契约时走哪条路径。
- **不适用条件**：不同 head 具有不同 mask 模式时不可设 `H=1`；对 FP32 严格 IEEE 精度有硬要求时不可切 `tf32`。
- **最小验证**：比较 `H=1` 与原 `H=num_heads` 的前向与反向梯度等价；覆盖序列长度整除与非整除 `block_size`/tile 边界；验证 `tf32` 缓存标签生效与精度容差；验证非白名单 shape/架构安全退回默认路径。

## K9 单算子绕开低效编译器展开与局部低精度替换

- **证据状态**：`lingbot_vla_v2` `upsample_bilinear_fast` 已核实解码器调用点及替换实现，目标 PyTorch lowering 待复核。`lerobot_act` `qk_tf32` 已核实 After，运行库 Before 待核实。
- **Before 形态**：解码器调用 2x bilinear `nn.Upsample`。Inductor 的 gather 展开属来源自述，未复测。QK 矩阵乘占比和是否值得替换需由目标 Trace 证实。
- **After 手法**：
  - **单算子脱离 Inductor 回原生 ATen**：对精确匹配参数签名的热算子（如 `Upsample(mode="bilinear", scale_factor=2, align_corners=False)` 且 4D/fp32/非空），用 `@torch.compiler.disable` 精确隔离直调 `torch.ops.aten.upsample_bilinear2d.default`，不满足条件的调用原样透传；同时确认引入的局部图断代价小于原退化 kernel 代价。
  - **局部低精度计算路径**：将 FP32 QK 矩阵乘替换为 TF32 输入 + FP32 累加的 `custom_op`（反向保持标准 `torch.bmm` 并显式拒绝高阶梯度），前置全量参数契约过滤，并将精度差异单独验收。
- **不适用 / 边界**：若退化来自全局性 Inductor Triton 启发式候选不足（如 `hip_triton_configs`）或大面积图断，**移交 `torch-compile-integration` / 编译后端调优侧**，不在本 Skill 内打全局编译器补丁；不可接受 TF32 精度误差时禁用局部低精度替换。
- **最小验证**：用 profiler 核对退化 kernel 消失且端到端耗时未因图断变慢；对局部 TF32 路径单独验证前反向梯度容差与任务指标收敛性。

## K10 定点 Triton / 原生扩展 Kernel 的门槛与安全边界

- **证据状态**：已核实 `lingbot_vla_v2` Muon 融合、`pointpillars` scatter、`openwam` ZeRO 组内打包与范数。`lerobot_act` `pool_backward_gather` 的 TorchVision 实现 Before 待核实，属受限参考。
- **Before 形态与接入风险**：PointPillars 按 batch 循环 scatter，Muon 逐张量更新。新增融合路径时核对小尺寸开销、指针表缓存、捕获期上传、原路径舍入和实际接管范围。
- **After 手法**：
  1. **最小元素数门槛**：按目标硬件测量何时走 ATen，何时用单张量或多张量融合。Muon 的局部门槛用于 momentum 分支。
  2. **指针表缓存与 CUDA Graph**：缓存需有边界并核对设备、张量地址与 stream 生命周期。Muon 缓存未命中时根据捕获状态选择建表路径，捕获期兼容性仍需在目标环境验证。
  3. **保留原精度与舍入时机**：通信打包保留通信前的 bucket dtype 舍入顺序；范数归约保持原 `fp64` 累加精度。
  4. **精确作用范围与梯度声明**：仅对显式标记的目标模块（或严格 shape/layout 白名单）启用，并显式拒绝不支持的高阶梯度。
  5. **独立小调用合并**：PointPillars 用一次平坦 `scatter_add_`（累加）合并各 batch 的逐样本覆盖写入；因累加与覆盖在重复坐标处不等价，遇到重复索引、逐维坐标越界（分别核对 `y`/`x` 而非仅扁平索引）、空输入或布局不符时必须回退原路径。同时注意热路径的运行时唯一性/越界检查（如 `torch.unique`、`bool(...any())`）会引入主机-设备同步，必须将检查与同步开销计入端到端实测，避免反噬合并收益（同步成为主瓶颈时交 `optimize-sync-elimination`）。独立注意力块的合并见 K5。
- **不适用条件**：通用逐元素表达式优先交 `torch.compile`（K4），非热点算子或通用 GEMM/Attention 从零研发不适用本模式。
- **最小验证**：分别测低于/高于元素数门槛、首次构建与缓存命中、CUDA Graph 捕获（如适用）、非目标模块回退、前反向数值等价及二阶求导报错。

---

## 不适用的场景汇总

- **热点不在这个算子上**：profiler 显示占比极低时，替换后端到端收益可忽略，反而增加兼容维护成本。
- **全局编译与后端启发式调优**：编译范围划分、重编译治理及全局 Inductor Triton 启发式（如 `hip_triton_configs`）交 `torch-compile-integration` / 后端调优侧。
- **从零开发通用高性能算子库**：本 Skill 覆盖算子替换、单算子后端调优及定点 Triton/原生扩展的接入守卫与验收，不替代通用 CUDA/HIP/Triton 算子库从零研发。
