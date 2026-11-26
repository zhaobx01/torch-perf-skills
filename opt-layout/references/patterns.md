# Channels-Last 通用模式（L1–L7）

常规评估从 L1/L2 开始，按触发条件查 L4–L7。L3 是受限参考。[案例索引](turbophysai-channels-last-evidence.md)记录基线、核查快照及对应符号。下文的风险和反例用于指导检查，具体项目是否遇到这些问题、是否提速均需另行验证。

每条模式均标注**证据状态**：
- **已核实改造案例**：Before 基线源码与 After 优化实现均已核实；
- **实现参考，基线待核实**：已核实 After 的静态契约，具体 Before 路径仍待对照。
- **受限参考 / 配置观察**：分别注明单模型、私有 API 或版本依赖的条件，以及源码或配置能够支持的结论。

---

## L1 按卷积子图选作用域与生命周期点（不按维数盲改）

- **证据状态**：已核实改造案例（`maptrv2`、`bevfusion`、`bevformer`、`sparse4d`、`lerobot_act`、`dreamzero`、`openwam`；Cosmos3 的 HIP meta 权重转换已做双侧核查，属受限参考）
- **待排查风险**：无差别对整模型调用 `model.to(memory_format=torch.channels_last)` 误伤非图像 4D 参数（如注意力/投影张量），或在每一层 forward 内反复转换权重/激活产生多余拷贝；混合 3D+2D 模型只用单一格式导致另一维卷积落回默认布局。
- **After 手法**：
  - **从小范围作用域起步**：优先只转卷积子模块（如 `img_backbone`、`backbone`、`encoder_img_feat_input_proj`），或提供可配置作用域（如 `scope in {"backbone", "model"}`）；若仅想限定卷积权重，评估 `torch.nn.utils.convert_conv2d_weight_memory_format` / `convert_conv3d_weight_memory_format`（或按 `isinstance(m, (nn.Conv3d, nn.ConvTranspose3d))` 与 `nn.Conv2d` 分别转 `channels_last_3d` / `channels_last`）。
  - **精确标记入口**：当模块存在多个无关 `Conv2d` 时，仅给目标子图首层打标记或装 `forward_pre_hook`（如 `LeRobot-ACT` 首层 `_CONV_MARKER`、`MapTRv2` `img_backbone` pre-hook）。选择一次性权重转换的生命周期点。普通模块可在加载后、创建优化器和分布式包装前处理，meta 初始化则核对后续实化是否保留 stride。检查最终参数引用与分布式顺序。
- **联动核对**：权重与激活入口可能读取不同开关。BEVFusion 的权重转换读 recipe 选项或环境变量，激活转换直接读 `MMDET3D_CHANNELS_LAST`，应分别记录实际命中。Sparse4D 的布局与元数据缓存同组，缓存命中依赖可接收 `cached_metadata` 的包装，按组验证和回退。
- **不适用条件**：纯 Transformer / 无卷积热点模型，或整网卷积占比极低、入口/出口往返拷贝成本高于卷积收益的路径。
- **最小验证**：核对非卷积 4D/5D 参数（token、相机维、索引、元数据）未被转换；确认卷积权重 `stride()` 已变且与优化器/分布式参数引用一致。

## L2 显式转换、退化维歧义与 5D↔4D 边界维护

- **证据状态**：已核实改造案例（`dreamzero`、`openwam`、`sparse4d`、`bevfusion`、`bevformer`、`maptrv2`）
- **待排查风险**：混用逻辑 `permute(0,2,3,1)` 与物理 `memory_format`；在 `view`/`reshape` 合并相机维或 `einops.rearrange('b c t h w -> (b t) c h w')` 切换 5D↔4D 时丢失 channels-last 布局或触发 `view` stride 报错；仅凭 `is_contiguous(memory_format=...)` 在退化维度（`N=1`/`C=1`/`H=W=1`）下误判后端已走快速路径。
- **After 手法**：
  - **物理布局不变逻辑维**：4D 始终保持 `[N,C,H,W]`（典型 stride `(H*W*C, 1, W*C, C)`），5D 始终保持 `[N,C,D,H,W]`（典型 stride `(D*H*W*C, 1, H*W*C, W*C, C)`）。注意区分同名但含义完全不同的逻辑轴排列（如 `MapTRv2` `bev_pool` 的 `[B,Z,H,W,C]`、`BEVFusion` `depth.py` 的 `permute(0,2,3,1).contiguous()`）。
  - **优先 `to(memory_format=...)`**：对单维退化形状的显式 restride，`x.to(memory_format=...)` 比 `contiguous(memory_format=...)` 更可靠（完全退化形状仍有歧义，不可单凭两个 `is_contiguous` 布尔值断定后端算法）。
  - **维护维度切换边界**：在合并 `(B, cams, C, H, W) -> (B*cams, C, H, W)` 之后、`grid_mask` 之后进入 `img_backbone` 之前转换；在视频 VAE 的 `rearrange('b c t h w -> (b t) c h w')` 进入 2D `resample`/`AttentionBlock` 前显式转为 `channels_last`，返回 5D 进 3D 卷积前恢复 `channels_last_3d`；输出送给依赖连续 NCHW/NCDHW 的外部扩展时，仅在该出口 `.contiguous()` 并计入开销。不盲目把 `view` 改成 `reshape` 而忽略隐式拷贝与 alias 变化。
- **不适用条件**：不需要进入卷积子图的中间特征不要顺手转换，避免放大跨算子拷贝。
- **最小验证**：覆盖单维为 1（`N=1`/`T=1`/`H=1`）、动态非方形与非连续输入；在每条卷积与维度切换边界打印 `shape`、`stride()` 及对应 `is_contiguous`，并用 profiler 确认无意外隐式复制。

## L3 3D 卷积前反向非对称布局（前向 `channels_last_3d` + 反向用连续 NCDHW 输入计算权重梯度）

- **证据状态**：受限单模型参考，已核实 LingBot-VLA-V2 与 Transformers 基线的 Before/After。
- **Before**：Qwen3-VL 的 patch embedding reshape、转换 dtype 后直接调用 `self.proj`。若怀疑目标硬件的 3D 布局反向退化，先测前向、权重梯度以及转换开销。
- **After 手法**：用自定义 `torch.autograd.Function` 分离前反向布局。该案例的前向 helper 使用 `@torch.compiler.disable`，每次调用按需将输入和权重转为 `channels_last_3d` 后执行 `F.conv3d`。反向显式检查 `ctx.needs_input_grad[0]`（若该路径设计为首层无输入梯度则直接抛 `RuntimeError` 防误用），并对 `hidden_states.contiguous()` 与 `grad_output.contiguous()` 调用 `torch.nn.grad.conv3d_weight(...)` 在连续 NCDHW 上算权重梯度。
- **不适用条件**：需要输入梯度、高阶梯度或超出该 patch embedding 的卷积契约。若要扩展，额外处理并验证 padding、dilation、groups 和对应梯度。原生路径达到性能目标时保留原生路径。
- **最小验证**：对比开启前后前向输出、`weight.grad` 与 `bias.grad` 在容差内一致；构造 `requires_grad=True` 输入验证按契约明确报错（或正确返回 `grad_input`）；实测前向与反向各自耗时确认非对称路径净收益为正。

## L4 优化器状态与 Checkpoint 布局兼容（保存前连续化清洗）

- **证据状态**：已核实改造案例（`lerobot_act` `channels_last.py`）
- **待排查风险**：如果优化器按参数布局初始化动量状态，卷积权重原地转为 `channels_last` 后，`exp_avg`/`exp_avg_sq` 等状态可能在默认连续格式下检查失败；直接保存 `optimizer.state_dict()` 时，核对序列化器是否接受这类状态，以及双向恢复后是否还能维持预期布局。ACT 基线将展平的状态交给 Safetensors，After 在保存边界连续化。
- **After 手法**：在 checkpoint 保存入口包装优化器状态导出逻辑——递归遍历 `optimizer.state_dict()`（保留 `param_groups` 不动），对所有状态张量执行 `.contiguous()` 后再保存；加载时核对与标准基线 checkpoint 的双向兼容。
- **联动核对**：ACT 原组同时安装构造期标记、首层 Conv2d 输入转换、保存包装和 MaxPool 反向替换。成组启停，并分别记录布局、checkpoint 和算子行为。
- **不适用条件**：纯推理场景，或保存框架本身已保证按连续布局序列化且经双向往返实测通过。
- **最小验证**：执行"channels-last 训练保存 → 关闭开关加载续训"与"标准基线保存 → 开启 channels-last 加载续训"双向往返。核对参数和优化器状态符合各路径的预期 stride，状态值和后续 step loss 满足容差。

## L5 自定义算子、融合算子与时序缓存的布局一致性契约

- **证据状态**：已核实 OpenWAM 的缓存改造和 MapTRv2 的轴语义。公共算子布局分支及 Cosmos3 融合路径为实现参考，基线待核实。
- **待排查风险**：激活和关联张量布局不匹配可能触发重排、参考回退或显式报错。按目标算子的实际分支判断。
- **After 手法**：
  - 逐个核对自定义/融合算子的布局契约（如 DCN wrapper 对齐 input/offset/mask/weight；GridSampler 的 NHWC 前向需 input/output 为 ChannelsLast 且使用 bilinear，反向需 input、grad_output、grad_input 为 ChannelsLast；Cosmos3 融合路径按各自契约检查布局、bf16、输入通道和梯度），核对所有关联张量。资格不满足时按原契约回退或显式报错。
  - 时序缓存按消费端的布局契约保存。OpenWAM 和 Cosmos3 用 `clone(memory_format=torch.channels_last_3d)`，再用真实退化 shape 验证 stride。
  - **正确性保护（非提速）**：核对参考路径保留的输出布局和数值。ACT 未标记的 pool 走原实现，已标记但布局、设备或形状超出契约时显式报错。Cosmos3 的融合资格失败走对应参考实现，契约检查失败可能先抛错。
- **不适用条件**：仅支持连续 NCHW/NCDHW 的第三方扩展——不要强塞 channels-last 张量，应在边界显式 `.contiguous()`。
- **最小验证**：用 profiler 或调试日志核实自定义算子的实际分支。缓存跨 chunk 复用及 `N=1`/`T=1` 时，按缓存和激活各自的 shape 检查 stride 是否满足同一内存格式契约。

## L6 与 `torch.compile` 的交互：编译边界、Stride Guards 与图复用

- **证据状态**：已核实改造案例（BEVFusion、BEVFormer）。OpenWAM 的私有 API 与图复用属于版本敏感的受限参考。DreamZero 的 DiT 案例同时改动 checkpointing，布局收益须另行消融。
- **待排查风险**：假定 eager 里的 `.contiguous(channels_last)` 在 `torch.compile` 融合图里一定保留；把转换随意放进编译区内引发额外 stride guards 与重编译；或在 `T=1`/`N=1` 等单元维下因等价 stride 不同导致重复编译多份图。
- **After 手法**：
  - 分别验证 eager 与实际使用的 `torch.compile`：明确布局转换位于编译区外还是区内（如 `BEVFusion` 放区外，`BEVFormer` 在区内），观察 guards、重编译次数与图断。
  - **版本依赖受限参考（非通用默认）**：若确证融合图丢失目标布局或在单元维重复建图，可按目标 PyTorch 版本评估 `OpenWAM` 的做法——编译态用 `torch._inductor.inductor_prims.force_stride_order`（启动时必须先校验符号存在性，该案例校验失败时显式报错），另在开了 `vae_pointwise_compile` 的设备路径上，仅针对通道归约的 5D channels-last 张量用 `as_strided` 规范单元维等价 stride，并配合 `mark_static` 复用图。基线已有独立 VAE 快路径，核对实际调用入口后再比较。
  - **边界分流**：全局 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION`、编译范围与重编译治理移交 `opt-compile`，做消融时单独开关。
- **不适用条件**：无编译需求的项目，或未经目标 PyTorch 版本验证的私有 Inductor primitive。
- **最小验证**：对比编译前后 `TORCH_LOGS=guards,recompiles` 与 Trace 中的实际卷积 kernel，确认无意外重编译风暴且布局生效。

## L7 ROCm / MIOpen 布局建议开关（`PYTORCH_MIOPEN_SUGGEST_NHWC` / `NDHWC`）

- **证据状态**：版本敏感的配置观察。模型级 `runtime*.yaml` 中有 11 个模型设置布局变量，Cosmos3 的启动文档另有设置。两处都只证明配置存在。上游 PyTorch v2.10.0 的实现需与目标构建分别核对。
- **待排查风险**：以为只设 `PYTORCH_MIOPEN_SUGGEST_NHWC=1` 就会自动转换模型与输入；或误以为未设置（unset）必然等于关闭；或直接照抄 `PYTORCH_MIOPEN_SUGGEST_NDHWC=1` 却未核对目标 PyTorch 构建是否读取该变量；或在消融中把布局变量与 `MIOPEN_FIND_MODE=1`、TF32 混在一起改。
- **After 手法**：
  - 把 `PYTORCH_MIOPEN_SUGGEST_NHWC`/`NDHWC` 视为**版本与构建相关的后端布局提示**——它不替代输入/权重转换，也不单独保证提速；容器训练时必须设在目标容器启动环境内。
  - **核对目标构建行为**：上游 PyTorch 2.10（`ATen/native/ConvUtils.h`）中 `PYTORCH_MIOPEN_SUGGEST_NHWC` 同时参与 4D 与 5D 卷积选择且默认值取决于 MIOpen 版本。该函数没有读取单独的 `PYTORCH_MIOPEN_SUGGEST_NDHWC`，厂商/DTK 分支可能不同。必须按实际安装版本核对，并以独立进程对照 `unset` / `0` / `1`。
  - **变量隔离边界**：`MIOPEN_FIND_MODE`（搜索策略与冷启动成本）、`TORCHINDUCTOR_LAYOUT_OPTIMIZATION` 与 TF32/GEMM 精度开关不属于本模式主责，消融时必须独立隔离，不将搜索策略或精度变化归因于 channels-last。
- **不适用条件**：CUDA 后端（不读取 MIOpen 变量）；仅靠设置环境变量而不转换输入/权重的场景。
- **最小验证**：在相同模型/数据/FindDb 状态下，以独立进程顺序消融「原始基线 → 仅代码 channels-last 转换 → 代码转换 + `SUGGEST_NHWC/NDHWC`」，记录原有环境变量值、首次启动耗时与稳态耗时。

---

## 不适用的场景汇总

- **非卷积热点模型**：纯注意力/MLP 模型或卷积占 step 耗时极低的负载，转换开销易大于收益。
- **逻辑维度改写需求**：需要改变张量逻辑轴顺序（如外部 API 要求 `[N, H, W, C]`）属于接口/数据格式改写，不是 `memory_format` 优化。
- **全局编译布局与后端搜索调优**：`TORCHINDUCTOR_LAYOUT_OPTIMIZATION`、`MIOPEN_FIND_MODE`、TF32/GEMM 开关分别移交 `opt-compile` 或运行时/后端调优侧。
