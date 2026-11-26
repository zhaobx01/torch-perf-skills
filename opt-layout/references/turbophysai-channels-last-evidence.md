# TurboPhysAI Channels-Last 跨模型证据

- **来源**：`zhaobx01/TurboPhysAI` 的 `corpus/all-models`，聚合官方 `main`（`1eb24269da`）及 13 个 PR。
- **本次核查快照 SHA**：`796d3f6bce998d91af9d336f506a769b2bdde927`。索引按该提交核查，源码链接打开聚合分支的对应文件。分支更新后可用核查 SHA 和符号复核。
- **范围和证据边界**：16 个模型中有 8 个模型目录及 Cosmos3 补丁包含显式布局转换。另抽查 OpenVLA 视觉路径作为负例。结论来自静态源码，设备数值、显存和性能未验证。源码注释中的性能原因或数字标为「来源自述，未复测」。方法和检查流程可离线使用。

## 案例索引

Before SHA 指语料库实际检出的基线。After 列各行共用上述核查快照 SHA。`main` 表示证据来自官方基准提交。

| 机制 | 模型 | 证据状态 | Before SHA | After 源码 | 来源 | 关键符号 |
| --- | --- | --- | --- | --- | --- | --- |
| 2D 转换作用域 | MapTRv2 | 已核实改造案例 | `e03f097abe` | [maptrv2/training.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/maptrv2/training.py) | #45 | `_convert_to_channels_last`、`_install_input_layout_hook` |
| 2D 编译区内转换 | BEVFormer | 已核实改造案例 | `66b65f3a1f` | [bevformer/backbone.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevformer/backbone.py)、[bevformer/training.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevformer/training.py) | `main` | `extract_img_feat`、`compiled_extract_img_feat` |
| 2D 编译区外转换 | BEVFusion | 已核实改造案例 | `326653dc06` | [bevfusion/backbone.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevfusion/backbone.py)、[bevfusion/training.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevfusion/training.py) | `main` | `extract_camera_features`、`_convert_conv2d_weights_to_channels_last` |
| 2D 输入与打包缓存 | Sparse4D | 已核实改造案例 | `249ffbb695` | [sparse4d/channels_last.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/sparse4d/channels_last.py)、[sparse4d/catalog.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/sparse4d/catalog.py) | #43 | `extract_feat`、`feature_maps_format_wrapper` |
| 首层标记、保存与 pool | LeRobot-ACT | 已核实改造案例 | `3f2179f3b6` | [lerobot_act/channels_last.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lerobot_act/channels_last.py)、[lerobot_act/pool_backward_gather.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lerobot_act/pool_backward_gather.py)、[lerobot_act/catalog.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lerobot_act/catalog.py) | #44 | `channels_last_init_wrapper`、`contiguous_optimizer_state_wrapper`、`max_pool_backward_gather_wrapper` |
| 3D 前反向布局分离 | LingBot-VLA-V2 | 已核实改造案例，受限单模型参考 | `47b0e478f3`（Transformers） | [lingbot_vla_v2/conv3d_channel_last.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/lingbot_vla_v2/conv3d_channel_last.py) | #40 | `Qwen3VLVisionPatchEmbed.forward`、`_ChannelLastConv3d` |
| 5D/4D VAE 与 DiT | DreamZero | 已核实改造案例，DiT 为组合改动 | `ab790c198f` | [dreamzero/wan_video_vae.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/dreamzero/wan_video_vae.py)、[dreamzero/CausalWanModel.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/dreamzero/CausalWanModel.py)、[dreamzero/catalog.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/dreamzero/catalog.py) | #42 | `init_channels_last`、`resample_forward`、`attention_block_forward`、`forward_train`、`init_weights` |
| VAE 缓存、编译约束 | OpenWAM | 已核实缓存对照，私有编译路径为受限参考 | `0c9ed80d80` | [openwam/ports/vae.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/ports/vae.py)、[openwam/__init__.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/__init__.py)、[openwam/vae_layout.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/vae_layout.py)、[openwam/pointwise.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/pointwise.py) | #27 | `_clone_cache`、`prepare_model`、`force_stride_order`、`vae_normalize` |
| HIP 构造期权重转换 | Cosmos3 | 已核实改造案例，受限参考 | `9726697a83` | [cosmos-framework.patch](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/model_examples/Cosmos3/cosmos-framework.patch) | #19 | `OmniMoTModel.build_net`、`apply_channels_last_conv_weights` |
| 融合 3D 卷积和时序缓存 | Cosmos3 | 实现参考，基线待核实 | `9726697a83` | [cosmos-framework.patch](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/model_examples/Cosmos3/cosmos-framework.patch) | #19 | `fused_conv3d_bias_ndhwc`、`fused_causal_cache_pad_conv3d_ndhwc` |
| 自定义算子布局分支 | BEVFormer / 公共算子 | 算子契约的静态实现参考，基线待核实 | `66b65f3a1f`（BEVFormer） | [bevformer/mdc.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevformer/mdc.py)、[modulated_deform_conv.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/operators/modulated_deform_conv.py)、[GridSampler.cu](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/kernel/grid_sample/GridSampler.cu) | `main` | `_hcu_forward`、`_hcu_backward`、`modulated_deform_conv2d`、`use_nhwc_kernel`、`use_nhwc_backward` |
| 逻辑轴顺序对照 | MapTRv2 / BEVFusion | 已核实轴语义 | `e03f097abe` / `326653dc06` | [maptrv2/bev_pool_fix.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/maptrv2/bev_pool_fix.py)、[bevfusion/depth.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/bevfusion/depth.py) | #45 / `main` | `normalize_bev_pool_layout`、`depth_lss_get_cam_feats` |
| 视觉路径负例 | OpenVLA | 负例对照 | `c8f03f48af` | [openvla/vision_timm.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openvla/vision_timm.py) | `main` | `dynamo_safe_intermediate_layers` 调用 `patch_embed` |

## 按机制核查

### 2D 骨干转换与作用域

- **MapTRv2**。基线 `extract_img_feat` 展平相机维后进入 backbone。After 增加 `model` / `backbone` 两种权重作用域，默认 `model`，非法作用域报错。训练入口启用布局且有 `img_backbone` 时安装 pre-hook，首输入为 4D Tensor 且未通过布局检查才转换。日志中的卷积计数遍历整个模型，不能据此确认所有目标权重的 stride。
- **BEVFormer 与 BEVFusion**。基线相机入口未显式转换。BEVFormer 在 `extract_img_feat` 中转换，启用 wrapper 编译时位于编译区内，此外训练入口还可转整模型。BEVFusion 仅转 Conv2d 权重，激活在相机维展平后、进入编译的 backbone 前转换。权重读 recipe 选项或环境变量，激活单独读 `MMDET3D_CHANNELS_LAST`，需分别验证命中。
- **Sparse4D**。基线 `extract_feat` 在 deformable 路径调用原 `feature_maps_format`。After 转换 backbone，激活在相机展平和 GridMask 后转换。元数据缓存按设备、相机数及各级空间尺寸索引。命中时传入新增的 `cached_metadata`，因此 `extract_feat` 与打包包装要同组启停。缓存未命中、无元数据或逆向打包时走原打包路径。

### LeRobot-ACT 的布局、状态保存和 pool

- 构造时转换 `backbone` 与 `encoder_img_feat_input_proj`，只给 backbone 的首个 Conv2d 打标记，Conv2d wrapper 仅转换该层输入。基线 `_save_single_optimizer_state` 展平状态后交给 Safetensors，After 先递归连续化状态张量，保留 `param_groups`。
- `lerobot_act.channels_last_pool` 同组注册构造标记、输入转换、保存包装及 MaxPool 替换。未标记 pool 走原实现。已标记 pool 要求 4D、非空 batch、HCU/CUDA、元素数小于 `2**31` 且输入为 channels-last，超出契约显式报错。启停和回退以组为单位，checkpoint 双向恢复和 pool 数值分别验证。

### LingBot-VLA-V2 的 3D 前反向布局分离

- 基线 `Qwen3VLVisionPatchEmbed.forward` 在 reshape 与转换 dtype 后直接调用 `self.proj`。After 通过禁用编译的 helper 转换输入和权重后调用 `F.conv3d`，反向使用连续 NCDHW 的输入和上游梯度调用 `torch.nn.grad.conv3d_weight`，有 bias 时单独归约。需要输入梯度时显式报错。权重梯度内核是否退化、分离后是否提速均需在目标环境单独测量。

### 混合 2D/3D VAE、缓存与编译约束

- **DreamZero**。VAE 构造按卷积维度转换权重，`Resample` 和 `AttentionBlock` 在 5D→4D 后转 channels-last，编码入口转换 5D 输入。DiT 的 `dreamzero.dit.channels_last_gradient_checkpointing` 同时转换 patch embedding 权重和训练输入，并移除基线的训练 checkpointing 分支。整组结果是组合效果，单项布局归因需另做消融。
- **OpenWAM**。固定基线的 `openwam/model/kernels/vae_fast.py` 已有 2D/3D channels-last，基线原生 `wan/models/vae.py` 用普通 `clone()` 取时序缓存。After 的 native VAE 路径增加权重转换、缓存格式和编译约束。[openwam/settings.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/settings.py) 中 `vae_channels_last` 默认关，由具体 recipe 决定是否覆盖。启用时在启动路径校验私有 `force_stride_order`，缺失显式报错。编译态应用该 stride 约束，关于普通连续化可能被融合消除的说明为「来源自述，未复测」。经 `vae_pointwise_compile` 的设备路径中，仅在 5D channels-last 且按通道归约时用 `as_strided` 规范单元维等价 stride，结合 `mark_static` 复用图。
- **Cosmos3 构造期布局**。在固定基线上核对 `OmniMoTModel.build_net` 后，确认补丁新增了 meta 构造阶段、FSDP 前的卷积权重转换，受 HIP 构建条件约束。补丁在该基线上通过 `git apply --check`。
- **Cosmos3 融合入口（实现参考，基线待核实）**。`fused_conv3d_bias_ndhwc` 处理无缓存 chunk。开启融合、bf16、输入和权重均为 channels-last-3d、`C_in ∈ {160, 320, 640}` 且相关张量无需梯度时，规范化 NDHWC stride 后调自定义算子。资格不足走参考卷积，仅在输入满足布局检查时保持参考输出布局。12 通道 stem 被排除，注释给出的性能原因是「来源自述，未复测」。
- **Cosmos3 缓存入口（实现参考，基线待核实）**。`fused_causal_cache_pad_conv3d_ndhwc` 另外检查缓存（如存在）的布局、关联 dtype 和 `groups == 1`。符合基本资格但关闭 concat 融合时，先拼接再调用无缓存入口。资格不足走参考路径。缓存使用 `clone(memory_format=torch.channels_last_3d)` 刷新。两条入口都可能在前置契约校验处先报错。

### 算子分支、轴语义与后端提示

- **DCN 与 GridSampler**。DCN wrapper 对齐 input、offset、mask 和 weight 格式。BEVFormer hipDNN DCN 返回 NHWC buffer 的 NCHW 逻辑视图，权重梯度路径显式转换上游梯度。GridSampler 的 NHWC 前向要求 input 和 output 为 ChannelsLast 且使用 bilinear；NHWC 反向要求 input、grad_output 和 grad_input 为 ChannelsLast。记录实际分支与额外格式转换。
- **逻辑轴变换**。MapTRv2 的 BEV 池化 `channels_last` 参数描述 `[B,Z,H,W,C]` 轴顺序，BEVFusion `depth.py` 的 `permute(0,2,3,1).contiguous()` 也改变逻辑轴。两处可作扫描器命名反例。
- **ROCm 配置观察**。指定快照的模型级 `runtime*.yaml` 中，BEVFormer、BEVFusion、DreamZero、FlashOCC、GR00T、LeRobot-ACT、MapTRv2、OpenPI、OpenWAM、PointPillars、Sparse4D 共 11 个模型设置 NHWC 或 NDHWC 提示。Cosmos3 的 [启动示例](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/model_examples/Cosmos3/README.md) 单列。[openwam/launch.py](https://github.com/zhaobx01/TurboPhysAI/blob/corpus/all-models/turbo_physai/optimizations/models/openwam/launch.py) 另通过 `setdefault` 填默认值。PyTorch v2.10.0 上游 `ATen/native/ConvUtils.h` 的 MIOpen 布局函数使用 `PYTORCH_MIOPEN_SUGGEST_NHWC` 判断 4D/5D。厂商构建是否读取单独的 NDHWC 提示需按目标版本核对。

## 已搜但未发现与归属说明

- 本次在 OpenVLA、GR00T、OpenPI、DiT4DiT、LingBot-VA、FlashOCC、PointPillars 的优化模型目录中未找到显式 `torch.channels_last` / `torch.channels_last_3d` 转换。OpenVLA 的负例限于核对过的视觉 patch embedding 路径。已有后端提示的模型仍需检查实际 tensor 布局。
- 本次未核实 AMP 与 channels-last 的专门补丁，未执行设备路径验证。全局 Inductor 布局、MIOpen 搜索策略和精度配置是独立的消融变量。
