# 让代码可被追踪，以及包装顺序与 key 兼容

本文收录 TurboPhysAI 各分支里与 `torch.compile` 直接相关的两类模式补充（Dynamo-safe 改写、FSDP1 包装顺序与 checkpoint key 兼容）。全部来自静态阅读源码，没有运行模型、训练或性能测试，不含任何加速数据；迁移前须在目标项目重新验证。16 模型全量固定提交索引见 [turbophysai-compile-evidence.md](turbophysai-compile-evidence.md)。

## A. 让周围代码可被追踪

### A1. 外部 kernel 注册为 `custom_op`（交接说明）

自定义 C++/HIP/Triton 算子若因缺少 fake/autograd 元信息在 `torch.compile` 区域内引发图中断，其 `torch.library.custom_op` + `register_fake` + `register_autograd` 封装契约与跨模型案例（BEVFormer `mdc.py`/`msda.py`、Cosmos3 `wan_vae_conv_bias`、LingBot-VLA-V2 `video_fa2.py`、LeRobot-ACT `qk_tf32.py`）已统一归入 **`optimize-kernel-replacement`（K6）** 维护；本 Skill 仅决定该算子应保留在图内（走 K6）还是局部 eager 隔离（走 P4）。

### A2. 为让周围代码进图而改造 eager 岛与 Dynamo 障碍

- **MapTRv2 `grid_mask.py`**：官方实现用 `np.asarray(PIL)` 得到只读数组再 `torch.from_numpy(...).cuda()`。源码注释（来源自述，未复测）称只读数组会让周围编译区域触发 graph break 或硬失败，且设备硬编码 CUDA。替换版拷贝出可写缓冲区、跟随输入设备，并加 `torch._dynamo.disable`，使该函数留在图外同时不破坏周围编译区域。
- **OpenVLA `vision_timm.py`**：timm `get_intermediate_layers` 对已有 `set` 再做 `set(n)`，在 torch 2.7.1 触发 Dynamo `assert source is None`；改为基于 `list` 的等价重写。版本相关，升级 PyTorch 后需重新确认。
- **OpenVLA `compile_fsdp1.py`**：编译开启时 `MixedPrecision` 设 `buffer_dtype=None`，源码注释（来源自述，未复测）称用于避开 Dynamo fake tensor `set_` dtype 报错。

## C. 包装顺序与 checkpoint key 兼容（OpenVLA FSDP1 路径）

- **逐层编译与 checkpoint 位置**：模块 docstring（来源自述，未复测）称基线 `apply_activation_checkpointing` 会把 checkpoint HOP 放进编译单元内部，因此改为按 `FSDP(layer) → torch.compile → checkpoint_wrapper(compiled)` 的顺序对 FSDP1 包装的 LLM 层逐层编译，而视觉塔内部无 checkpoint HOP 故整塔编译。
- **运行时解析 FSDP 类**：编译分支用 `_fsdp_class()` 在调用时动态查找 `FullyShardedDataParallel`，避免模块级绑定到未打 `openvla.fsdp.prefetch` 补丁的旧类而静默丢失预取配置。
- **保存键名清洗**：逐层编译后 `state_dict` 键含 `._orig_mod.`，`save_checkpoint_wrapper` 保存时执行 `strip_orig_mod` 并从 `self.vlm._orig_mod` 取回原模块（加载端对称性未核对）。适用范围限于 FSDP1，不能直接推广到 FSDP2 或其他框架。
