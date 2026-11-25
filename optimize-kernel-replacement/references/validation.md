# 正确性、性能和失败回退

先约定精度与任务指标容差，再动代码。算子替换理论上数值等价，但融合实现的归约顺序、舍入、dtype 提升或 TF32 路径可能不同，必须实测。涉及精度变更（如 FP32 → TF32）的替换不得与数值等价替换混报，必须单独确认容差与训练稳定性。

## 数值正确性与契约覆盖

- **通用要求**：相同权重、输入、随机状态下，比较前向输出、loss、反向梯度；逐位一致或在项目容差内。检查 NaN/Inf、输出结构、dtype 与设备。
- **融合优化器（AdamW / Apex / Muon）**：
  - 相同初始参数与梯度跑多步，比较参数与优化器状态；做"融合保存 → 标准加载续训"与"标准保存 → 融合加载续训"双向 checkpoint 往返。
  - 校验不支持选项、多参数 step 不一致、同组部分 `grad=None` 均按约定报错而非静默走偏；对生成器参数（`model.parameters()`）验证不会因检查被提前耗尽。
  - 非 AdamW（如 Muon）额外覆盖：张量元素数低于和高于融合门槛（如 `_FUSE_MIN_ELEMS`）、单张量 vs multi-tensor 指针表路径、指针表缓存命中与淘汰、以及 FSDP 下含 `grad=None` 参数时所有 rank 同步参与 collective 不死锁。
- **FlexAttention 与注意力后端**：
  - 覆盖变长、等长（dense vs varlen）、空 batch、全 mask、causal 与非 causal、GQA（`repeat > 1`）与等头宽（`repeat == 1`）；**必须比较反向梯度**，不能只看前向（HIP 等平台上的 NaN 常仅在反向暴露）。
  - FlexAttention 专项：验证 `H=1` 共享 `BlockMask` 与原 `H=num_heads` 构造等价；覆盖序列长度能整除与不能整除 `block_size`/tile 的边界；验证 tile 整除 `SPARSE_*_BLOCK_SIZE`；若开启 TF32 精度（如 `flex_attn_tf32`），验证 `torch.compiler.config.cache_key_tag` 已变化、不复用旧 `ieee` 缓存，并单独核验前反向精度与收敛曲线。
  - 架构白名单模板（如 `gfx936` 专属配置）：验证命中白名单与未命中白名单（不同 shape/stride/dtype/架构）两条路径的输出均一致且未命中时退回默认配置。
- **归一化、RoPE 与线性层**：覆盖最后一维刚好等于和小于阈值、非连续输入、不同 dtype、空 batch；复数→实数 RoPE 确认保持原精度（如 `float64`）；带 RoPE 频率缓存的包装确认缓存不进入 `state_dict()` 且变 shape 时有界释放；2D `addmm` 替换重点核对 BF16 舍入敏感层。
- **`custom_op`、定点 Triton 与原生 C++/HIP 扩展**：
  - 在 eager 与 `torch.compile` 下各跑通前向和反向；确认 `register_fake` 返回的 shape/dtype/device 与真实输出完全一致。
  - 仅支持一阶梯度的算子（如 `qk_tf32`、`pool_backward_gather`），在开启高阶求导场景验证明确抛 `RuntimeError`。
  - 原生 C++/HIP 扩展与融合卷积：对白名单内层和显式排除层分别验证；故意传入不支持的 size、通道数、dtype、非连续布局或带梯度张量，确认在 Python 守卫层即回退或抛清晰异常，绝不进入底层 C++ 导致 coredump。
  - 带指针表异步拷贝的 Triton kernel：在启用 CUDA Graph 捕获（如适用）下验证走安全路径。
- **回退与单算子编译隔离路径**：故意让可用性检查失败（卸载库、传不支持 dtype、mock 私有接口不匹配哈希/AST），确认走原实现且有清晰日志；对单算子 `@torch.compiler.disable` 绕开 Inductor 展开的算子，核验图断前后数值一致且未破坏周边状态。

## 分布式与 checkpoint

DDP/FSDP/DeepSpeed 下确认各 rank 走同一条执行路径（可用性检查依赖本地库或环境时，rank 间不一致会引发 collective 死锁或精度漂移）。检查与 CPU offload 等分布式开关的互斥关系。checkpoint 往返后，下一步前反向结果与未替换时一致。

## 性能验收

- **四层结论拆分**：`开关已开启` ≠ `目标 kernel 真实命中` ≠ `数值与功能等价` ≠ `端到端稳态提速`。先用 profiler 确认目标 kernel 被调用、原热点占比下降，再比较端到端 step 时间。
- **统计口径**：多次测量报告中位数与离散程度，分别记录首次编译/图构建与稳态耗时，注明预热步数。对单算子 `@torch.compiler.disable` 绕开 Inductor 展开，需同时核验引入的局部图断未抵消收益。
- **隔离变量**：同一次实验只改一个替换；与 `torch.compile`、布局、同步消除等改动分开消融。
- **诚实交付**：汇报 PyTorch/后端/硬件/厂商库版本、输入 shape、命令、开关、容差、结果与未验证项。缺少目标设备或依赖时完成静态接入与可运行的 CPU/边界单测，明确标为"设备数值与性能未验证"，不把源码注释当作实测结论。

## 回退（按生命周期界定）

- **纯调用期开关**（每次调用实时判定路由、无进程级全局状态残留）：关闭开关即可直接恢复原路径，无需重启进程。
- **涉及进程级状态或构造期绑定**：当改动涉及环境变量/进程级配置、已安装的模块/注册表包装（如 `AttentionInterface`、`OPTIMIZERS`）、已捕获的编译图缓存（尤其涉及 `cache_key_tag` 或 Inductor hook），或已构造的优化器状态时，关闭开关后须**重启进程或重新构建模型与优化器**并按需隔离编译缓存，确保干净对照。
