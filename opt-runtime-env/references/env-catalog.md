# 五类后端启动变量速查

取值列以 [TurboPhysAI 语料](turbophysai-env-evidence.md) 的值为例。按 [选型模式](patterns.md) 核对瓶颈与具体构建，核实变量的读取时机、默认值、支持范围和覆盖顺序。回退统一恢复已记录的值或未设置状态，以新进程复验；有缓存或其他依赖时按 [验收规范](validation.md) 成组恢复。

## R1 精度与数学库

| 变量 | 语料值 | 作用、前提与注意点 |
| --- | --- | --- |
| `HIPBLASLT_ALLOW_TF32` | `1` / `0` | 允许或限制匹配构建中 hipBLASLt 的 FP32 GEMM 使用 TF32，核对实际数学路径和任务容差。 |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `1` | OpenWAM 的 hipBLASLt 偏好配置，需确认目标构建识别该名称，并核对实际分派。可与 Lt 数学策略一起对照。 |
| `ROCBLAS_MATH_MODE` | `1` | 核对目标 rocBLAS/DTK 的枚举与适用算子，记录内部数学和累加模式。 |
| `MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32` | `1` | 语料中的 MIOpen TF32 卷积配置，核对目标构建、形状、算法及 accuracy 模式覆盖。 |
| `ENABLE_MIOPEN_DEFORMER_CONV_FP16` | `TRUE` | 构建特定的卷积路径开关。仅在目标 MIOpen/DTK 已确认支持时评估对应精度及实际分派。 |
| `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` | `1` | 目标 PyTorch 构建中的 BLAS TF32 覆盖。核对它与代码精度 API、后端及敏感层保护的优先级。 |
| `TRITON_F32_DEFAULT` | `tf32` | 在支持的构建中影响未显式指定精度的 FP32 dot，关联实际编译选项与缓存作用域。 |
| `DISABLE_ADDMM_CUDA_LT` | `1` | 构建支持时限制 `addmm` 的 Lt 路径。确认实际回退路径，并查启动配置是否用于规避原故障。 |
| `TORCHINDUCTOR_EMULATE_PRECISION_CASTS` | `0` | 部分构建中控制 Inductor 保留低精度舍入的中间 cast。v2.10 上游源码默认读取 `0`，目标构建可能不同。按实际基线与生成代码验收。 |

## R2 卷积搜索与布局

| 变量 | 语料值 | 作用、前提与注意点 |
| --- | --- | --- |
| `MIOPEN_FIND_MODE` | `1` | 支持该枚举时对应 NORMAL 查找。关注卷积占比、shape 变化、已有数据库和运行周期。 |
| `MIOPEN_FIND_ENFORCE` | `SEARCH` | 目标库可在 API 未请求时尝试调优；已有 PerfDb 记录仍可跳过，部分 API 策略优先。分别看首次和重复搜索。 |
| `PYTORCH_MIOPEN_SUGGEST_NHWC` / `PYTORCH_MIOPEN_SUGGEST_NDHWC` | `1` | 目标构建设定对应 2D/3D 卷积布局提示。用实际输入/权重 stride、卷积路径和转换成本取证。 |

## R3 硬件队列、PCIe 与显存分配器

| 变量 | 语料值 | 作用、前提与注意点 |
| --- | --- | --- |
| `GPU_MAX_HW_QUEUES` | `16` | 核对 HIP 每进程每设备可创建的硬件队列上限及例外。这个值来自受限配方，各构建的支持值与效果需要实测。 |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1` | 与 PCIe peer access 条件相关，核对 GPU 间互联、P2P 能力与 BAR 条件，验证实际传输路径。 |
| `HIP_BUFFER_EXTRA_SIZE` | `0` | 厂商 HIP/DTK 构建中的缓冲预留配置参考，核对单位、实际显存占用与路径稳定性。 |
| `HIPBLASLT_WORKSPACE_SIZE` | `32768` | OpenWAM `configure_zero` 中的 hipBLASLt workspace 条件默认值参考（源码注释单位为 KiB，约 32 MiB）；启用前核对目标构建的单位、显存余量与覆盖顺序。 |
| `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 新版公开接口使用前者，后者为兼容别名；以目标版本、实际 allocator backend 和读取优先级为准。检查碎片、重试和 Graph 兼容。 |

## R4 多卡通信与 CPU 线程

| 变量 | 语料值 | 作用、前提与注意点 |
| --- | --- | --- |
| `NCCL_ALGO` | `Ring` | 目标 NCCL/RCCL 构建的集合算法筛选。对照消息大小与实际拓扑，确认初始化和稳态算法。 |
| `NCCL_RINGS` | 显式环路串 | MapTRv2 的 8 卡强拓扑绑定，核对构建是否支持与物理互联、rank 映射是否一致。 |
| `NCCL_TOPO_FILE` | 字面串 `"null"` | MapTRv2 的构建参考。公开 NCCL 接口读取拓扑 XML 路径。特殊值的语义由目标构建确认。 |
| `ALLREDUCE_STREAM_WITH_COMPUTE` | `1` | MapTRv2 的构建特定重叠选项；核对实际 stream 顺序、集合完成和多 rank 退出。 |
| `TORCH_NCCL_ENABLE_MONITORING` | `0` | 控制 ProcessGroupNCCL 的监控线程，影响 watchdog 失去心跳后的自动终止。Flight Recorder 的记录能力另查 `TORCH_NCCL_TRACE_BUFFER_SIZE`。 |
| `TOKENIZERS_PARALLELISM` | `false` | 分词库并行策略，结合 worker 启动方式、抢核与实际在线分词负载对照。 |
| `OMP_NUM_THREADS` | `1` | OpenMP 线程数控制。按每节点可用 CPU、rank/worker 分配及绑核情况比较。 |

## R5 TorchInductor / Triton 编译与缓存

| 变量 | 语料值 | 作用、前提与注意点 |
| --- | --- | --- |
| `TORCHINDUCTOR_LAYOUT_OPTIMIZATION` | `1` | 目标 Inductor 构建的布局策略，查编译后 stride、守卫及实际转换成本。 |
| `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING` | `1` / `0` | 受构建约束的参数调优开关。OpenWAM 不同配方采用不同值，单独测量首次编译、缓存复用和稳态收益。 |
| `TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS` / `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` | `ATEN` | 在适用的 Inductor autotune 路径限制候选后端。核对实际候选列表及生成产物。 |
| `TORCHINDUCTOR_COMPILE_THREADS` | `1` | 编译并行度，结合多 rank、CPU 分配和主机内存验证首编代价。 |
| `TORCHINDUCTOR_ALIGNMENT_ASSERTS` | `0` | OpenPI 显式关闭对齐断言的构建参考。确认目标版本支持，并取证具体断言、输入契约及关闭的影响。 |
| `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` | 独立目录，包括 `/dev/shm` | 检查实际缓存命中、目录容量、共享环境和并发写入。 |
| `TORCH_COMPILE_CACHE_KEY_TAG` | 语料中的 `flex-attn-tf32` | 支持此开关的版本可用标签区分编译产物。与影响生成数学行为的开关成组验证，并复验基线缓存。 |
