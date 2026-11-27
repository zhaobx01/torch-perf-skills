# 排行方向类别与下游优化 Skill 的衔接

排行里的“方向类别”只指去向，**本 Skill 不实施优化、不改项目源码**。下表列出每类对应的下游 Skill 与交接前需核对的证据；用户确认要改时再切换到对应专项 Skill。

| 方向类别 | 先核对的证据 | 下游去向与分流规则 |
| --- | --- | --- |
| **launch 开销 / 融合 / 编译** | §3 launch 小间隙占 step%；§7 图断/重编译计数；T1 中 elementwise 占比 | **`opt-compile`**（编译边界 P1–P7、重编译与图中断治理、全局 Inductor 调优） |
| **计算热点**（不进排行，单列） | §2 最大计算类别（attention / optimizer / norm / GEMM / conv）及 top kernel | **`opt-kernel`**（融合优化器、FlexAttention/FA2、RoPE、厂商/定点 kernel 替换）；<br>若热点集中在 Conv2d / Conv3d / 3D VAE，先看 **`opt-layout`** |
| **同步消除** | §4 判定为“GPU 饥饿”的调用点、空闲/CPU% 及代码行 | **`opt-sync`**（`.item()`/`.cpu()`/`nonzero`/布尔掩码索引改写与验证） |
| **数据搬运** | §5 大拷贝的带宽、前置 CPU `copy_`、拷贝前 GPU 空档；§3 头部/搬运等待 | **`opt-dataloader`**（DataLoader `pin_memory`/`persistent_workers`/`spawn`、`non_blocking` 与 copy stream 预取、CPU 采样/预处理向量化及离线编码缓存）；若为步内小张量同步或 D2H 读回则交 **`opt-sync`** |
| **慢 rank 排查** | §1 最慢 rank、§3 最慢 rank 补充归因、§6 集合通信起始极差 | **先于通信重叠排查**：对比该 rank 的数据加载、GC 抖动、设备状态与外部负载；若为单卡数据加载/IO 偏慢交 **`opt-dataloader`**，若为周期性偶发步长尖刺/轮换慢 rank 交 **`opt-memory`** 排查 Python GC 与热路径 `empty_cache()` |
| **通信重叠** | §6 暴露占通信%、重叠率、起始极差；§1 暴露通信 | **`opt-distributed`**（梯度归约重叠、按 owner 归约、AllGather 流水、FSDP/DDP/ZeRO-2 通信策略；起始极差大时先按其分流规则查慢 rank 根因） |
| **显存 / 重算**（不进排行，单列） | 峰值显存 `allocated/reserved` 余量或 OOM、反向重算占比、周期性步长尖刺 | **`opt-memory`**（显存富余时减 AC / 上 SAC 换吞吐；显存紧绷时关训练 KV cache、压激活/碎片；多卡周期抖动时 warmup 后 `gc.freeze()`） |
| **后端运行时环境变量**（零代码侵入，单列） | §2 GEMM/Conv/编译/通信/显存热点，结合运行记录（`command.json`/环境变量）发现显式关闭 TF32、缺失或可疑的后端启动开关（如 `HIPBLASLT_ALLOW_TF32`、`MIOPEN_FIND_MODE`、`PYTORCH_CUDA_ALLOC_CONF`、`NCCL_RINGS`） | **`opt-runtime-env`**（仅改启动配置/环境变量，不改源码；把热点与运行记录中的环境证据一并交接，源码扫描留到下游按需执行） |

## 交接使用规则

- **聚焦 Top 1–2**：先看排行第 1~2 项（及计算热点最大项），不要同时推进所有方向。
- **上限不是收益**：动手前用同口径（无 profiler、充分预热、重复采集）的基线测速确认；优化后再采一份同口径 trace 复算对照。
- **跨报告对比口径**：本 Skill 第一版不含自动 A/B 对比；`summary.json` 带 `schema_version` 与口径元信息，对比前先核对 step 窗口、是否带栈、batch、阈值与环境开关是否一致。
- **带栈 trace 仅用于归因**：带 `with_stack` 的 trace 只用于定位代码行，不得拿它的 step 时间比较性能。
