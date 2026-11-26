# 同步消除模式（P1–P8）

先定位张量设备、调用频率和上下游契约，再选改法。证据标签描述各案例的源码核实程度。运行时命中、设备等价与收益按 [validation.md](validation.md) 验证。来源和原始提交见 [TurboPhysAI 同步证据](turbophysai-sync-evidence.md)。

## P1 标量计算与选择留在设备上

- **Before**。热路径把设备张量转为 Python 标量，只用于下一个张量算子或简单条件选择。也可能经过框架 getter 间接读回，或把读出的标量再构建为设备张量。
- **After**。用零维张量传递值，保留与原路径一致的梯度边界。原 `.item()` 切断标量梯度时，以 `.detach()` 保持该契约。DiT4DiT 的 scheduler sigma 用同设备同 dtype 的张量直接参与计算，先核对 scheduler 的 `sigmas` 实际所在设备。
- **前提与反例**。核对 AMP 下 dtype、标量精度和梯度。`where` 接收已算好的分支值，改写前查清分支副作用、异常和原先跳过的计算。掩蔽位置可能含 NaN/Inf 时使用满足语义的选择，掩码乘法会传播 NaN。即时中止、跳步和 checkpoint 判定中的同步属于「正确性保护（非提速）」。
- **最小验证**。同输入比较输出、dtype 和梯度，覆盖空选择和非有限值，用运行时证据确认目标读回减少。
- **证据状态**。MapTRv2 loss 归一化和布尔 mask 冗余清洗、DiT4DiT scheduler sigma 为「已核实改造案例」。OpenWAM 的私有 `_global_grad_norm` 为「已核实改造案例」「受限/单框架参考」「版本兼容性待核对」。只有确认为 ZeRO-2 已完成裁剪、无 CPU offload 且存在同设备单元素张量时才读私有字段，其他情况走公开 API。

## P2 从源头保留 CPU 元数据，批量转换或从形状推导

- **Before**。CPU 来源的尺寸或长度上卡后在步内被读回。位置编码在 Python 循环零散提取设备标量。布局固定时也可能从设备 mask 做 reduce + `.item()` 计算本可推导的计数。
- **After**。在搬运入口保留符合条件的元数据，其他输入透传原路径。CPU 消费多项小元数据时按字段批量搬运，再构造设备结果。可由原布局证明的计数直接用 shape 和宿主整数计算。Qwen 元数据缓存优先缓存几何、索引和插值权重，保留可训练 embedding 的每步查表。
- **前提与反例**。核对体量、无反向依赖及设备往返成本。缓存键纳入受影响的形状、设备、dtype 和模型配置，限制容量，按 tracing/export 与模型生命周期处理缓存。在构键时仍从设备取值的实现，即使命中也可能有 D2H。在编译区域用局部 eager 边界时与编译专项协作。
- **最小验证**。测 CPU 来源与透传、冷/热/淘汰、图像/视频/纯文本分支以及特殊帧数。对缓存构键也计数和追踪拷贝。
- **证据状态**。Cosmos3 搬运入口、OpenWAM CPU 文本长度、GR00T 和 LingBot-VLA-V2 的不同位置编码做法、DiT4DiT 的 `cond_count` 为「已核实改造案例」。保留版本、硬件和各缓存实现的差异。

## P3 动态形状与冗余去重

- **Before**。布尔抽取或更新、`nonzero` 及 `unique` 可能引入数据相关尺寸和读回。已经唯一且次序满足下游的索引仍执行去重。
- **After**。在结果形状固定的更新中使用有等价语义的 `torch.where` 或算术。必须抽取元素时，评估显式 `nonzero` + `index_select`、或固定容量加有效 mask。前者仍可能在 `nonzero` 同步。只有证明唯一性和排序语义后才删去 `unique()`。
- **前提与反例**。确认 mask dtype、写入语义及梯度。固定容量覆盖溢出处理，还要实测额外计算。任意整数索引和普通设备端选择只按各自调用链判断。
- **最小验证**。全 0、全 1、单元素、空结果与梯度。分别查目标读写和仍保留的动态算子。
- **证据状态**。BEVFusion `QuickCumsum` 的索引与更新改写、PointPillars 删除冗余去重为「已核实改造案例」。

## P4 Attention 封装内的变长元数据与守卫

- **Before → After**。对于已经核对过的封装调用链，可在前提成立时绕开重复的 unpad/变长元数据，或缓存后直传 `cu_seqlens`、`chunk_lengths`。FA2 dense/varlen 选择由主要症状决定主导 Skill，见入口分流。
- **前提与反例**。明确 backend、prefill/decode、cache、padding 方向和 mask 布局。只有调用方已保证 mask 连续、有效 token 和 padding 输出语义均可接受时，首列检查才可作为其中一个守卫。版本或前提无法确认时回到原路径。记录守卫、缓存构键与回退残留的同步。
- **最小验证**。测右 padding、左 padding、含洞 mask、decode/cache、不同后端，并逐支记录运行时命中、输出和梯度。缓存要分开核对冷/热和 tracing 路径。
- **证据状态**。Cosmos3 Qwen 视觉辅助量缓存为「已核实改造案例」，其中 HCU 的 HF 实例包装属于「受限/单模型参考」「版本兼容性待核对」。OpenVLA 的 FA2 prefill 包装属于「实现参考，基线待核实」「版本兼容性待核对」。

## P5 延迟或合并指标读回

- **Before**。非日志微步、非输出 rank 或前向阶段过早把 loss 物化为 Python 数值。多个标量逐项归约和读回，或每步马上等待指标。
- **After**。对确认不会消费的微步跳过物化。前向仅为指标保留 `.detach()` 张量，消费端负责在实际日志边界物化，两端成组启停与回退。多个指标可堆叠后单次归约和 D2H。允许滞后时可用 pinned buffer 和事件在稍后读取，参照 P8。
- **前提与反例**。核对所有消费点和 rank 的键序、dtype 提升、缩放、步号及容差。即时早停、动态调参、best checkpoint 等消费者需原有时序。及时释放未消费指标，收尾按契约执行 `flush()`。
- **最小验证**。梯度累积为 1 和大于 1、非输出 rank、多 rank 归约、最后一笔指标和正常/异常收尾。
- **证据状态**。DiT4DiT、LeRobot-ACT、BEVFusion 和 OpenWAM 对应改写为「已核实改造案例」。ACT 的回传类型与 logger 包装构成强耦合组。OpenWAM side stream 路径依赖异步集合与后端契约。

## P6 缩小显式同步的范围

- **Before**。`dist.barrier()` 在每个累积微步运行，或诊断计时器在训练热路径同步设备。
- **After**。只保留任务确实需要的同步步骤。确认无消费者依赖诊断计时后，可用对应框架版本支持的开关关闭计时或同步。
- **前提与反例**。把同步移入日志分支前，检查分布式一致性、异常传播、buffer 生命周期与 checkpoint 可见性。「正确性保护（非提速）」包括落盘依赖的 staging 确认及跨 rank 失败传播。
- **最小验证**。对照微步和 optimizer step 的集合同步次数，测试多 rank 日志/GC、保存、恢复和异常路径。
- **证据状态**。LingBot-VA `barrier()` 收到 `should_log` 为「已核实改造案例」，日志分支的设备同步保留。OpenWAM 的 DeepSpeed throughput 配置为「受限/单框架参考」「版本兼容性待核对」。

## P7 CPU 求解边界的批量拷贝

- **Before**。CPU SciPy 求解接收完整代价矩阵，匹配的行索引和列索引分别 H2D。
- **After**。代价矩阵仍一次 D2H。把行列索引在主机 `np.stack`，一次 H2D 后在设备上 `unbind`。
- **前提与反例**。索引形状、dtype、次序和空结果必须兼容，矩阵变大时复查 CPU 求解成本。只隔离 SciPy 段到编译图外时，交编译专项处理捕获边界。
- **最小验证**。检查空匹配、并列代价、最终分配和 D2H/H2D 次数。保留正常 CPU 求解路径可回退。
- **证据状态**。BEVFusion 结果 H2D 从两次合为一次，为「已核实改造案例」。MapTRv2 是编译边界交接案例。

## P8 步内非阻塞拷贝的依赖与生命周期

- **Before**。小张量阻塞上传或指标立即同步读回，GPU/CPU 消费者等待时机与复制工作耦合。
- **After**。按目标后端选择 pinned host buffer 和 `non_blocking=True`。异步 D2H 由 CPU 读取时，CPU 必须在读前确认复制完成，例如等待复制结束事件。设备消费者可在消费流等待事件，或使用同流顺序。跨 stream 另外保护 allocator 生命周期，例如对跨流使用的设备张量调用 `record_stream(consumer_stream)`。
- **前提与反例**。保留 host buffer 直至传输完成。H2D 完成前避免覆盖或复用 pinned 源，D2H 完成前避免读或覆盖目标。pageable 内存可能阻塞，显式 pin 也有开销，核查目标构建和实测重叠。跨流等待与分配器生命周期分别验证。batch 级预取交数据管线。
- **最小验证**。核对设备、pinned 状态、事件发生在复制之后、CPU 读取发生在完成之后及正常收尾。检查跨流复用、多 rank 数据一致性和端到端时间。
- **证据状态**。OpenWAM 阻塞指标改为 pinned 延迟读回，为「已核实改造案例」。其小张量 `upload` 和与 `deferred_gather` 联动的侧流属于该实现的受限参考。

## 受限补充

- **矩阵求逆错误检查（`torch.linalg.inv_ex`）**：将 `torch.inverse` 改为 `torch.linalg.inv_ex(..., check_errors=False)` 可避免隐式错误检查同步，但会放弃奇异矩阵自动抛错。仅限已证明上游矩阵可逆或显式处理 `info` 的受限场景，保留错误检查的回退路径属于「正确性保护（非提速）」。详见 [证据文档 §3](turbophysai-sync-evidence.md) 与 [validation.md](validation.md)。
