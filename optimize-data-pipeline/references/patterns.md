# 数据供给与 H2D 改写模式

案例定性按 [TurboPhysAI 证据](turbophysai-data-evidence.md) 中的来源逐项标注。可迁移的实施建议需要在目标版本、数据与训练路径上验证。

## D1 · DataLoader 并发、固定内存与 H2D 配置

**证据**。Sparse4D、DiT4DiT、LeRobot-ACT 为已核实改造案例。FlashOCC 的旧版框架基线待核实。GR00T 传输前降 dtype 为受限单模型参考。

- **Before**。训练路径零 worker 供给不足、批张量未被固定、每个 epoch 重建 worker，或 Accelerate 沿用默认阻塞搬运。零 worker 也可能是主动保留的兼容设置。
- **After**。以真实批大小调整 worker、`persistent_workers`、`prefetch_factor`，独立验证 `pin_memory` 和 `non_blocking`。使用 Accelerate 时保留原有 `DataLoaderConfiguration` 的其他字段。零 worker 时确保 `persistent_workers=False`、`prefetch_factor=None`、`multiprocessing_context=None` 或省略，版本新增参数按目标构造器核对。固定内存可以独立评估。
- **边界**。主机预算按节点计算，关注每节点 rank 数、各 rank worker 数及排队批量。选择进程启动方式前检查底层线程池和 pickle 能力（D3）。若使用 Hugging Face tokenizer，检查多进程与 tokenizer 并行设置，在启动 worker 前配置。对源码版本和容器配置敏感。
- **受限子模式**。若大块输入 H2D 已暴露，且下游原本会转 BF16，可评估由可 pickle 的 collator 仅对目标 CPU 字段提前转换。GR00T 的实现要求 DeepSpeed + BF16、FP16 关闭，且只转换 FP32 `pixel_values`。验收数据值与下游输出容差、H2D 字节和 collate CPU 成本。配方开关关闭时恢复原 collator。
- **最小验证**。确认实际 CPU 张量 `is_pinned()` 和传输调用参数，比较跨 epoch 等待、主机内存及 `/dev/shm`。区分主机侧返回、设备端拷贝和 H2D 与 kernel 的重叠。

## D2 · 自定义 batch 的 pin-memory 适配

**证据**。MapTRv2 的 MMCV 容器和 After 包装已静态核实。原生 pin 线程与递归 API 属于「版本兼容性待核对」，Sparse4D 和 BEV 的实现细节各异。

- **Before**。`pin_memory=True` 已开启，但自定义 batch 容器内部的 CPU Tensor 仍是 pageable。
- **After**。优先为自有类型实现 `pin_memory()`。必须包装框架入口时，遵循原生递归语义，保留 `namedtuple`、自定义 Mapping 和 `cpu_only`，跳过已在加速设备或已固定的 Tensor，确保重复安装幂等。
- **边界**。`cpu_only` 的元数据继续留在 CPU。与非阻塞 H2D 联用时同时检查 loader、collate 和搬运路径。已有包装应先识别再决定接入，避免二次遍历。全局包装的启停按进程生命周期处理。
- **最小验证**。构造混合嵌套容器，验证类型、键、值、CPU-only 字段和可搬运张量的 pin 状态。确认重复安装无额外遍历。

## D3 · 有活跃线程池的 worker 启动与重建

**证据**。OpenVLA 为已核实改造案例，`fork` 与活跃 TF 线程池的冲突来自实现说明，未复测死锁。

- **Before**。OpenVLA 的 RLDS 训练 loader 设 `num_workers=0`，基线注释说明 RLDS 自带并行。
- **After**。只在目标数据集的 loader 上选择 `spawn`。序列化可重建的构造参数，在子进程重建图。先核对环境标记和播种条件，分别验证 RNG、按 worker 分片与 rank 顺序。
- **正确性保护（非提速）**。构造参数、collate 或 worker init 无法 pickle 时保留可用路径。MapTRv2 对含不可 pickle 状态的数据集选择 per-loader `fork`，应先确认底层线程与设备初始化允许 fork。零 worker 可用于维持兼容性。
- **最小验证**。覆盖启动、首批、完整迭代、重复与漏读样本、resume 及干净退出。测量 spawn 启动成本后再决定是否持久化 worker。

## D4 · 独立 copy stream 预取

**证据**。LeRobot-ACT 为已核实改造案例。BEVFormer 的预取几何联动和 BEVFusion 的线程预取见[证据详情](turbophysai-data-evidence.md#2-成组契约与受限路径)。

- **Before**。批量 H2D 仍排在调用方计算流上，实测暴露在 step 头部。
- **After**。在目标设备确认源 Tensor 实际 pinned、传输为 non-blocking，另建 copy stream。逐批记录 Event，消费计算流等待 Event。遍历实际 batch 结构，把设备 Tensor 记录到消费流，或实现等价的分配器生命周期保护。持有不可修改的 pinned 源缓冲直到拷贝完成。
- **成组启停与回退**。预取依赖非阻塞搬运与有效 pinning。按实测输入守卫回到原 loader 路径。若预取额外生产供模型消费的字段，生产、消费及缺字段回退应成组验收。分布式路径保留 RNG 同步、`set_epoch`、`skip_batches` 与 checkpoint 时机。
- **正确性保护（非提速）**。显式结束迭代器，阻止新任务，join 后台线程并传播异常，再完成 outstanding 拷贝后释放缓冲。生成器的 `finally` 需要实际关闭或耗尽才执行。资源不足时保留基线路径。
- **最小验证**。测空、单批、短 epoch、提前 break 且保留 iterator 引用、异常/二次迭代、断点恢复、多流与内存峰值。核实 DMA 与 kernel 在目标设备确实重叠。

## D5 · 按需解码、向量化与连续采样

**证据**。LingBot-VLA-V2、LeRobot-ACT、MapTRv2 均有 Before 与 After 源码。

- **按需解码**。在 HF Dataset 整行读取触发无关列解码时用 `select_columns` 或惰性视图。仅在替代读路径已保证每个被裁剪字段可用时启用。LingBot 的整行视图要求所有相机列同时存在于 `delta_indices` 和数据列中。视图的字段裁剪与查询路径的字段回补需要成组启停。回退时确认相机字段和变换输入完整。
- **采样向量化**。ACT 将基线已有的 `randperm` 加逐索引 `searchsorted` 批量化。其实现拒绝 `shuffle=False`，迁移应守卫在支持路径上，其他配置使用原实现。保留绝对到相对索引映射及种子。
- **连续分片与 resume 成组**。连续迭代和 checkpoint epoch 保护共用启停。ACT 门禁为 `split_batches=False`、`even_batches=True`、`drop_last=False`、非空 epoch，恢复偏移需按全局 batch 对齐且未越界。保留基线尾部补齐、`set_epoch` 与断点状态写入时机。向量化是独立候选。
- **几何**。MapTRv2 仅替换 `shift_fixed_num_sampled_points_v2` 的 Shapely 逐点插值。迁移时核对端点、零长度、重复顶点、随机 shift 选择、裁剪与 padding；保留其余配置对应实现。
- **最小验证**。逐字段比较键、shape、dtype 与容差，按 rank 比较同种子索引序列、跨 epoch 尾批及非零断点后的首批和后续批次。

## D6 · 冻结编码器离线特征缓存

**证据**。OpenWAM 为已核实改造案例，也是受限单模型工作流参考。

- **Before**。以在线编码的实际调用和重复输入为候选。窗口重叠造成的重复解码与收益幅度须由目标项目的调用链和测量判断。
- **After**。构建阶段将冻结且处于 eval 的编码器输出按稳定键落盘，训练时读取匹配项并释放对应设备编码器占用。OpenWAM 的 VAE cache 以 episode 和 window 定位，T5 文本 store 以 prompt 定位。原子发布后再读取，校验模型权重、预处理、切窗和提示词契约，并支持中断续建。
- **边界**。VAE 缓存输入需要确定性。任何改变编码输入的随机增强，都需保留在线路径或重新建立能表达增强结果的契约。训练恢复后按实际加载的权重检查缓存，校验后再启动 worker。缺失、陈旧或混用缓存应可诊断，退回在线编码需要恢复相应 encoder 与 reader。
- **最小验证**。抽样对照在线特征及训练输入，测试 cache miss、损坏文件、权重变化和中断重建，测量磁盘与 CPU IO 资源。
