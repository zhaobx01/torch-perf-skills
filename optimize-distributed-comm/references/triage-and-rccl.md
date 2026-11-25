# 慢 Rank 分流、Trace T5 口径与 HIP/RCCL 死锁防护

本文处理分布式通信优化中最容易踩的两个坑：**① 把慢 rank 偏斜误判成通信重叠不足**；**② 在 ROCm/HIP（RCCL）上改写通信流导致多卡乱序死锁**。

---

## 1. 先排慢 Rank，再做通信重叠（对接 `trace-bottleneck-report` T5）

通信 kernel（`nccl*` / `rccl*`）的耗时包含两部分：**真正的数据传输时间** + **早到 rank 等最慢 rank 到达集合点的时间**。如果快卡一直在等慢卡，调大 bucket、开 prefetch 或重写 AllGather 流水都不会有收益。

按 `trace-bottleneck-report` 的 §1（最慢 rank）与 §6（T5 通信表）做分流：

| Trace 观察信号 | 根因判断 | 主导动作 |
| --- | --- | --- |
| T5 暴露通信高，**但起始极差（中位/最大）大**，或 §1 显示某 rank 空闲/step 显著偏慢（`> --slow-rank-ms`） | **假通信瓶颈（慢 rank 偏斜）** | **暂停通信重叠改造**；对比慢 rank 与快 rank 的 DataLoader 耗时、变长序列/有效 token 数、单卡 GC 尖刺、设备频率/温度或外部进程干扰 |
| 各 rank 起始极差小（时钟可比且基本齐整），但 T5 **重叠率低**、**暴露占通信%高** | **真通信暴露（重叠不足）** | 进入本 Skill 的 `C1`–`C4`（必要时 `C6`–`C8`）提高通信与计算重叠 |
| 各 rank 齐整且重叠率已较高，但 T5 **集合次数极多 / 平均字节极小**（算法带宽远低于链路带宽） | **小包碎通信** | 优先看 `C3`（合并 FSDP2 `fsdp_layers_per_group`）、`C1`（关闭 `optimize_ddp` 碎图）或 `C4`（调大 `reduce_bucket_size`） |
| 通信总字节数本身异常偏大（如全卡重复 `all_gather` 或偏斜 `ReduceScatter` 零填充） | **冗余通信量** | 优先看 `C3`（`reshard_after_forward=False`）、`C5`（owner 分片 + `all_to_all_single`）或二级 `C7`（owner 定向 Reduce） |

> **多机时钟不可比提示**：`trace-bottleneck-report` 检查各 rank step 起点差 `>= 50 ms`（或多机环境）时会把起始极差标为“不可比”。此时不能靠绝对时间戳算极差，改用各 rank **非通信计算+空闲时长分布**、步长中位数差异与通信 kernel 内最慢/最快耗时极差交叉推断是否存在慢 rank。

---

## 2. HIP / RCCL 正确性保护（非提速）

以下 3 条机制用于**预防多卡死锁与通信器崩溃**，在分析与实施交付中必须单标为 **「正确性保护（非提速）」**，绝不混入性能增益候选，也不得在后续调优时被误删：

### R1 单 ProcessGroup 规范提交流（防 RCCL 多 stream 乱序死锁）`「正确性保护（非提速）」`
- **背景与风险**：FSDP2 或自定义异步通信可能从多个 CUDA/HIP stream 向同一个 `ProcessGroup` 提交集合通信。RCCL 要求同一 communicator 上各 rank 的集合调用保持严格一致的全局提交顺序，多 stream 并发提交极易在不同 rank 间交错导致永久 hang（死锁）。
- **做法**：为每个 `(device, process_group)` 维护一条规范（canonical）提交 stream；每次提交集合前用 `collective_stream.wait_stream(caller_stream)` 接力，在规范 stream 上提交 `all_gather_into_tensor` / `reduce_scatter_tensor` / `all_reduce`，再让调用方 stream 按需 `wait_stream`（或通过 FSDP2 `set_custom_all_gather` / `set_custom_reduce_scatter` 统一接管）。
- **代偿**：单 stream 串行化提交可能略微约束跨流并发，属于**以可控开销换取零死锁的正确性保护**。

### R2 元数据集合切独立 Gloo Group（规避 RCCL `gather_object` 故障）`「正确性保护（非提速）」`
- **背景与风险**：在部分 HCU/RCCL 构建上，PyTorch Distributed Checkpoint（DCP）等路径通过 NCCL/RCCL 后端执行 Python 对象序列化收集（`gather_object`）可能失败或卡死，即使普通张量通信正常（「版本兼容性待核对」）。
- **做法**：懒创建一个独立 CPU/Gloo 后端的 `ProcessGroup`（`dist.new_group(backend="gloo")`），专门传给 DCP 元数据收集接口，与训练张量通信组完全隔离。

### R3 条件分支下的 FSDP / 分布式 Collective 次数对齐 `「正确性保护（非提速）」`
- **背景与风险**：当不同 rank 走不同控制流分支（如动态采样步数、按 batch 内容决定是否执行 CFG 文本分支）时，FSDP 会在某些 rank 多触发一次或少触发一次 `AllGather`，直接触发分布式死锁。
- **做法**：在进入条件分支或动态循环前，先做一次全局规约（如 `any` / `max`）对齐所有 rank 的分支标志或迭代上限；本 rank 不需要额外计算时，补跑 dummy forward（保持 FSDP collective 序列完全对齐）后丢弃输出。

---

## 3. 多 ProcessGroup 与异步通信器的隐蔽风险

- **避免与业务小通信争队列**：像 `deferred_gather` 这类跨步在后台持续运行的重型 broadcast，如果复用默认 `ProcessGroup`，会卡住步末的小指标 `all_gather` / `all_reduce`。为后台长通信建独立 `ProcessGroup` 可以解耦队列，但每增加一个 NCCL/RCCL 通信器都会额外占用数百 MB 级显存，须核对显存预算。
- **跨 stream 张量生命周期**：只要在 Stream A 分配张量、在 Stream B 等待集合完成或拷贝它，必须对涉及张量调用 `tensor.record_stream(stream_B)`，防止缓存分配器提前复用引发静默数据污染。
