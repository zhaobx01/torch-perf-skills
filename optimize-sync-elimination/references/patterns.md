# 同步消除模式（P1–P8）

以下模式按 **Before 坏味道 → After 手法 → 前提与不适用 → 最小验证** 组织。所有标记为 `[已核实改造案例]` 的条目均已对照 TurboPhysAI 16 模型快照的 Before 基线与 After 实现源码核实（索引见 [turbophysai-sync-evidence.md](turbophysai-sync-evidence.md)），不含未复测的性能承诺。

---

## P1 标量计算与分支留在设备上 `[已核实改造案例]`

- **Before 坏味道**：对设备张量调用 `.item()`、`int(x)`、`float(x)`、`bool(x)`、`if tensor:` 读回 Python 侧，仅为了传给下一个张量运算（如 loss 归一化因子、clamp 下限）、做简单条件选择，或通过框架 getter 隐式触发 `.item()`（如读取已算好的 `grad_norm`）。
- **After 手法**：
  - 下游只需张量运算输入时，保留零维设备张量，用 `.reshape(())` 或 `.detach()` 替代 `.item()`。
  - 条件分支改为设备端 `torch.where`、`clamp`、掩码乘法；对本身不可能含 NaN/Inf 的 `torch.bool` 张量，直接删除冗余的 `nan_to_num`（尤其旧版本 `.cpu().numpy()` 路径）。
  - 框架内部已有设备端标量属性（如 DeepSpeed ZeRO-2 的 `_global_grad_norm`）且契约可核实时，直接读取设备张量避免 getter 内部的 `.item()`。
- **前提与不适用**：
  - 确认下游算子接受零维张量且不改变混合精度 dtype 提升规则。
  - **正确性保护（非提速）**：若该标量用于即时决定是否中止训练、跳过 NaN 步或触发 checkpoint，消除同步会延迟错误暴露，必须明确标注权衡而非默认删掉。
- **最小验证**：同输入/随机态下前后数值与 dtype 一致；grep 或 Trace 确认目标 `.item()` 消失。

---

## P2 CPU 元数据留在 CPU，或一次性转 CPU 并缓存 `[已核实改造案例]`

- **Before 坏味道**：
  1. 本就来自 CPU 的小整数/尺寸元数据（如 `image_size`、文本有效长度）随 batch 整包 `.to(device)` 上卡，前向里又被 `.item()`/`.tolist()` 读回。
  2. 多模态位置编码/网格元数据（如 `grid_thw`、`input_ids`、`attention_mask`）在 GPU 上进入 Python `for` 循环，每轮把 GPU 标量传给 `torch.linspace`/`torch.arange`/`.item()`，引发数十至数百次逐标量 D2H。
- **After 手法**：
  - **源头留在 CPU**：在 batch 搬运边界精确拦截该字段保留在 CPU，其余字段正常上卡；不满足拦截条件时原样透传给原实现。
  - **一次性批量转 CPU + 缓存**：入口处一次性 `.detach().to("cpu").tolist()`，在 CPU 侧完成全部索引/模板构造，最后将结果张量一次性 `.to(device)`；对重复出现的形状组合按 `(grid_key, device, dtype)` 做有界缓存（如上限 16/64 条，满则清理防显存/内存泄漏）；若外层处于 `torch.compile` 区域内，对查表辅助函数加 `@torch._dynamo.disable` 隔离。
- **前提与不适用**：
  - 仅适用于体量小、不参与反向求导的索引/几何/尺寸元数据；大张量往返 CPU 反而引入更高拷贝与 CPU 算力开销。
  - 缓存必须检查 `not torch.jit.is_tracing()` 及设备/dtype 一致性，禁止缓存带梯度的张量。
- **最小验证**：覆盖未命中拦截条件、缓存命中/未命中、多图/视频/纯文本单模态分支，确认输出索引与原实现逐位一致。

---

## P3 动态形状同步与冗余去重（布尔索引 / `nonzero` / `.unique()`）`[已核实改造案例]`

- **Before 坏味道**：
  1. `x[bool_mask]` 布尔掩码索引或依赖 `nonzero()` 输出长度动态分配缓冲区，迫使设备回传非零个数。
  2. 对已经单调递增、天然无重复的索引（如 `torch.nonzero(cond, as_tuple=False).squeeze(-1)`）冗余调用 `.unique()`，触发额外排序与动态长度同步。
- **After 手法**：
  - 掩码赋值改为 `torch.where(cond, a, b)` 或算术表达（如 `back -= kept.to(back.dtype)` 替代 `back[kept] -= 1`）。
  - 需按掩码抽取元素且下游接受索引表时，用 `kept_indices = torch.nonzero(kept, as_tuple=False).flatten()` 配合 `torch.index_select`（并确认前后分配链是否仍触发同步）；已知上界时改用固定大小缓冲区 + 有效 mask。
  - 经确认输入天然唯一且无需重排时，直接删除冗余 `.unique()`。
- **前提与不适用**：
  - 若固定形状 + mask 方案引入大量无效计算，需实测权衡同步收益与多余计算开销。
  - 删 `.unique()` 前必须证明上游索引严格无重复且下游不依赖额外排序副作用。
- **最小验证**：覆盖全 0、全 1、单元素、空结果边界；验证选中与未选中位置的反向梯度一致。

---

## P4 绕开高层 varlen / Attention 封装里的隐式 `.max().item()` `[已核实改造案例]`

- **Before 坏味道**：FlashAttention 等高层封装（或 `_update_causal_mask`）在内部对 `seqlens` 做 `torch.nonzero` + `.max().item()` 或反复构建 `cu_seqlens` / `chunk_lengths`。
- **After 手法**：
  - 在可验证的特定前提下（如 FA2 + prefill + 右 padding 无前置 0），直接返回 `None` 绕开 unpad/varlen 路径（可保留一次低成本右 padding 校验，不满足时显式回退原函数）。
  - 对已知打包结构或可缓存的 `grid_thw`，预构造/缓存 `cu_seqlens`、`chunk_lengths` 直接传给底层实现。
- **前提与不适用**：
  - 强依赖序列布局假设（如右 padding、batch=1 打包、`past_seen_tokens == 0`），**必须显式校验前提并在不满足时回退**（或用 `torch._check` 声明），严禁静默走错分支。
  - 若改动目的是消除 `torch.compile` 重编译（如 `max_seqlen` 分桶），归 `torch-compile-integration` 主导。
- **最小验证**：分别构造满足前提（右 padding）与不满足前提（左 padding、含 cache、变长）的输入，确认输出与梯度均与原路径一致且不满足时确实走回退。

---

## P5 指标聚合、梯度累积跳过与延迟读取 `[已核实改造案例]`

- **Before 坏味道**：
  1. **梯度累积微步白同步**：每个 micro-step 末尾都对 loss 调用 `.item()`，但外层仅在 `sync_gradients=True`（真正 optimizer step）才打日志。
  2. **前向内部过早物化**：`model.forward()` 返回 `loss_dict` 时立刻调 `.item()`，即使本步不记录日志或本 rank 不输出。
  3. **逐项归约与逐项读回**：循环内对多个 loss 分别 `dist.all_reduce` + `.item()`。
  4. **每步阻塞读回**：每个 optimizer step 立刻同步读取 loss / `grad_norm` 排空 GPU 队列。
- **After 手法（按侵入度由低到高）**：
  1. **微步跳过**：在 `not sync_gradients`（非日志步）直接返回 `{}` 或保留设备张量，不调用 `.item()`。
  2. **推迟到日志边界（强耦合原子组）**：前向内只存 `loss.detach()` 设备张量，仅在 logger/wandb 消费边界才执行 `.item()`——**生产端与消费端必须成组启停与回退**。
  3. **多标量合并单次归约与单次 D2H**：`torch.stack` 合并全部标量 → 单次 `dist.all_reduce` → 单次 `.cpu().tolist()`。
  4. **延迟一步异步读取**：拷到 `pin_memory=True` host buffer + `cuda.Event.record()`，在下一步 `push` 时再 `event.synchronize()` + `.tolist()`（若与异步集合通信共存，可在独立 `cuda.Stream` 上 wait 并对源张量调用 `record_stream` 防止 allocator 提前复用）。
- **前提与不适用**：
  - 延迟一步读取要求本步训练控制流不依赖即时指标（如早停、动态调 lr、按本步 loss 选 best checkpoint 需核对步号语义）。
  - 推迟 `.detach()` 张量时必须确保无引用滞留导致计算图/显存泄漏（务必 `.detach()`，且在非日志步及时丢弃）。
- **最小验证**：核对日志步号与数值对齐、多 rank 归约值一致、梯度累积开启/关闭时均无未消费张量报错与显存增长。

---

## P6 消除库与训练循环中多余的显式同步 `[已核实改造案例]`

- **Before 坏味道**：
  1. 训练循环把 `torch.cuda.synchronize()` 或 `dist.barrier()` 写在梯度累积判断之外，导致每个 micro-step 强制全卡同步。
  2. 分布式框架/计时器默认开启每步同步（如 DeepSpeed `timers.throughput` 默认在 `engine.step` 内调用同步）。
- **After 手法**：
  - 将 `barrier()` / 显式同步收回到真正需要的日志、GC 或 checkpoint 分支内（仅在 `should_log` / 真正 optimizer step 触发）。
  - 通过框架配置开关关闭纯诊断/计时用同步（如 `timers.throughput = {"enabled": False, "synchronized": False}`）。
- **前提与不适用**：
  - **正确性保护（非提速）**：checkpoint 落盘前后的 `staging_stream.synchronize()` / `dist.barrier()`、分布式 OOM/异常汇总前的同步，属于正确性保护，不应作为性能瓶颈删除。
  - 删除显式同步前，需核实后续代码没有隐式依赖该同步点来保证 buffer 可释放或文件可见。
- **最小验证**：开启与关闭梯度累积（`grad_accum=1` 与 `>1`）各跑通多步，确认日志、checkpoint 保存/恢复与多 rank 行为正常。

---

## P7 必须留在 CPU 的求解收敛为单次 D2H / H2D `[已核实改造案例]`

- **Before 坏味道**：匈牙利匹配（`scipy.optimize.linear_sum_assignment`）等纯 CPU 求解函数内部夹杂零散的 `.item()` 或多次分片搬运，或被错误包进 `torch.compile` 区域。
- **After 手法**：把完整代价矩阵一次性 `.detach().cpu()` 搬到主机求解，结果 `np.stack` / `torch.as_tensor(..., device=device)` 一次性搬回设备，并保持在编译图边界之外。
- **前提与不适用**：这是“收敛为一次”而非“彻底消除”；若匹配矩阵尺寸显著增大，需重新评估 CPU 求解本身是否成为瓶颈。
- **最小验证**：确认整个求解函数只触发 1 次 D2H + 1 次 H2D，且匹配结果与原实现完全一致。

---

## P8 步内非阻塞拷贝的安全条件 `[已核实改造案例]`

> 范围说明：本条仅服务于步内小张量 H2D/D2H 与延迟指标异步拷贝（如 `P5`）。DataLoader 架构改造（`pin_memory`/`persistent_workers`、batch 级独立 H2D 预取流）归**数据管线**方向主导。

- **Before 坏味道**：盲目加 `tensor.to(device, non_blocking=True)` 或 `host.copy_(dev, non_blocking=True)`，但未确认源/目标主机内存已 pinned，或在非默认 stream 上缺少同步等待与 `record_stream` 保护。
- **After 手法**：
  1. 确保主机侧张量来自 `pin_memory=True`（或对小张量显式 `.pin_memory()`），否则 `non_blocking=True` 静默退化为同步拷贝。
  2. 在真正消费数据前建立明确同步点（`event.synchronize()`、`stream.wait_event()` 或同 stream 串行化）。
  3. 跨 stream 传递张量时，对在源 stream 分配、在另一 stream 消费的张量调用 `tensor.record_stream(consumer_stream)`，防止缓存分配器提前复用显存。
- **最小验证**：核查“改了没变快”→ 是否未 pinned 或消费点过早等待；核查“变快但偶尔出错”→ 是否缺少同步点或 `record_stream`。

---

## 通用不适用场景

- 纯 CPU 数据预处理路径（不涉及 GPU 队列，优化它属于 CPU/数据管线问题）。
- 已经在 `torch.compile` 区域内部触发的 graph break（优先交 `torch-compile-integration` 诊断）。
- 一次性初始化或 checkpoint 加载期的 `.item()`（不在训练热路径，收益可忽略）。
- 承担 NaN/Inf 即时拦截或分布式一致性保护的同步点（若确需权衡，必须单标「正确性保护（非提速）」并由用户拍板）。
