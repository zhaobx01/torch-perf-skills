# Trace 中的同步证据

本文说明如何从已采集的 PyTorch Profiler Trace（Chrome trace JSON / `.json.gz`）中核实同步是否真实发生以及是否真正造成 GPU 饥饿。本 Skill **不替用户启动训练或采集 Trace**；若尚无 Trace，请按缺证据降级协议给出最小采集建议，待用户授权或自行采集后再分析。

---

## 1. 工具选择：优先复用 `trace-bottleneck-report`，单 Trace 速查再用本脚本

| 场景 | 推荐工具 | 原因与边界 |
| --- | --- | --- |
| 已有完整多 rank Trace、需要判断同步是否真的导致 **GPU 饥饿** 及可回收上限 | 同仓库 `trace-bottleneck-report`（§4 同步与 GPU 饥饿表） | 按“同步调用窗口内设备是否空闲”判定真实代价，避免把 GPU 满载时的 `.item()` 误判成瓶颈 |
| 只有单份 Trace 需要快速检索同步事件、聚合 Python 调用栈，或在 HIP/DTK 上摸排实际事件命名 | 本 Skill 的 [scripts/analyze_trace_sync.py](../scripts/analyze_trace_sync.py) | 轻量、纯标准库、支持 `--list-names`；**只统计事件名与 CPU 侧耗时，不自动检查 GPU 轨道空闲** |
| 尚无 Trace，只想先在源码里筛可疑点 | 本 Skill 的 [scripts/scan_sync_points.py](../scripts/scan_sync_points.py) | **仅产出 `[静态]` 候选**，不能证明代码路径会执行或运行时会同步 |

---

## 2. 在 Trace 中核对哪些事件

不同后端的事件命名存在差异（HIP/DTK 的具体名称因构建版本而异，标为 **「版本兼容性待核对」**，首次分析陌生环境 Trace 务必先跑 `--list-names` 核实）：

| 关注类别 | CUDA 典型事件名 | ROCm / HIP 常见事件名（版本兼容性待核对） |
| --- | --- | --- |
| 显式同步 | `cudaDeviceSynchronize`、`cudaStreamSynchronize`、`cudaEventSynchronize` | `hipDeviceSynchronize`、`hipStreamSynchronize`、`hipEventSynchronize` |
| D2H / H2D 拷贝 | `Memcpy DtoH`、`Memcpy HtoD`、`cudaMemcpyAsync` | `Memcpy DtoH`、`Memcpy HtoD`、`hipMemcpyAsync`（部分构建沿用 CUDA 分类名） |
| Python 标量读回 | `aten::item`、`aten::_local_scalar_dense` | 同名（ATen 层不区分后端） |
| 动态形状与去重 | `aten::nonzero`、`aten::masked_select`、`aten::index`、`aten::_unique2` / `aten::unique_dim` | 同名 |

```bash
# 第一步：摸排真实事件名（尤其在 HIP/DTK 环境）
python3 scripts/analyze_trace_sync.py trace.json.gz --list-names

# 第二步：按同步类别汇总事件次数、总时长与 Python 调用栈
python3 scripts/analyze_trace_sync.py trace.json.gz
python3 scripts/analyze_trace_sync.py trace.json.gz --json
```

---

## 3. 判定某次同步是否真的造成“GPU 饥饿”

**不要单看 CPU 侧 `aten::item` 耗时下结论**，按 4 步交叉确认：

1. **定位 CPU 同步区间**：在 CPU 线程找到 `aten::item` / `Memcpy DtoH` / `*Synchronize` 的起止时间与调用栈（需采集时开启 `with_stack=True`；带栈 Trace 仅用于定位代码行，不用于对比绝对 step 耗时）。
2. **核对同期 GPU 轨道**：查看同一时间窗内设备 stream 轨道是否为空白。
   - 若同步期间 GPU 持续满载（例如 ≥90% 在跑真实计算/通信），说明 CPU 只是正常等待已有工作完成，**不是同步消除的高优目标**。
   - 若 GPU 在同步结束前就已排空并出现持续空白，直到同步返回后 CPU 才发出下一批 kernel launch，则属于真实的**同步饥饿**。
3. **核对发生位置与频率**：区分是每个梯度累积 micro-step 触发、每个 optimizer step 触发，还是仅初始化/checkpoint 步触发一次。
4. **改后同口径对照**：对比优化前后的两份 Trace，确认**同步事件次数**与**对应位置 GPU 空白总时长**同时下降；若只少了事件次数但 GPU 空白不变，说明该点不在关键路径上。

---

## 4. 常见误报与误读

- **张量早已在 CPU 上（静态扫描最常见误报）**：如先执行 `grid_thw.detach().to("cpu")` 做一次批量 D2H，再对结果调 `.tolist()`；或在 CPU 上构造的 `indices[i].tolist()`。看到扫描命中 `.tolist()` / `.numpy()` / `.item()` 时，先向上读 5–10 行确认对象所在设备。
- **次数高 ≠ 耗时大**：上万次微秒级已空闲读取，总代价可能远低于一次几十毫秒的阻塞 D2H 拷贝；先按总耗时和 GPU 空闲时长排序，不要按静态命中次数排序。
- **多 stream 并发假象**：CPU 线程在 `aten::item` 上等待时，若另一条 GPU stream 仍在满载执行，主机等待并不等价于 GPU 闲置。
- **采集窗口混入编译/预热**：跳过前若干步再采集稳态窗口；若与 `torch.compile` 同测，务必区分冷启动步与稳态步。
