# 来源与维护说明

版本：2.0.0；更新日期：2026-11-25。本文件仅作包内演进与来源说明，普通任务无需加载。

## v2.0.0 精简与 16 模型语料对齐（2026-11-25）

对照仓库统一的 `SKILL_DESIGN_PRINCIPLES.md`，保留 `P1–P7`、`decisions.md`、`validation.md` 与 7 份历史单模型案例的成熟骨架，做以下增量精炼：

1. **职责瘦身与上下游分流**：
   - 上游 Trace/HTA 瓶颈统计统一交给本地 `trace-bottleneck-report`（精简版 `hta-analysis.md` 已迁入 `trace-bottleneck-report/references/`）；
   - 删除 5 份外围/重叠文件：`hta-analysis.md`、`diagnostic-skills-primer.md`、`companion-skills.md`（取消 7 个外部 GitHub Skill 自动下载，仅对 `debug-graph-breaks` 与 `analyze-memory-snapshot` 保留可选软引用）、`data-movement.md`（同步/搬运/布局交由兄弟 Skill 承接）、随包提示词副本 `训练瓶颈分析报告生成提示词.md`（以仓库顶层 `prompts/` 为唯一维护源）；
   - 将 [turbophysai-compile-patterns.md](turbophysai-compile-patterns.md) 中重复的 `custom_op` 案例表收缩为指向 `optimize-kernel-replacement`（K6）的交接说明；同时明确承接全局性 Inductor 调优（`TORCHINDUCTOR_LAYOUT_OPTIMIZATION`、`hip_triton_configs`）。
2. **16 模型固定快照增量补充**：
   - 新增 [turbophysai-compile-evidence.md](turbophysai-compile-evidence.md)，统一索引 `zhaobx01/TurboPhysAI` 聚合分支 `corpus/all-models`（固定快照 `796d3f6bce998d91af9d336f506a769b2bdde927`），并在 P1/P3/P4/P5/P7 增量补齐 GR00T、DiT4DiT、LingBot-VLA-V2、PointPillars、OpenPI、FlashOCC、LeRobot-ACT、DreamZero、Sparse4D 的代表机制。
3. **契约对齐**：补齐双模式交付要素、四层/五层结论拆分与按生命周期界定的回退规范。

## 历史演进摘要（v1.0 – v1.10.0）

- **v1.1 – v1.4**：沉淀自 MapTRv2 本地快照（基线 `e03f097a`）、OpenVLA（`e7aaad02`，基线 `c8f03f48`）、MapTRv2 TurboPhysAI 包装版（`6fdd1d96` / `bc8ced4a`），建立 A/B/C/D 候选分类与自包含离线规则。
- **v1.7 – v1.10**：补充 Cosmos3 局部 AOT（`bb302a63`，基线 `9726697a`）、BEVFormer/BEVFusion（`bb302a63`）、OpenWAM（`e4528c39`），抽象出 `P1–P7` 跨模型问题索引、Dynamo-safe 改写、FSDP1 包装顺序与“注册≠调用点命中”验收规则。
- **官方依据（可选联网核对）**：PyTorch `where_to_apply_compile`、`torch.compile` API、`common_graph_breaks`、`torch_compile_full_example`、`torch.compiler.disable`；离线时以包内规范及目标项目实际 PyTorch 版本为准。
