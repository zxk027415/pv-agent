# PV Diagnosis Agent · 设计文档

> 版本：v0.1（2026-10-24）· 进度更新：2026-10-25，M1 规则诊断链路已实跑，DeepSeek 真实调用与 Agent 评测尚未执行
> 一句话：基于 LangGraph + MCP 的光伏电站时序故障诊断 Agent，在注入已知故障的公开数据上做可复现评测，并用固定评分器驱动 harness 自进化。

---

## 0. 目标、非目标与成功标准

**目标**
1. 给定电站和时间段，Agent 自主取数、分析、核验证据，输出「是否有故障 → 哪台逆变器 → 什么故障 → 哪段时间 → 证据链」的诊断报告。
2. 诊断能力以 MCP Server 形式提供，Agent、Claude Desktop 等任意 MCP 客户端都能调用。
3. 有一套带标准答案的基准集，所有结论都能用数字说话：多档消融对比 + 自进化前后对比。

**非目标**（刻意不做，避免过度设计）
- 不做 Temporal / 分布式工作流、RBAC、多租户、六层记忆。
- 不做实时流接入；数据按批次回放即可。
- 不训练大模型；LoRA 微调列为 M4 可选项。
- 不追求 SOTA 检测算法；诊断算法以统计 + 物理模型 + 同站对比为主，够用、可解释。

**成功标准（M2 验收时必须有的数字）**
- 留出集上根因 Top-1 准确率，以及 A0→A3 各档之间的差距（见 §9）。
- 正常场景误报率。
- 证据忠实度（LLM-as-Judge）与工具调用成功率。
- 写操作未经审批执行次数 = 0。

**诚实边界**
- 本项目与远景实习代码**零复用**：新仓库从零编写，只用公开数据，不复制 AIP 下任何代码、转换结果或分析产物。
- 对外表述只写跑出来的数字；规则基线如果比 Agent 强，就如实写出来，并分析原因。

---

## 1. 定位与差异化

| 参考项目 | 借鉴什么 | 本项目的不同之处 |
|---|---|---|
| IBM/AssetOpsBench（KDD 2026） | 领域 MCP Server 拆分，场景化评测 | 领域是光伏，不是冷水机组；故障通过注入构造，有可量化的真值 |
| HolmesGPT（CNCF） | toolset 设计、默认只读、证据驱动的调查报告 | 对象是工业时序数据，不是 K8s 日志 |
| TimeCopilot | LLM 与时序模型结合 | 做诊断，不做预测 |
| google-research/harness2 | 提议 → 试跑 → 合并（Propose–Probe–Compose）式的 harness 优化 | 简化为单层循环，评分器固定 |
| Dream-RSI | 从历史中生成新练习环境 | 用故障原语组合生成针对薄弱类别的新场景，只进开发池 |
| 自己的 AutoResearch 经验 | 固定评分器，候选只叠加补丁，多窗口复验 | 对象从调参换成 Agent harness |

---

## 2. 数据

### 2.1 数据源（2026-10-24 已验证可匿名访问）

AWS 开放数据桶 `oedi-data-lake`，前缀 `pvdaq/2023-solar-data-prize/`，无需凭据。

| 用途 | system_id | 位置 | 规模 | 关键属性 |
|---|---|---|---|---|
| **主数据** | **2107** | Arbuckle, CA | 893 kW DC，24 台逆变器 | 固定支架，倾角 25°、方位 180°；有 POA 辐照度；5–15 分钟分辨率；2018-12 → 2023-11 |
| 跨站泛化 | 9069 | Social Circle, GA | 38.7 MW，40 台逆变器 | 固定支架，倾角 20°、方位 180°；多个 POA 辐照计；5 分钟分辨率 |
| 备选 | 2105 | Maui, HI | 110 kW | 朝向「varies」，不适合物理模型，不用 |

目录结构：`<id>_OEDI/data/*.csv` + `<id>_OEDI/metadata/<id>_system_metadata.json`，元数据里有经纬度、支架、组件、逆变器和各测点的说明。

**M1 第一件事要验证的**：
- `measured_on` 是 UTC 还是当地时间。2105 的 00:30 出现 GHI≈538，推测是 UTC，2107 需要单独确认。
- 2107 各逆变器的数据列名和单位（kWh 累计量还是 kW 功率）。
- 缺测比例；挑出数据完整的 6–12 个月作为工作区间。

### 2.2 数据处理流水线

```
download.py  ──▶ data/raw/<id>/            (原始 CSV，不入库)
normalize.py ──▶ data/clean/<id>.parquet   (统一时区为 UTC、统一单位为 kW 和 W/m²、长表 schema)
datacard.py  ──▶ data/clean/<id>_card.json (覆盖率、缺测、各逆变器额定容量等)
```

统一 schema（长表）：`ts (UTC) | system_id | device_id | metric | value`，metric ∈ {ac_power_kw, dc_power_kw, poa_wm2, ambient_c, module_c}。

`data/` 整体写入 `.gitignore`，仓库里只放脚本。

---

## 3. 故障注入与场景生成

### 3.1 故障原语（6 类 + 正常）

在**干净的真实数据**上按公式注入，每类故障对应一条可以区分它的「判别证据」，这条证据也是 Agent 应该去找的东西。

| 代码 | 故障 | 注入方式 | 判别证据 |
|---|---|---|---|
| F1 | 积灰 soiling | 单台逆变器功率 × (1 − r·t)，r ∈ [0.1, 0.5] %/天，清洗日重置 | 相对同站设备的性能比呈缓慢线性下降，且有阶跃式恢复 |
| F2 | 局部遮挡 shading | 每天固定时段（如 15:00–17:00）功率 × (1 − s)，s ∈ [0.3, 0.7] | 损失集中在每日的同一时段，且逐日重复 |
| F3 | 逆变器停机 | 辐照度 > 200 W/m² 时功率 = 0，持续若干小时到若干天 | 功率为零、辐照正常、同站其他设备正常 |
| F4 | 组串失效 | 某时刻起功率阶跃 × (1 − k/N)，k/N ∈ {1/8, 1/4} | 阶跃下降后保持在一个固定比例，不随时段变化 |
| F5 | 辐照计漂移 | POA 读数 × (1 + d·t) 缓慢漂移 | **全站**设备同时「看起来」性能下降，但设备之间的相对关系不变 |
| F6 | 通讯中断 | 单台设备整段数据缺失（NaN） | 是缺测而不是零值；其他测点正常 |
| N | 正常 | 不注入；可以包含阴雨天等干扰 | 用来衡量误报 |

### 3.2 场景与真值

```json
{
  "scenario_id": "S2107-F2-0013",
  "system_id": 2107,
  "query_window": ["2019-06-01", "2019-06-21"],
  "faults": [{"type": "F2", "device_id": "inv07", "start": "2019-06-08", "end": "2019-06-21",
              "params": {"window": "15:00-17:00", "severity": 0.5}}],
  "difficulty": "medium",
  "split": "test"
}
```

难度维度：严重度（轻 / 中 / 重）、故障时长、窗口内的阴雨天比例、双故障叠加（只放 hard 档）。

### 3.3 数据划分（防泄漏）

- **开发集**和**留出集**在**时间段**和**逆变器**两个维度上同时不相交（这条直接沿用光伏项目「未见日期 × 未见设备」的经验）。
- 9069 上的场景只用于跨站测试，不参与任何调优。
- 留出集在自进化过程中**不可见**，每个阶段最多评测一次。

### 3.4 场景生成器（借鉴 Dream-RSI）

`scenario_gen.py` 根据开发集的失败分布，组合新场景：优先生成 Agent 表现差的故障类型、难度和干扰组合。生成的场景**只进入开发池**，绝不进入留出集。

---

## 4. 总体架构

```
            ┌──────────────── ⑥ 评测与可观测 ────────────────┐
            │ 基准集 · 固定评分器 · LLM-as-Judge · Ragas · Langfuse │
            └───────────────────────▲────────────────────────┘
┌─ ⑤ 自进化 ───────────┐            │ 打分
│ 提议 → 试跑 → 合并    │──改进──▶ harness（提示词 / Playbook / 工具描述）
└──────────▲───────────┘
           │ 失败轨迹 / 经验
┌─ ④ 记忆 ──────────────────────────────────────────────┐
│ 短期：LangGraph State + PostgresSaver                  │
│ 长期：MemoryStore 接口 → OpenViking 或 pgvector         │
└──────────▲────────────────────────────────────────────┘
┌─ ③ Agent（LangGraph）─────────────────────────────────┐
│ 范围检查 → 规划 → 并行取证 → 假设 → 核验 → (审批) → 报告  │
└──────────▲────────────────────────────────────────────┘
           │ MCP（stdio / streamable HTTP）
┌─ ② MCP Servers ──────────────────────────────────────┐
│ pv-data │ pv-analytics │ pv-knowledge │ pv-ops（写操作）│
└──────────▲────────────────────────────────────────────┘
┌─ ① 数据与仿真：PVDAQ 2107/9069 · 故障注入 · 场景生成 ──┐
└──────────────────────────────────────────────────────┘
```

---

## 5. MCP 工具设计

框架用 FastMCP。每个工具的入参和出参都用 Pydantic 定义；出参统一带 `evidence_id`，用于后续引用核验。

### 5.1 pv-data（只读）

| 工具 | 入参 | 出参要点 |
|---|---|---|
| `list_devices` | system_id | 设备列表、额定容量 |
| `get_timeseries` | system_id, device_ids, metrics, start, end, resample | 降采样后的序列摘要和统计量；**不返回原始大表**，避免撑爆上下文 |
| `data_quality` | system_id, device_ids, start, end | 覆盖率、缺测段、零值段、异常跳变 |
| `get_site_info` | system_id | 经纬度、倾角、方位、组件 / 逆变器型号 |

### 5.2 pv-analytics（只读，计算全部在工具内完成）

| 工具 | 说明 |
|---|---|
| `expected_power` | 用 pvlib 根据 POA 辐照度和温度算理论功率，返回逐设备的性能比 PR 序列与偏差 |
| `peer_compare` | 同站设备横向对比：每台设备相对站内中位数的归一化比值，以及离群设备 |
| `detect_anomalies` | 统计检测（滚动 z-score / 变点检测），返回异常段列表；可选 `method="chronos"` 做对比 |
| `profile_pattern` | 对异常段做形态画像：阶跃 / 渐变、日内时段集中度、零值占比、是否逐日重复 |
| `compare_periods` | 两个时段的性能比、发电量对比 |

### 5.3 pv-knowledge（只读）

| 工具 | 说明 |
|---|---|
| `search_fault_kb` | BM25 + 向量混合检索 → BGE-Reranker 重排，返回故障机理、典型特征和处置建议，带出处 |
| `get_playbook` | 按故障类型读取诊断 Playbook（SKILL.md 格式，也是自进化的对象） |

知识库语料只收**许可证允许再分发的公开资料**，比如 NREL 公开报告、pvlib 文档。收录前逐份核对许可证，核对结果记录到 `kb/SOURCES.md`。

### 5.4 pv-ops（写操作，必须人工审批）

| 工具 | 说明 |
|---|---|
| `create_work_order` | 生成工单（落到本地 SQLite 或 JSON，模拟工单系统） |

**pv-ops 默认不加载**，只有显式开启时才挂到 Agent 上；即使挂上，调用前也必须经过 `interrupt` 审批（见 §6.3）。

---

## 6. Agent 设计（LangGraph）

### 6.1 State

```python
class DiagState(TypedDict):
    request: DiagRequest            # system_id, window, question
    plan: list[Step]
    evidence: dict[str, Evidence]   # evidence_id -> 工具结果摘要
    hypotheses: list[Hypothesis]    # fault_type, device_id, window, confidence, evidence_ids
    verdict: Verdict | None
    critique: list[str]             # 核验节点给出的问题
    loops: int                      # 取证-核验循环次数，上限 3
    pending_action: Action | None   # 待审批的写操作
    memory_hits: list[CaseRef]
```

### 6.2 图结构

```
START → guard_in → recall_memory → plan → investigate(并行工具调用)
      → hypothesize → verify ─(证据不足 且 loops<3)→ investigate
                             └(通过)→ propose_action ─(需要写操作)→ approval[interrupt] → act
                                                    └(不需要)→ report → guard_out → write_memory → END
```

- `investigate`：根据计划并行调用互不依赖的工具，结果写入 `evidence`。
- `verify`：用规则检查 + LLM 自评。每条假设至少引用 2 条独立证据；还要检查是否排除了 F5（全站漂移）这类混淆项。不通过就把 `critique` 回传，回到取证。
- `report`：结构化输出（Pydantic），包括结论、证据表、置信度，可以输出「无法判定」。

### 6.3 Guardrails（三层）

| 层 | 规则 |
|---|---|
| 输入 | 请求必须在诊断范围内；system_id 必须存在；时间窗不超过 90 天 |
| 工具 | 每个节点有工具白名单；参数用 Pydantic 校验；单次诊断设工具调用上限；写工具只能经由 `approval` 节点调用 |
| 输出 | 报告里每个数字都必须能在 `evidence` 中找到出处；每条结论都必须带 evidence_id；不满足就降级成「需要人工复核」 |

人工审批用 LangGraph 的 `interrupt()` 暂停执行，API 层用 `Command(resume=...)` 恢复。审批记录写入运行日志。

### 6.4 模型

通过 `init_chat_model` 统一接入，可切换。2026-10-25 用户选择 **DeepSeek 的 OpenAI 兼容接口**；当前入口默认 `deepseek-chat`，尚未发起真实调用。每次评测记录配置模型、接口返回模型名和温度；别名不等于固定版本，正式评测仍需核实并锁定实际版本。

---

## 7. 记忆

| 类型 | 实现 | 内容 |
|---|---|---|
| 短期 | LangGraph State + `PostgresSaver` | 单次诊断的执行检查点；中断后可恢复，也可回放 |
| 长期 · 案例 | `MemoryStore.add_case / search_cases` | 历史诊断的摘要、最终结论、是否正确（评测时才知道） |
| 长期 · 经验 | `MemoryStore.add_lesson` | 自进化循环沉淀下来的经验 |
| 程序性 | `playbooks/*.md`（SKILL.md 格式） | 各类故障的诊断步骤，可以被自进化修改 |

`MemoryStore` 是一个抽象接口，有两个实现：
- `PgVectorStore`：默认实现，M2 完成；
- `OpenVikingStore`：M3 再接入。它把案例、经验和 Playbook 映射到 `viking://` 下的不同目录，利用分层摘要（先看 L0 / L1 摘要，再按需读原文）控制上下文长度。OpenViking 的 Python 接口目前**没有实际验证过**，M3 开工前先做一个小实验；如果不可用，就保留 pgvector 实现，并在文档里说明原因。

**防泄漏**：记忆只能从开发集的运行结果写入；评测留出集时记忆只读，留出集的结果不回写。

---

## 8. 自进化循环（轻量 RSI）

**改进对象**（harness，模型权重不动）：系统提示词、`playbooks/*.md`、工具描述文本。三者都存在 `harness/` 目录下，做版本管理。

**一轮流程**：
1. **评测**：用当前 harness 跑开发集，固定评分器打分，收集失败轨迹（从 Langfuse 导出）。
2. **提议（Propose）**：LLM 读失败轨迹，提出 k 个补丁，每个补丁都是对 harness 文件的一个 diff，并附上它针对的失败模式。
3. **试跑（Probe）**：每个补丁分别在「失败子集 + 随机抽样的开发子集」上运行，观察行为差异。
4. **合并（Compose）**：把互不冲突、单独有效的补丁合并；在完整开发集上**严格更优（超过阈值）且没有类别退化**才采用。
5. **沉淀**：被采用的补丁及其理由写成 lesson，存入记忆；被拒绝的补丁也记录下来，以免重复提出。

**约束**：
- 评分器代码固定，有版本号，自进化过程不能改动。
- 每轮最多 k=4 个补丁，共 N=5 轮，所有轮的 token 用量都要记录。
- 留出集只在第 0 轮和最后一轮各评测一次，用来报告最终提升。

对外报告的指标：第 0 轮 → 第 N 轮在**留出集**上的 Top-1 提升，以及每一轮在开发集上的分数曲线。

---

## 9. 评测

### 9.1 基准集规模（初版）

| 集合 | 组成 | 用途 |
|---|---|---|
| 开发集 | 2107 · 6 类 × 10 + 正常 20 = 80 个场景 | 调试与自进化 |
| 留出集 | 2107 · 6 类 × 10 + 正常 20 = 80 个场景（时间和设备都与开发集不相交） | 报告主指标 |
| 跨站集 | 9069 · 6 类 × 5 + 正常 10 = 40 个场景 | 泛化 |

### 9.2 指标（评分器为纯代码，固定）

| 指标 | 定义 |
|---|---|
| 检测 P / R / F1 | 场景级的「有故障 / 无故障」 |
| 设备定位准确率 | 故障设备判断正确 |
| 根因 Top-1 / Top-3 | 故障类型判断正确 |
| 时间窗 IoU ≥ 0.5 的比例 | 预测的故障时间段与真值的重叠程度 |
| 正常场景误报率 | 正常场景中报出故障的比例 |
| 证据忠实度 | LLM-as-Judge，按 1–5 分打分：结论是否被引用的证据支撑；同时用代码检查数字能否在证据中找到出处 |
| 工具成功率 / 平均调用数 | 来自运行轨迹 |
| 成本 | 每次诊断的 token 数、耗时 |
| 安全 | 未经审批的写操作次数（必须为 0） |

知识检索子模块单独用 Ragas 评测（context precision / recall、faithfulness），配一套 30 题左右的问答集。

### 9.3 消融（主表）

| 档位 | 配置 |
|---|---|
| B0 | **规则基线**：不用 LLM，只用阈值规则加判别证据表 |
| A0 | LLM，只给数据摘要，不给工具 |
| A1 | + pv-data |
| A2 | + pv-analytics（物理模型 + 同站对比 + 形态画像） |
| A3 | + 核验节点 + pv-knowledge |
| A4 | + 长期记忆 |
| A5 | + 自进化后的 harness |

### 9.4 可观测

所有 LLM 和工具调用都接入 Langfuse，记录 trace、成本和延迟。基准集同步为 Langfuse Dataset，每次评测运行对应一个 Run，便于对比。部署方式默认用 Langfuse Cloud 的免费档，因为数据都是公开的，不涉及隐私；自托管需要 Postgres + ClickHouse + Redis + 对象存储，太重，不做。

---

## 10. 服务化

- **FastAPI**：
  - `POST /diagnose`：通过 SSE 流式返回各节点的进度和最终报告
  - `POST /runs/{id}/approve`：审批写操作
  - `GET /runs/{id}`：查询运行详情
- **前端**：Gradio 单页，包括诊断输入、流式过程、证据图表和审批按钮。
- **MCP 独立运行**：每个 Server 都可以单独通过 stdio 启动，在 Claude Desktop 中配置后直接调用（作为演示）。
- **Docker Compose**：包含 app、postgres（带 pgvector）和 3 个 MCP Server。

---

## 11. 目录结构

```
pv-diagnosis-agent/
├── docs/DESIGN.md
├── pyproject.toml            # uv 管理，Python 3.11
├── data/                     # .gitignore
├── src/pvdiag/
│   ├── data/                 # download / normalize / datacard
│   ├── sim/                  # faults.py（原语）· scenarios.py · scenario_gen.py
│   ├── mcp_servers/          # pv_data.py · pv_analytics.py · pv_knowledge.py · pv_ops.py
│   ├── agent/                # state.py · graph.py · nodes/ · guards.py
│   ├── memory/               # base.py · pgvector_store.py · openviking_store.py
│   ├── evolve/               # propose.py · probe.py · compose.py
│   ├── eval/                 # scorer.py（固定）· judge.py · run_bench.py · ablations.py
│   └── api/                  # FastAPI + SSE
├── harness/                  # system_prompt.md · playbooks/*.md · tool_descriptions.yaml
├── kb/                       # 知识库语料 + SOURCES.md
├── bench/                    # 场景 JSON（由脚本生成，可以入库）
├── ui/                       # Gradio
├── tests/
└── docker-compose.yml
```

---

## 12. 技术选型与版本（2026-10-24 从 PyPI 查询）

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.11 | pvlib 0.16 要求 ≥3.11，langfuse 5 要求 <4.0 |
| langgraph | 1.2.17 | |
| langgraph-checkpoint-postgres | 3.2.0 | |
| langchain-mcp-adapters | 0.3.2 | `MultiServerMCPClient`，支持 stdio 和 http |
| fastmcp | 4.1.3 | |
| langfuse | 5.0.4 | |
| pvlib | 0.16.1 | |
| chronos-forecasting | 2.3.2 | 可选，Mac CPU 上的速度待测 |
| rank-bm25 / FlagEmbedding | 0.2.2 / 1.4.2 | 混合检索与重排 |
| openviking | 0.5.2 | M3 前先做小实验验证 |

开工时用 `uv` 锁定版本；上表只说明调研时的最新版，不代表一定兼容，兼容性在 M1 安装时验证。

---

## 13. 里程碑（按业余时间每天 2–3 小时估算）

| 里程碑 | 内容 | 预计 | 验收标准 |
|---|---|---|---|
| **M1 跑通** | 2107 下载、清洗、数据卡；F3 / F4 两类故障注入；pv-data 和 pv-analytics 的核心工具；LangGraph 主流程（不含核验节点）；10 个场景的冒烟评测 | 1 周 | 端到端跑通，能输出结构化报告；评分器能打分；pytest 通过 |
| **M2 出数字** | 6 类故障 + 正常场景；完整基准集；核验节点、Guardrails、人工审批、PostgresSaver；pv-knowledge 与 Ragas；Langfuse；B0、A0–A3 消融 | 1.5 周 | 消融主表跑完，留出集指标齐全 → **更新简历** |
| **M3 进化** | MemoryStore（pgvector → OpenViking）、A4；自进化 N 轮、A5；场景生成器；9069 跨站评测 | 1 周 | 第 0 轮 → 第 N 轮的留出集对比；跨站结果 |
| **M4 打磨（可选）** | Gradio、Docker Compose、README 和架构图、在 Claude Desktop 演示；小模型 LoRA 微调做工具调用 | ≥0.5 周 | 新环境下一条命令即可启动 |

---

## 14. 风险与待验证

| 风险 | 应对 |
|---|---|
| 2107 时区、单位、缺测情况不明 | M1 第一步写数据卡；如果不可用，改用 9069 的某个子集 |
| 规则基线 B0 比 Agent 强 | 如实报告。Agent 的价值放在可解释报告、混淆项排除、新故障组合泛化上；同时分析失败原因 |
| 注入故障过于理想化，任务太简单 | 设难度分级、加阴雨天干扰和双故障叠加；看真实数据中有没有自然故障，有就作为额外的定性案例 |
| LLM 成本 | 每次跑完整基准集前先估算 token；开发阶段用开发子集；工具只返回摘要 |
| 自进化对开发集过拟合 | 留出集隔离；采用补丁要求无类别退化；报告时同时给出开发集和留出集曲线 |
| OpenViking 接口不成熟 | 用 `MemoryStore` 接口隔离，保底实现是 pgvector |
| 知识库语料许可证 | 只收许可证允许再分发的资料，并记录在 SOURCES.md |

---

## 15. 待决问题

1. **默认 LLM 已选 DeepSeek**（2026-10-25）：OpenAI 兼容接口、function calling；仍待配置安全凭据、授权试跑并核实可固定的实际模型版本。消融可增加其他模型做对照。
2. **Langfuse**：用 Cloud 免费档还是自托管？默认 Cloud。
3. **是否公开仓库**：建议 M2 完成后公开，届时检查 `.gitignore` 和 kb 语料的许可证。

---

## 16. 简历映射（完成后才能写，且只写实际跑出来的数字）

| 简历关键词 | 对应模块 |
|---|---|
| LangGraph 状态图、条件路由、循环纠错 | §6 |
| MCP Server / 工具设计 | §5 |
| 人工审批（HITL）、Guardrails | §6.3 |
| 会话检查点恢复、长期记忆 | §7 |
| 混合检索 + 重排、Ragas | §5.3、§9 |
| LLM-as-Judge、消融评测、Langfuse | §9 |
| 自进化（Harness2 风格）、OpenViking | §7、§8 |

---

## 17. 开发记录（2026-10-25）

### 当前进度

| 模块 | 可核验状态 |
|---|---|
| 工程与下载 | 现有 Python 3.11 / uv 配置、匿名 S3 下载器；下载器原有 4 个测试通过。开始本轮时，这些文件尚未提交，本轮保留其工作区状态 |
| 标准化与数据卡 | 新增 `normalize.py`、`datacard.py`；修正 `.gitignore`，只忽略仓库根目录 `/data/`，避免漏掉数据处理源码 |
| 2107 实际数据验证 | 原始 electrical、irradiance、environment、meter 四份 CSV；筛选 UTC `[2019-01-01, 2021-01-01)`，生成 **5,383,405** 条长表记录、24 台逆变器及电表/气象测点 |
| 基线筛查与模拟 | 已实现缺测/日间零值/同站离群筛查、F3/F4 注入、场景真值及数据划分隔离校验；已生成 F3/F4 各 5 个开发冒烟场景，逐个验证落盘结果 |
| 只读 MCP 工具 | pv-data 4 个工具、pv-analytics 5 个工具均已接入 FastMCP / Pydantic；stdio、Streamable HTTP 实测通过；真实场景完成 90 次工具协议调用。2107 缺少可核实的单机物理参数，expected_power 返回 unavailable |
| 诊断图与评分器 | 已接 LangGraph、原生 MCP 客户端、有界并行取证、结构证据校验及评分器 v1；B0 已跑完 10 个开发冒烟场景，原始报告/轨迹/指标保存在本地 data/eval/m1_smoke/ |
| M1 剩余工作 | DeepSeek 真实单场景验证及 10 场景 Agent 评测；当前 B0 数字不代表 LLM 成绩。后续还需可核实的物理参数与更完整的基线审核 |
| M2–M4 | 尚未实现；没有可报告的诊断准确率或消融结果 |

### 数据核验结论与处理边界

- 旧版 `measured_on` 使用元数据中的 **PST8PDT 当地时间**：原始辐照及电表功率的逐小时均值在约 12–13 点达到峰值，支持当地时间解释；不能把这批时间戳直接标成 UTC。
- `utc_measured_on` 按 UTC 读取。年度扩展和其他修订版默认不混入本次数据；仅靠文件名不能确认其时间语义，使用前需单独验证。
- 24 个逆变器 AC 通道及电表输出的元数据单位是 **kW**，不是累计 kWh；POA 是 **W/m²**；环境温度是 **F**，已换算为 °C。没有直接 DC 功率通道，不擅自由电压、电流构造等效值。
- 保留 NaN、零值与负值；不插补、不削峰。当地时间的夏令时歧义/不存在记录排除并在 normalization JSON 中计数；该计数覆盖所读源文件，而非只统计筛选窗口。重复键只有同值时合并，冲突值直接报错。
- 覆盖率定义为 **UTC 15 分钟 bin 内至少有一个有限观测值**，不是 NaN 比例。推荐区间要求元数据描述的全部逆变器 AC 通道和至少一个 POA 通道，每个完整月均达到 95% 覆盖率。
- 本次推荐 UTC `[2019-01-01, 2020-01-01)` 共 **12 个月**，其中最差月份、通道覆盖率为 **98.0847%**；2019–2020 全窗口逆变器通道覆盖率约 **99.9772%**，POA 约 **98.9042%**。
- **覆盖率不等于无故障**。故障注入前仍需筛查真实功率异常和可比设备；不能把自然停机样本当成正常真值。逆变器额定容量未在源元数据中显式提供，数据卡保留 null，不从型号或全站容量猜测。

### 本地复现

使用现有 `.venv`（pandas、pyarrow、httpx、pytest、ruff 已安装）；也可在已有 uv 工程环境中用 `uv run python` 替换下方 Python 命令。原始数据和生成产物均不入库。

```bash
.venv/bin/python -m pvdiag.data.download 2107 --include 2107_electrical_data.csv --include 2107_irradiance_data.csv --include 2107_environment_data.csv --include 2107_meter_15m_data.csv
.venv/bin/python -m pvdiag.data.normalize 2107 --start 2019-01-01 --end 2021-01-01
.venv/bin/python -m pvdiag.data.datacard 2107
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

产物为 `data/clean/2107.parquet`、`2107_normalization.json`、`2107_card.json`。默认仅选择 `*data.csv`；显式选择其他来源时可重复传入 `--pattern`，时区经核实后可用 `--source-timezone` 覆盖。

### 基线筛查与 F3/F4 开发场景

- 新增 `src/pvdiag/sim/baseline.py`、`faults.py`、`scenarios.py` 和模拟测试；没有新增依赖。
- 在 2019 年实测数据中，POA >200 W/m² 的 15 分钟 bin 共 **11,549** 个；其中 `inv04` 有 **2,128** 个 AC ≤0.1 kW 的 bin，`inv08` 有 **2,270** 个。这里只标为异常候选，不声称已确定真实根因。严格全年筛查没有设备通过，因此不能直接把全年当成正常真值。
- 筛查以 UTC 15 分钟均值对齐，不插补；要求 POA 和目标设备覆盖率 ≥99%、高辐照 bin ≥16、日间无缺测/近零功率，且按经验 P95 功率归一化后的独立同站中位数比值始终在 `[0.85, 1.20]` 内。P95 不是额定容量；恒定的既有损失仍可能漏检。
- 从 **2019 年上半年、inv01–inv12** 的开发范围扫描不重叠的 7 天窗口，选取并再次筛查三台设备组成的队列；每个窗口分别生成 F3、F4 单故障场景。观测文件只含该队列的 AC 功率、指定 POA、环境温度和队列内组件温度，不暴露未选设备或不一致的原站电表总功率。
- F3：在 `[start, end)` 内，同一 UTC 15 分钟 bin 的指定 POA 均值严格 >200 W/m² 时，把目标 AC 功率置零；POA 缺测则报错。F4：目标 AC 功率阶跃乘以 `1−1/8` 或 `1−1/4`。NaN 保留，输入不修改，不模拟 DC 和电气动态；无实际变化的注入会被拒绝。
- 场景真值在 `bench/m1_smoke.json`；每个场景的 `timeseries.parquet`、`truth.json`、`audit.json` 在忽略的 `data/scenarios/<scenario_id>/` 下。诊断端只能读取观测文件，不能读取真值、基线筛查统计或注入审计。
- 清单记录源 Parquet 的 SHA-256、筛查版本和阈值。本次 10 个场景已独立核对字段、设备/时间范围、注入公式与变化行数；在临时目录重新生成后，清单和全部 10 份 Parquet 的 SHA-256 一致。现有 **40 个测试通过**，Ruff 检查通过。
- **这些只是开发冒烟数据，真值仅覆盖注入故障**：`truth_scope=injected_faults_only`、`baseline_screen.status=screened_not_manually_verified`。没有生成 N 正常标签，没有运行诊断评分或留出集评测。完整基准集前仍需审核基线，不能据此报告正常误报率。
- 划分校验同时检查开发/留出集的查询时间与全部可见逆变器（包括对照设备）不相交；跨站系统不能混入同站开发/留出集。本轮清单全部为 `dev`，尚未生成留出集。

复现本次 10 个场景：

```bash
.venv/bin/python -m pvdiag.sim.scenarios 2107 \
  --start 2019-01-01 --end 2019-07-01 \
  --devices inv01 inv02 inv03 inv04 inv05 inv06 inv07 inv08 inv09 inv10 inv11 inv12 \
  --poa-device sensor_149574
```

单独筛查可用 `python -m pvdiag.sim.baseline 2107 --start ... --end ... --devices ... --poa-device sensor_149574`，报告写入本地 `data/clean/2107_baseline_screen.json`。

### 只读 MCP 工具层

- 新增 `src/pvdiag/mcp_servers/`：`models.py` 定义 Pydantic 入参/出参，`store.py` 加载只读观测快照，`data_tools.py` / `analytics_tools.py` 实现计算，`pv_data.py` / `pv_analytics.py` 提供独立服务入口，`cli.py` 统一启动参数。
- 安装并验证 FastMCP **4.1.3**、Pydantic **2.14.1**、pvlib **0.16.1**。已在现有 uv 配置中增加相应依赖并更新锁文件；`uv lock --check` 通过。`pyproject.toml` / `uv.lock` 开始本轮时尚未跟踪，本轮继续留在工作区，避免把原有配置内容混入代码提交。
- 仅加载显式指定的观测 Parquet、站点元数据与可选物理参数 JSON；不读取 `truth.json`、`audit.json` 或场景清单。模型只能看到快照中存在的设备；输出不包含源路径或可能含故障标签的场景 ID。
- 每份工具结果包含稳定 `evidence_id`、工具版本、请求参数和观测/元数据/模型参数共同构成的 `source_sha256`。进程启动后使用内存快照，更新文件需重启服务。9 个工具均声明 read-only / non-destructive / idempotent，不加载 pv-ops。
- 时间窗使用 UTC 半开区间 `[start, end)`，最长 90 天，边界对齐 15 分钟。`get_timeseries` 返回重采样统计和全响应最多 240 个采样点，分桶从请求起点开始；缺失 metric 返回零覆盖率和 null 数值，不伪造零功率。质量报告的三类片段共享 64 段预算，按设备及类别分配，截断显式标记。
- 同站对比使用**独立于目标设备的至少两台设备**，先算目标/同站中位数，再除以明确参考时段的比值中位数；至少 4 个参考日间点才返回归一化比值。不能用整段故障后的数据重新定标。实际第 2 组场景的前 48 小时缺少足够高辐照点，工具因此返回 null；Agent 规划时必须检查参考时段有效性。
- `detect_anomalies` 当前实现滚动 z-score（前 12 个有效日间点，至少 4 点，标准差下限 0.01，阈值 4），另报告缺测段和高辐照零值段。没有实现 Chronos 或独立变点检测；持续损失的完整区间应结合 `peer_compare`，不能把 z-score 不再触发当作恢复。
- `profile_pattern` 提供有效日间零值比例、相对功率中位数/IQR、日趋势及当地小时画像；形态是描述性规则，不是根因结论。`compare_periods` 对不重叠窗口统计已观测 15 分钟均值积分、覆盖率和相对性能，缺测不补零，发电量差异仍受天气与时长影响。
- `expected_power` 实现 pvlib PVWatts DC/AC + Faiman 温度模型。仅接受显式提供且注明来源的 `rated_dc_kw`、`rated_ac_kw`、`gamma_pdc`；默认风速 1 m/s、Faiman 默认系数、逆变效率 0.96 均列为假设。返回逐设备有限长度 PR 序列、偏差与可比高辐照点积分；缺物理参数或缺可比点时返回 unavailable。当前 2107 全部未配置这组参数，不报告虚构的理论 PR。
- 已验证 10 场景 × 9 工具的 **90 次 MCP 协议调用**；在明确的故障前参考时段下，5 个真实 F4 场景的相对比值均与注入比例相差小于 0.02。这只是已知时间段上的工具数值核对，不是端到端诊断或准确率评测。
- 当前 **62 个测试通过**，涵盖只读范围、非法请求、缺测/零值、参考时段不足、F4 比值、形态、物理参数缺失/AC 削顶、序列限长、证据稳定性，以及真实 stdio/HTTP 子进程调用；Ruff 通过。下一步实现 LangGraph 的规划/取证/报告及固定评分器，然后评测完整诊断流程。

独立启动（客户端通过 MCP 管理 stdio 进程；两台服务须使用同一份观测快照）：

```bash
.venv/bin/python -m pvdiag.mcp_servers.pv_data \
  --observations data/scenarios/S2107-F4-0001/timeseries.parquet \
  --metadata data/raw/2107/metadata/2107_system_metadata.json
.venv/bin/python -m pvdiag.mcp_servers.pv_analytics \
  --observations data/scenarios/S2107-F4-0001/timeseries.parquet \
  --metadata data/raw/2107/metadata/2107_system_metadata.json
```

加 `--transport http` 可在 `127.0.0.1` 启动 Streamable HTTP，默认端口分别为 8001 / 8002，端点 `/mcp`。调用参数使用 `{"request": {...}}`。物理参数可通过 `--parameters <path>` 传入，结构为 `{"inv03": {"rated_dc_kw": ..., "rated_ac_kw": ..., "gamma_pdc": ..., "source": "可核验出处"}}`；不能将示例省略号或猜测值当成实际配置。

### LangGraph 主流程、DeepSeek 接口与评分器

- 新增 `src/pvdiag/agent/`，状态定义、范围检查、MCP 网关、LangGraph、B0 reasoner、LLM reasoner 和单次诊断 CLI 分开；`harness/system_prompt.md` 作为版本化提示词。图为 `guard_in → plan → investigate → hypothesize → verify → report → guard_out`，结构校验失败或计划还需取证时回到 plan，最多三轮、默认最多三十次工具调用。轮次或预算耗尽仍不足时降级人工复核。
- LangGraph **1.2.17**、LangChain **1.4.7**、langchain-openai **1.7.2** 已安装。**验证出的兼容性差异**：PyPI 最新 langchain-mcp-adapters **0.3.2** 要求 `mcp>=1.24,<2`，FastMCP **4.1.3** 要求 `mcp>=2.2,<3`，uv 无法同时解析。当前使用 FastMCP 原生 Client 连接两个服务，保持 MCP 协议和工具 schema；未强装冲突依赖。
- 调用前校验节点工具白名单、Pydantic 参数、系统/设备/传感器和查询/参考时间范围；响应校验 schema、实际请求、稳定 evidence_id 哈希及两台服务的观测快照指纹。模型看不到原始 Parquet、场景 ID、基线筛查结果或注入真值。远端错误只保留类型，不把可能含凭据或路径的异常文本传入报告。
- 报告采用定性置信度。每条结论要求至少两个不同的非元数据工具引用；数值通过 JSON Pointer 精确溯源，时间边界也须对应已有证据时间戳。定性摘要/理由禁止自由嵌入数字；结构核验失败会重试或清除不可信结论，返回 needs_review。**该核验不等于语义忠实度评测，也尚未实现完整的混淆因素排除、自评、知识检索或写操作审批**。
- 当前使用进程内 InMemorySaver，已验证检查点保存和读取；没有声称实现跨进程持久化或 PostgresSaver。记忆、pv-ops、审批、中长期存储和服务化 API 仍属于后续工作。
- B0 用纯代码从辐照摘要选择有足够日间点的参考期，再结合同站低值段/高辐照零值段、形态及分期对比生成可溯源假设，仅覆盖当前 F3/F4；其他情况保守输出 inconclusive，不能认定正常。B0 与 LLM reasoner 共享图及工具协议，但运行元数据明确区分模式。
- DeepSeek 接口通过 `init_chat_model` 和 `with_structured_output(method="function_calling")` 接入，记录实际返回模型名、输入/输出 token 和缺失用量标记。每次输出最多 4096 token、禁用隐式自动重试、请求有超时与载荷字符上限。已完成真实 SDK 的 schema 构造及注入式离线模型测试，**没有发起收费调用**。调用须显式 `--allow-model-calls`；凭据从 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` 环境变量或忽略的本地 `.env` 读取，不输出、不入库。仅凭模型别名仍无法证明正式模型版本固定。
- 新增 `src/pvdiag/eval/scorer.py`（v1）与 `run_bench.py`。评分器为纯代码并记录源码 SHA-256：场景检测 P/R/F1、根因 Top-1/Top-3、设备定位、根因与设备联合匹配、同设备同根因的时间 IoU 命中、拒答、数值引用、工具成功率和用量。双故障使用最大一一匹配并惩罚多报；根因与设备各自正确但对应错误时联合指标为错。只有正常样本才计算误报率，缺失 token 用量标为未知，不填零；LLM-as-Judge 与价格未接入时保持 null。
- 评分仅在诊断结束后读取真值，诊断图只接收公开的系统、设备和查询窗。当前 runner 只允许 `dev`，留出集在打开观测前即拒绝，避免意外消耗留出评测次数。
- 当前 **84 个测试通过**，Ruff、格式与 `uv lock --check` 通过；新增测试涵盖真实 LangGraph/MCP 诊断、凭空数值/时间/设备拒绝、越界调用在远端执行前拒绝、证据篡改拒绝、重试修正、预算耗尽、检查点读取、离线结构模型接入、模型错误脱敏，以及评分器的手算 IoU、多故障匹配、正常拒答和缺失用量处理。

**B0 · 十个开发冒烟场景实跑结果（2026-10-25）**

| 指标 | 结果 |
|---|---|
| 场景检测 P / R / F1 | 1.0 / 1.0 / 1.0 |
| 根因 Top-1 / Top-3、设备及联合匹配 | 各 10/10 |
| 时间 IoU ≥0.5 | 9/10 |
| 数值引用匹配率、工具成功率 | 各 100% |
| 平均工具调用数、取证轮数 | 每场景 8 次、3 轮 |
| 未审批写操作执行数 | 0（未加载写工具） |
| LLM 调用及 token | 0（纯规则） |
| 正常误报率、语义 Judge、美元成本 | null；未评测/未定价，不能当作零 |

唯一时间窗未达标的是 `S2107-F4-0001`：真实注入窗口为 UTC `[2019-01-11, 2019-01-15)`，规则仅从可见低值段定位到 `[2019-01-12 18:00, 2019-01-13 22:45)`，未为了该样本修改评分标准。上述结果仅为理想化注入、自动筛查基线上的开发冒烟，不能视为真实生产准确率、留出泛化或 DeepSeek 成绩。

复现 B0（不会调用外部模型）：

```bash
.venv/bin/python -m pvdiag.eval.run_bench --mode b0
```

单次诊断也可用 `python -m pvdiag.agent 2107 --observations <parquet> --metadata <json> --devices <可见逆变器...> --start <UTC> --end <UTC>`。未来真实 DeepSeek 试跑须在本机安全配置凭据并明确授权费用后，加 `--mode llm --allow-model-calls`；不要把密钥粘贴进聊天或命令记录。下一步验证真实 DeepSeek 调用，再推进六类故障、正常场景、核验/知识库与 M2 消融；M1 尚未按 LLM 端到端标准验收。
