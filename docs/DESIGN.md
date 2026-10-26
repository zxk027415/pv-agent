# PV Diagnosis Agent · 设计文档

> 版本：v0.1（2026-10-24）· 进度更新：2026-10-26，规则诊断、BGE 检索、PostgresSaver、FastAPI/SSE/身份绑定审批与 Gradio 单页已验证；已补基线审核、rule-v2 六类开发对比及 A0–A3 消融执行边界，真实 LLM 消融与 Ragas 尚未完成，DeepSeek 单例已授权但缺本机凭据
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
| 留出集 | 2107 · 6 类 × 10 + 正常 20 = 80 个场景（时间、全部可见逆变器和故障目标均与开发集不相交） | 报告主指标 |
| 跨站集 | 9069 · 6 类 × 5 + 正常 10 = 40 个场景 | 泛化 |

**设备隔离补充（2026-10-26，用户已明确）**：严格隔离包含 F5 的 POA 故障目标；逆变器对照设备也计入隔离范围。非故障目标的气象共变量作为观测上下文，不据此声称传感器跨设备泛化。当前 2107 仅有 `sensor_149574` 一个 POA 通道，因此 F5 留出集等待独立传感器数据，不以共享 POA 的跨时间测试替代跨设备结论。

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
- **实现状态（2026-10-26）**：FastAPI 已加入 `GET /profiles`、`GET /runs/{id}/events` 与 `POST /runs/{id}/resume`，使用服务端数据白名单、开发 Bearer 身份、角色与归属检查，支持持久 SSE 重放和审批先落库。配置和使用方法见 §17 末节；尚未公开部署。
- **前端（2026-10-26 已实现本机开发版）**：Gradio 单页接入诊断输入、节点流式过程、UTC 证据图表与身份绑定审批；独立 UI 虚拟环境通过 HTTP 连接 API，配置与验证见 §17 末节。
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
│   ├── api/                  # FastAPI + SSE
│   └── ui/                   # Gradio HTTP 客户端，独立虚拟环境
├── harness/                  # system_prompt.md · playbooks/*.md · tool_descriptions.yaml
├── kb/                       # 知识库语料 + SOURCES.md
├── bench/                    # 场景 JSON（由脚本生成，可以入库）
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

1. **默认 LLM 已选 DeepSeek**（2026-10-25）：OpenAI 兼容接口、function calling；用户已授权一个开发场景、最多 6 次模型调用、每次最多 4096 输出 token。当前仅缺本机安全凭据，尚未消耗授权；试跑后仍需核实可固定的实际模型版本。扩大评测范围需另行授权。
2. **Langfuse**：用 Cloud 免费档还是自托管？默认 Cloud。
3. **是否公开仓库**：建议 M2 完成后公开，届时检查 `.gitignore` 和 kb 语料的许可证。
4. **F5 留出设备隔离已确认**（2026-10-26）：保持严格故障目标隔离，等待补齐独立 POA 传感器数据；当前不生成或报告 F5 跨设备留出成绩。

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

## 17. 开发记录（更新至 2026-10-26）

### 当前进度

| 模块 | 可核验状态 |
|---|---|
| 工程与下载 | 现有 Python 3.11 / uv 配置、匿名 S3 下载器；下载器原有 4 个测试通过。开始本轮时，这些文件尚未提交，本轮保留其工作区状态 |
| 标准化与数据卡 | 新增 `normalize.py`、`datacard.py`；修正 `.gitignore`，只忽略仓库根目录 `/data/`，避免漏掉数据处理源码 |
| 2107 实际数据验证 | 原始 electrical、irradiance、environment、meter 四份 CSV；筛选 UTC `[2019-01-01, 2021-01-01)`，生成 **5,383,405** 条长表记录、24 台逆变器及电表/气象测点 |
| 基线筛查与模拟 | 六类注入及双故障、真值与数据划分校验已实现，划分现在包含 F5 传感器目标；80 个 M2 候选已逐个重建核对并绑定 20 份原始基线审核包，实际人工接受数仍为零，未注入候选不能作为正常真值 |
| 只读 MCP 工具 | pv-data 4 个工具、pv-analytics 5 个工具均已接入 FastMCP / Pydantic；stdio、Streamable HTTP 实测通过；真实场景完成 90 次工具协议调用。2107 缺少可核实的单机物理参数，expected_power 返回 unavailable |
| 诊断图与评分器 | 已接 LangGraph、有界 MCP 取证、结构/领域核验及离线自评接口；评分器 v1.2 新增知识只读分类和审批记录一致性检查，B0 rule-v2 的 M1 十场景仍为根因/定位 10/10、时间窗 9/10；同输入 M2 故障子集根因/定位 33/60、时间窗 32/60；新增积灰恢复、遮挡与采集缺失候选规则，短窗 F1 与 F5 仍未命中 |
| 知识库 | 已核对许可并收录 5 份 pvlib 文档、6 份项目 Playbook；两项 MCP 工具、可选图接入和 30 题 BM25/BGE 对比已实跑。混合检索文档 Recall@3 为 1.0、MRR 为 0.9833；自编标签、不是 Ragas 或 A3 成绩 |
| M1 剩余工作 | DeepSeek 真实单场景验证及 10 场景 Agent 评测；当前 B0 数字不代表 LLM 成绩。后续还需可核实的物理参数与更完整的基线审核 |
| 人工审批与模拟工单 | `interrupt` / `Command(resume=...)` 和本地 CLI 已接通；工单默认关闭，SQLite 持久化具体草案、人工决定和幂等工单，拒绝/篡改/重放路径已测试 |
| 诊断持久化 | 已接真实 AsyncPostgresSaver、显式 schema 初始化、start/resume/show CLI、版本/数据绑定、活动锁及独立模型预算账本；受控审批已在不同 Python 进程间恢复，实际 M1 十场景保存/重开结果一致 |
| 服务化 | FastAPI/SSE、服务端 profile 白名单、开发身份/角色、请求归属、幂等诊断、持久事件重放及身份绑定审批已验证；默认仅监听本机，Gradio 接入见下节，尚无生产身份系统 |
| M2 | 六类原语、短窗候选、领域核验、混合检索、审批及检查点/API 恢复已实现；正式正常/留出基准集、Ragas、Langfuse 与完整 LLM 消融尚未完成 |
| 消融执行边界 | 已接 A0–A3 显式档位、统一固定摘要、网关工具白名单、隔离核验反馈、共享模型预算与恢复契约；271 项测试通过，新增 30 项离线边界/评测/真实 PostgreSQL 检查，不代表真实模型消融成绩 |
| M3–M4 | M4 的 Gradio 本机单页已接通；长期记忆、自进化、Docker Compose、README 和完整演示打包仍未完成，没有进化、泛化或完整消融结果 |

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
- 报告采用定性置信度。每条结论要求至少两个不同的非元数据观测工具引用；数值通过 JSON Pointer 精确溯源，时间边界也须对应已有观测证据时间戳。定性摘要/理由禁止自由嵌入数字；结构核验失败会重试或清除不可信结论，返回 needs_review。默认仍为结构核验；可选领域核验和模型自评见后文。**这些检查不等于独立语义忠实度评测，完整混淆因素排除和 Ragas 仍未完成**。
- 默认单次诊断仍用 InMemorySaver；后续已加入可选 PostgresSaver 跨进程恢复和独立模型预算账本，详见末节。SQLite 保存本地工单及人工决定，PostgreSQL 保存诊断检查点与模型用量，两者职责不同。长期记忆仍待开发；审批 HTTP API 与服务化的后续实现见末节。
- B0 用纯代码从辐照摘要选择有足够日间点的参考期，再结合同站低值段/高辐照零值段、形态及分期对比生成可溯源假设，该早期 rule-v1 仅覆盖 F3/F4，rule-v2 后续扩展见末节；证据不足时保守输出 inconclusive，不能认定正常。B0 与 LLM reasoner 共享图及工具协议，但运行元数据明确区分模式。
- DeepSeek 接口通过 `init_chat_model` 和 `with_structured_output(method="function_calling")` 接入，记录实际返回模型名、输入/输出 token 和缺失用量标记。每次输出最多 4096 token、禁用隐式自动重试、请求有超时与载荷字符上限。已完成真实 SDK 的 schema 构造及注入式离线模型测试，**没有发起收费调用**。调用须显式 `--allow-model-calls`；DeepSeek 默认只读取 `DEEPSEEK_API_KEY`，OpenAI 只读取 `OPENAI_API_KEY`，其他端点默认读取 `PVDIAG_LLM_API_KEY`，可用 `--credential-env` 显式指定字段；从环境或忽略的本地 `.env` 加载，不输出、不入库，不自动跨提供方回退。仅凭模型别名仍无法证明正式模型版本固定。
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

单次诊断也可用 `python -m pvdiag.agent 2107 --observations <parquet> --metadata <json> --devices <可见逆变器...> --start <UTC> --end <UTC>`。DeepSeek 单场景试跑已获授权，凭据就绪后使用 `--mode llm --allow-model-calls --max-model-calls 6 --verification structural`；不要把密钥粘贴进聊天或命令记录，不自动扩大为十场景评测。M1 尚未按 LLM 端到端标准验收。

### 六类故障原语与 M2 开发候选

- `sim/faults.py` 已补齐 F1/F2/F5/F6。F1 按实际经过天数线性衰减，速率固定在设计的 `0.001–0.005/天`，指定清洗时刻恢复并重新计时，不插补 NaN；拒绝非正功率乘数。F2 按**显式站点时区**的每日半开时段衰减，严重度 `0.3–0.7`，支持跨午夜及夏令时。F5 仅修改目标 POA 通道，正漂移率限制为 `0–0.02/天`（不含零），AC 功率保持不变；F6 保留时间戳及行数，把目标设备在时间段内的全部测点置 NaN，包括夜间零功率和组件温度。
- `Scenario` 支持无注入候选、六类单故障及仅 hard 档的双故障。F5 真值指向指定 POA 传感器，其余故障指向可见逆变器。双故障按物理变化 → 传感器漂移 → 通讯丢失执行，因此停机判据使用漂移前辐照；用去掉单个故障的反事实结果检查最终可观察贡献，拒绝完全被另一个故障遮蔽的标签。
- 正常候选**只表示未注入**：`faults=[]`、`truth_scope=unmodified_baseline_candidate`、`evaluation_eligible=false`。未注入不等于无自然故障，评测 runner 在读取观测前拒绝这类候选，不能以此给出正式正常误报率。基线仍需独立审核。
- 新增 `sim/benchmark.py` 静态开发候选生成器，生成版本为 `m2-v1`，默认 35 天窗口；M3 依据失败分布生成场景的 `scenario_gen.py` 尚未实现。当前只生成 dev，包含全部可见对照设备的时间/设备隔离检查沿用原逻辑，没有生成或评测留出集。
- 为检查慢变数据可行性，使用已下载原始文件额外清洗了 2018 年及 2021–2022 年数据，存入忽略的 `data/clean/dev_2018/`、`data/clean/dev_extended/`。在当前 inv01–inv12 候选范围与严格筛查下，这些时段及 2019 年没有合格的 35 天队列；2019 年有 17 个首选 7 天片段。逐年日内峰值大致对齐，尚无明确整段时差证据；同时零值可能来自真实停机或采集问题，未擅自修正原始零值或放宽筛查凑数。
- 本次使用**显式 7 天窗口**在 2019 年生成 **80 个开发候选**：F1–F6 各 10 个，加 20 个不同设备/时间范围的未注入候选。短窗口中的积灰和漂移效应较弱，统一标为 hard，没有放大物理速率使之易检出。**这不是已经验收的 M2 正式基准集**：长窗口慢变数据、正常真值与多档难度仍待完善。
- 清单位于 `bench/m2_dev_candidates.json`，观测与真值分离落盘在 `data/m2_candidates/<id>/`；没有覆盖 M1 文件。已逐个独立核对 80 份观测的公式、作用范围、字段、变化行数及未改动对照；临时目录重新生成的清单与全部 80 个 Parquet SHA-256 一致。新增四类真实数据均通过 MCP 质量/同站工具调用验证；没有据此报告新的 Agent 分类成绩。
- 输出保护和评分器修复了 F5 的定位约定：必须指向快照中有 POA 测点的传感器，而非强制当作逆变器。评分器升级为 **v1.1**，版本/源码哈希显式变化，这是新增领域支持的工程变更，不是 harness 自进化修改评分规则。旧 v1 的结果保留；v1.1 在独立目录 `data/eval/m1_scorer_v1_1_regression/` 复验 M1，仍为根因/定位 10/10、时间窗 9/10，其他匹配规则未变。
- 当前 **103 个测试通过**，Ruff、格式与锁文件检查通过。新增验证覆盖清洗重置、时区/夏令时/跨日遮挡、漂移不改功率、整设备缺测、无操作拒绝、双故障因果顺序/掩蔽拒绝、候选确定性与正常真值隔离。旧 M1 清单和十份观测重新生成后仍逐字节一致。

复现本次短窗候选（不运行诊断、不调用模型）：

```bash
.venv/bin/python -m pvdiag.sim.benchmark 2107 \
  --source data/clean/2107.parquet --start 2019-01-01 --end 2020-01-01 \
  --devices inv01 inv02 inv03 inv04 inv05 inv06 inv07 inv08 inv09 inv10 inv11 inv12 \
  --poa-device sensor_149574 --timezone PST8PDT --window-days 7 \
  --per-type 10 --normal-count 20
```

后续继续正常基线审核、长窗口数据和六类判别工具/规则；知识库与本地审批的后续进度见下文。DeepSeek 单场景调用授权已取得，目前等待本机安全凭据。

### M2 领域核验、自评接口与调用预算

- 新增 `agent/verification.py` 和 `harness/verification_prompt.md`，支持 `--verification structural|domain|domain_llm`。默认 `structural` 保持原结构检查；`domain` 叠加六类必要证据检查；`domain_llm` 再调用模型自评。问题及建议工具反馈到下一轮规划，最终输出守卫再次检查领域证据。该批实现时尚未接知识库；后续 BM25 接入见下文，完整 A3 仍未实评。
- 领域条件：F1 需多日缓降及清洗恢复；F2 需多日重复窄时段损失与时段外恢复；F3 需高辐照零值及同站近零；F4 需非零稳定比例损失并排除窄时段遮挡；F5 需 POA 质量、稳定同站关系和使用同参数计算的前后共同物理性能下降与多日趋势；F6 需目标全部可见测点缺失及至少两台同期有效对照。`expected_power=unavailable` 不能充当 F5 支持。
- 正常结论需完整质量、同站对比、异常排查及独立物理性能支持；不能省略已发现异常，不能以正常平均值覆盖共同下降趋势。以上是**保守的必要条件，不是因果证明或独立语义 Judge**，不能证明已排除所有真实混淆因素。
- `profile_pattern` 新增重复损失小时数、时段外中位比值、最大日恢复阶跃、恢复时刻和恢复前日斜率。重复小时须跨日出现，时段外统计包含这些小时之外的全部样本；恢复按 UTC 日中位数计算，恢复前斜率至少需三日。证据版本升为 `m2-v1`，其他工具版本保持原值。
- `RunResult` 记录模式、每轮核验问题、检查过的证据、是否执行自评、报告状态及定论准备状态，并记录 guards/verification 源码的 `verification_policy_sha256`；自评模式的 harness 哈希包含自评提示词。模型判定通过不能覆盖规则失败；伪造自评引用、自评失败及预算不足均不能视为通过。
- 规划、诊断和自评共享 `--max-model-calls` 硬预算，默认 6，在请求发出前检查；单次输出上限仍为 4096 token，SDK 自动重试为零。启用自评不会自动提高预算，耗尽则保守返回 needs_review。DeepSeek 单例仍选结构模式，避免把新增自评挤入此次授权的预算。
- 本批全套 **123 个测试通过**，收尾时重跑受影响的 **65 个测试亦通过**；Ruff、本轮 11 个 Python 文件的格式检查、`uv lock --check` 和差异空白检查通过。全仓格式检查仍发现原有未跟踪 `data/download.py` 的格式问题，未改动或纳入本批提交。新增测试以真实工具计算受控数据，覆盖六类支持、错误标签/窗口/来源、正常漏报、自评不能覆盖规则、假自评引用、反馈后重规划修正、共享模型预算与凭据端点隔离；模型自评仅使用离线替身，尚未实调。
- 已在真实 M1 十场景分别运行结构和领域模式：结果保存在 `data/eval/m1_structural_verification_regression/` 与 `data/eval/m1_domain_verification/`，两组根因/定位均 10/10、时间 IoU 达标 9/10，每场景 8 次工具调用。评分器仍为 v1.1，SHA-256 为 `64dd580859cd91893f654152565596338ba8d66e229759653c520e0298be8364`；没有调整评分标准，仍是 B0 成绩。
- 真实短窗候选的已知假设组件检查记录于 `data/m2_known_window_verification.json`：F1 恢复证据不足、F2 重复时段证据不足、F5 物理/趋势证据不足，F6 通过。检查已知故障窗口不等于盲诊，不报告为六类 Agent 准确率；正常正式基准和留出集仍未就绪。
- 最近检查未发现本机 `DEEPSEEK_API_KEY`，也没有 `data/runs/deepseek_authorized_pilot.json` 收据；因此真实请求为零，单场景授权尚未消耗。凭据就绪后仅执行既定 M1 单例，将报告、实际返回模型名及 token 用量写入该收据；不得自动重开预算或运行第二个场景。

### 知识库语料、Playbook 与检索开发基线

- `kb/SOURCES.md` 记录首批逐项许可核对。实际语料来自已安装 `pvlib==0.16.1` 的五个函数完整文档字符串，保留 BSD-3-Clause 版权、条件及免责声明；原始模块、收录内容及许可分别记录 SHA-256。未复制引用论文的正文，也未声称本轮已联网核对上游 tag 与发行包完全相同。`knowledge/build_corpus.py` 可在已有环境中离线、逐字节重建，只接受已核对版本与许可哈希。
- `playbooks/F1.md` 至 `F6.md` 使用 SKILL.md frontmatter 格式，包含取证、混淆、无法判定边界与人工核实建议。它们由项目编写，不是独立专家标注或已确诊案例，不执行操作。manifest 区分第三方 BSD 语料与 `project-authored` 流程；项目整体开源许可仍未确定。
- `KnowledgeStore` 只读 manifest 指定的固定目录，验证内容/许可哈希、大小、路径及重复项；不读取 `data/`、`bench/` 或诊断真值。启动后保持内容快照。知识指纹由语料清单、检索配置和检索实现源码哈希共同决定，与观测指纹分开。
- 新增 `pv-knowledge` 的 `search_fault_kb`、`get_playbook` 两个 Pydantic/FastMCP 只读工具，默认 stdio，可通过本机 8003 端口提供 HTTP。Agent 默认仍只挂载两台观测服务，显式 `--knowledge-root kb` 才启用知识服务；网关在连接时固定知识指纹并逐响应验证。运行结果及评测摘要记录 `knowledge_source_sha256`。
- 知识内容仅作为机理参考：两个知识工具不能满足报告的两个观测工具要求，检索分数不能作为设备测量值，文档日期不能作为故障时间。已更新 system prompt，并以结构守卫和测试落实这些边界；没有把检索到 Playbook 当作写操作权限。
- BM25 支持拉丁词和中文单字/双字，查询求和顺序固定，跨进程证据 ID 一致。`hybrid_bge` 路径实现向量余弦检索、BM25/RRF 候选融合与本地 CrossEncoder 重排；模型目录内容哈希和运行时版本会进入配置，仅允许本地加载、禁用远程代码。该批实现时仅完成 sentence-transformers **v5.1.2** 的官方接口源码核对与替身测试；本轮真实模型准备及实测见下节。缺失依赖或模型时显式失败，不静默回退 BM25 或调用在线嵌入服务。
- 新增 `bench/kb_qa_dev.json` 的 **30 题**开发问答和 `eval/retrieval.py`。问题覆盖六类故障边界及物理模型假设，逐题记录参考回答和相关文档 ID；参考回答/标签不传入检索器。当前标签由项目编写，尚无独立专家复核。BM25、k=3 实跑的宏平均文档 Precision@3 为 **0.3556**、Recall@3 为 **0.95**、MRR 为 **0.9111**，原始结果在 `data/eval/kb_bm25_dev.json`。两项未完整召回涉及中文 Faiman 输入/系数问题，英文原文未进入前三；保留这些失败，不据此修改标签凑数。以上是文档 ID 重叠指标，**不是 Ragas context precision/recall、回答忠实度或故障准确率**。
- 本批已验证知识工具的真实 stdio 与本机 HTTP 调用，HTTP 收据在 `data/runs/kb_http_smoke.json`；受控数据跑通“图规划 → 知识/观测并行取证 → 领域核验 → 报告”。真实 M1 在显式挂载知识服务的领域模式下回归，仍为根因/定位 10/10、时间窗 9/10、每场景 8 次工具调用，结果在 `data/eval/m1_knowledge_mount_regression/`。B0 不使用知识，因此该回归只证明挂载兼容，不能证明知识对诊断有提升，也不能算 A3 成绩；评分器 v1.1 未改。
- 当前全套 **149 个测试通过**，Ruff、本轮 14 个 Python 文件的格式检查、`uv lock --check` 和差异空白检查通过。新增 26 个测试覆盖逐字节重建、双语检索/空结果、来源与许可、不可变快照、路径/符号链接越界、篡改与重复项、MCP/图接入、知识指纹隔离、知识不得替代测量或时间边界、融合/重排错误关闭，以及检索指标手算与标签隔离。全仓格式检查的原有下载器问题仍未改动；真实 BGE 尚未验证。

复现已验证的开发链路（不调用外部模型）：

```bash
.venv/bin/python -m pvdiag.knowledge.build_corpus
.venv/bin/python -m pvdiag.mcp_servers.pv_knowledge --transport http
.venv/bin/python -m pvdiag.eval.retrieval --knowledge-root kb
.venv/bin/python -m pvdiag.eval.run_bench --mode b0 --verification domain \
  --knowledge-root kb --output data/eval/m1_knowledge_mount_regression
```

后续仍需 Ragas、独立语料/标签审核，并结合真实 LLM 评测完整 A3；审批 API、正式正常/留出基准和其余消融仍待开发。本地审批及持久化进度见后文。DeepSeek 单例授权不扩展为检索 Judge 或批量诊断调用。

### 真实本地 BGE 准备与混合检索对比

- 已核对两款模型固定版本的模型卡及其链接的完整 MIT 许可，详细来源和 SHA-256 记录在 `kb/SOURCES.md`、`kb/model_assets.json`，许可文本保存在 `kb/licenses/FlagEmbedding-LICENSE.txt`。嵌入为 `BAAI/bge-small-zh-v1.5@7999e1d3359715c523056ef9478215996d62a620`，重排为 `BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`；没有使用可移动别名代替版本。
- 新增 `knowledge/model_assets.py`：只下载清单指定文件，流式计算大小/哈希，完成校验后才发布文件；失败清理本次临时文件，保留已完成文件以供重试。已有内容不符时拒绝覆盖。每个模型目录保留卡片、完整 MIT 许可和版本清单收据；加载时再次校验。实际下载共 **1,230,814,896 字节**，位于忽略的 `data/models/`，不入 Git，也没有调用收费模型服务。
- 可选运行环境通过 `requirements/knowledge.in` 和 macOS arm64 / Python 3.11 的 `requirements/knowledge-macos-arm64.txt` 锁定，实际安装 sentence-transformers **5.1.2**、Transformers **4.57.6**、PyTorch **2.11.0**；原有未跟踪 `pyproject.toml` / `uv.lock` 未改动，可选包尚未纳入核心 uv 配置，Linux 环境未实测。
- 实际使用 CPU、float32、4 个 CPU 线程、512 维嵌入，embedding/reranker 最大长度均为 512 token，批大小分别为 8/4；更长文档受模型截断限制。嵌入查询使用 BGE 中文检索前缀，重排保留原始 logits，不能解释成概率。模型文件指纹、固定清单、依赖版本和上述配置记录到检索元数据。
- 在 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 下完成真实混合检索及 stdio MCP 两项工具调用，收据在 `data/runs/kb_bge_stdio_smoke.json`，返回知识指纹与评测一致。未读取场景真值、未生成回答或运行 Judge，也没有消耗 DeepSeek 单场景授权。
- 对同一份 30 题与标签运行两种模式，QA SHA-256 相同，未针对结果修改问题、参考回答或相关文档标签：

| 文档检索指标（k=3） | BM25 | BM25 + 向量/RRF + BGE 重排 |
|---|---:|---:|
| 宏平均 Precision@3 | 0.3556 | 0.3778 |
| 宏平均 Recall@3 | 0.9500 | 1.0000 |
| MRR | 0.9111 | 0.9833 |
| 30 次查询总耗时 | 0.017 秒 | 25.516 秒 |

- 原始结果分别在 `data/eval/kb_bm25_bge_comparison.json`、`data/eval/kb_bge_dev.json`；新增逐题查询耗时和快照加载耗时。混合模式此次快照加载约 **3.655 秒**，不声称是严格冷启动测量。两道中文 Faiman 问题均完整召回，但 KB028 的首个标注文档仍在第二名。语料只有 **11 份**，全部进入最多 20 个重排候选，因此当前对比主要反映重排收益，不能单独证明向量召回或融合阶段的增益。
- 当前全套 **162 个测试通过**，新增 13 个覆盖下载幂等、损坏/超量/网络失败、错误收据/许可/权重、路径/符号链接越界、非许可仓库和 pickle/代码文件拒绝。真实模型运行验证独立于替身测试完成。仍无独立专家标签、Ragas、生成回答忠实度或完整 A3 诊断收益结论。

复现本地模型准备与离线检索：

```bash
uv pip install --python .venv/bin/python -r requirements/knowledge-macos-arm64.txt
.venv/bin/python -m pvdiag.knowledge.model_assets
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pvdiag.eval.retrieval \
  --retrieval-mode hybrid_bge \
  --embedding-model-path data/models/bge-small-zh-v1.5 \
  --reranker-model-path data/models/bge-reranker-base \
  --output data/eval/kb_bge_dev.json
```

### 人工审批、SQLite 工单台账与写操作审计

- 新增 `ops/store.py`、`ops/gateway.py` 和 `mcp_servers/pv_ops.py`。写服务只有 `create_work_order` 一个工具，不能通过 MCP 注册草案、批准动作或更改审批记录；只模拟本地人工核查工单，不连接设备控制或外部工单系统。读工具网关与模型规划清单不接入该写工具。
- 默认不加载写服务。CLI 须显式 `--enable-ops`，可用 `--ops-db` 选择台账，否则使用忽略的 `data/ops/work_orders.sqlite3`。单独设置 `--ops-db` 会拒绝启动；程序入口传入 `ops_store` 时必须同时提供可信的人工审批处理器，默认没有自动批准回调。
- 图增加 `propose_action → approval[interrupt] → act`，未启用、无故障、证据不足时直接报告。即使诊断选择结构模式，工单草案也必须额外通过领域必要证据检查。草案包含请求、具体报告、全部引用证据及观测/知识/提示词/运维策略指纹；动作 ID 绑定 run_id 和草案摘要。审批响应须带匹配的动作 ID、摘要、严格布尔决定与审批人，字符串 true/false、数字、空审批人、其他动作的决定均无效。
- CLI 展示具体草案并等待手工输入 `approve`；其他输入或 EOF 都按未批准处理。`Command(resume=...)` 的决定记录到 SQLite；拒绝决定也保存，既有决定不能静默替换。程序调用者的审批处理器和本地 OS 用户属于可信控制面，本轮没有实现远程身份认证、审批 HTTP API 或 Gradio 审批按钮，不能把任意模型输出当成人工身份。
- `act` 在请求写服务前再次核对当前报告、证据和策略是否仍与获批快照一致；暂停期间修改报告会阻断写入。服务端只从台账读取已注册的具体内容，检查审批 ID、决定及内容摘要，不接收调用者替换设备或报告。SQLite 外键、事务及动作唯一约束保证同一动作只产生一张工单；重复调用返回原收据和创建时间。审批记录、草案及工单可在重新打开数据库后读取。
- `RunResult` 增加动作草案、审批记录、动作结果及 `ops_policy_sha256`；运维指纹覆盖图、状态、CLI、台账、写网关和服务源码，并纳入草案摘要。写调用轨迹记录动作/审批 ID；错误只保留类型。诊断结论与动作结果分开：拒绝工单不抹去已有诊断，RPC 失败也不等于服务端肯定未落库，同一动作可幂等核对或重试。
- 新功能暴露了 v1.1 评分器的工具分类限制：它只认识原先九项只读工具，会把知识检索和全部成功工单都计成未审批写入。因此本次做**工程版升级至 v1.2**，增加两个知识只读工具，并检查成功工单轨迹、草案、审批、收据与运行上下文的完整绑定；仅有同名工具、批准标志或不匹配记录不能免责。故障检测、根因/设备匹配、时间 IoU 与数值引用规则不变，没有因样本成绩而放宽阈值；旧版产物保留。该检查验证可信执行记录的一致性，不独立认证审批人身份，实际执行授权由台账实施；仍不允许自进化修改评分器。
- 全套 **180 个测试通过**，其中新增 **18 个**覆盖默认关闭、真实中断/恢复、拒绝/错误决定、领域不足、报告篡改、跨动作/跨 run 审批、数据库重开、12 次并发重试、提交后响应丢失重试、真实 stdio 服务及 CLI EOF 拒绝；评分测试另核对知识工具分类、已批准操作及损坏审批链的区别。全部批准样例都属于临时目录中的受控测试，不表示真实电站工单已经获批，也没有触发生产操作或收费调用。
- Ruff、本轮 10 个 Python 文件的格式检查、`uv lock --check` 和差异空白检查通过，未新增依赖。v1.2 评分器 SHA-256 为 `03acd7aafa7b0d0eb83ea4120a3fd887b38a7ef5bc8298903c99edb740fc0670`；当前工程规则固定在此版本，后续自进化不得修改。
- 真实 M1 默认只读领域模式在 `data/eval/m1_scorer_v1_2_regression/` 回归，根因/定位仍 10/10、时间窗 9/10、每场景 8 次工具调用、写入数为零；与 v1.1 的诊断指标保持一致。该批只完成 SQLite 工单台账，尚未实现进程重启后的图恢复；后续 PostgresSaver 验证见下节。

本地交互入口（会等待人工确认；不自动批准）：

```bash
.venv/bin/python -m pvdiag.agent 2107 \
  --observations data/scenarios/S2107-F3-0001/timeseries.parquet \
  --metadata data/raw/2107/metadata/2107_system_metadata.json \
  --devices inv03 inv04 inv05 --start 2019-01-08 --end 2019-01-15 \
  --verification domain --enable-ops \
  --output data/runs/interactive_inspection.json
```

当前这条命令默认使用 B0，不触发外部模型；本轮未代替用户在真实场景批准工单。后续继续审批/API 身份边界、Ragas、正常基准和完整消融，项目仍在开发。

### PostgresSaver、跨进程恢复与持久化模型预算

- 新增 `agent/persistence.py`、`runtime.py`、`sessions.py`，接入真实 `AsyncPostgresSaver`。原单次诊断与持久化入口共享运行时，默认仍不连接 PostgreSQL。CLI 提供显式 `init`、`start`、`resume`、`show`；连接仅从环境变量或忽略的 `.env` 中的 `PVDIAG_POSTGRES_DSN` 读取，可用 `--dsn-env` 指定另一个字段，连接值不进入状态或输出。新入口不自动创建数据库或初始化表，必须先明确执行 `init`，且拒绝 public/系统 schema。
- 本机安装 PostgreSQL **17.11**，实际使用独立临时集群测试，未启动登录自启服务。Homebrew 安装时同时更新了 OpenSSL、readline 等依赖。可选 Python 依赖在 `requirements/persistence.in`、`persistence-macos-arm64.txt` 锁定：langgraph-checkpoint-postgres **3.2.0**、langgraph-checkpoint **4.3.0**、psycopg / binary **3.3.6**，与现有 LangGraph **1.2.17** 和 langchain-core **1.7.0** 兼容；原有未跟踪工程配置未改动，Linux 部署尚未实测。
- 每个持久化 run 固定请求与执行契约，包括观测/知识指纹、运维台账位置摘要、诊断/核验/运维/工具源码和提示词指纹、配置模型/接口摘要、预算/模式、Python 及关键依赖版本。恢复时不重新解释请求，也不允许更换数据或模式后继续旧 run；不匹配则在推进图和发出模型请求前拒绝。模型密钥不参与保存，允许通过安全渠道轮换凭据。该契约锁定客户端配置，仍不能把提供商的模型别名当作已固定的后端版本。
- 每次执行用 PostgreSQL 会话 advisory lock 独占相应 run，进程退出时由数据库释放；`show` 在未完成节点上探测实际活动锁，区分仍运行与可恢复的中断，不只依据保存的状态字符串。已经完成的 `resume` 直接返回原结果；重复恢复不会重新规划、请求模型或创建第二张工单。
- 保存与恢复使用显式类型白名单的 JsonPlusSerializer，关闭 pickle fallback；真实测试验证请求、结构化报告、审批中断和收据的序列化。持久化执行使用同步检查点写入。诊断总耗时改用可跨进程的墙钟起止时间，包含等待审批的时间；工具调用耗时仍使用进程内计时。
- 模型预算使用与检查点独立的 PostgreSQL 账本：**请求前**在事务内占用调用名额，完成后再写实际已知 token 与接口返回模型名。进程在请求途中退出时保留该名额，用量标为未知；恢复不会释放不确定名额或重开六次预算。累计调用/用量在新 reasoner 和最终报告中重新加载，规划、诊断、自评仍共享上限；每次输出最多 4096 token、SDK 自动重试仍为零。数据库不可用时不绕过预算继续调用模型。
- 在真实 PostgreSQL 上通过两个独立 Python 进程验证“暂停待审批 → 进程结束 → 恢复 → 本地测试工单创建”，再次恢复和 `show` 返回原结果；验证了不同上下文拒绝、并发控制器互斥，以及进程 `os._exit` 后的锁释放/预算保留。另用离线 SDK 替身跑满六次后恢复审批，最终报告仍保留六次及已知用量，没有新增调用。所有批准均属于临时受控数据，不表示真实电站工单已获用户批准；没有真实收费 LLM 请求。
- 全套 **189 个测试通过**，包含新增两个无数据库单元检查和七个真实 PostgreSQL 集成检查；无专用测试连接时，七个集成检查会明确跳过，不使用应用连接代替。真实 M1 十场景逐个保存、重开运行时并恢复，结果逐个一致，产物在 `data/eval/m1_postgres_regression/`，根因/定位仍 10/10、时间 IoU 达标 9/10、每场景 8 次工具调用。评分器仍为 v1.2，SHA-256 保持 `03acd7aafa7b0d0eb83ea4120a3fd887b38a7ef5bc8298903c99edb740fc0670`。

持久化入口示例（先安全配置专用开发库连接，`init` 会在所选 schema 中创建检查点和账本表）：

```bash
uv pip install --python .venv/bin/python -r requirements/persistence-macos-arm64.txt
.venv/bin/python -m pvdiag.agent.sessions init --schema pvdiag
.venv/bin/python -m pvdiag.agent.sessions start dev-inspection 2107 \
  --observations data/scenarios/S2107-F3-0001/timeseries.parquet \
  --metadata data/raw/2107/metadata/2107_system_metadata.json \
  --devices inv03 inv04 inv05 --start 2019-01-08 --end 2019-01-15 \
  --verification domain --enable-ops --output data/runs/pending.json
.venv/bin/python -m pvdiag.agent.sessions show dev-inspection
.venv/bin/python -m pvdiag.agent.sessions resume dev-inspection \
  --observations data/scenarios/S2107-F3-0001/timeseries.parquet \
  --metadata data/raw/2107/metadata/2107_system_metadata.json \
  --verification domain --enable-ops --interactive \
  --output data/runs/resumed.json
```

`start` 默认保存待审批动作后返回，`resume --interactive` 才在终端展示具体内容并等待手工决定；`show` 不需要模型凭据或观测路径。恢复须保留对应 SQLite 工单台账和原始输入快照，PostgresSaver 不备份这些外部文件。本轮验证后停止本任务创建的临时数据库；远程审批/API 身份认证、Ragas、长期记忆、正式正常/留出基准及完整消融仍未完成，目标保持推进中。


### FastAPI、SSE 与身份绑定审批（2026-10-26）

- 新增 `src/pvdiag/api/` 和 `config/api.example.json`，通过服务端 profile 固定观测快照、设备白名单、知识库、模型模式和可选 SQLite 工单台账。HTTP 客户端只提交 `profile_id` 与 `DiagRequest`，不能指定路径、模型、收费开关或审批人。示例默认 B0、领域核验、BM25，`ops_db: null` 关闭工单。LLM profile 还须服务启动时显式 `--allow-model-calls`，每 run 最多六次调用；**启用服务不自动获得新的收费授权**。
- 本地开发身份来自 `PVDIAG_API_TOKENS`，只在 `Authorization: Bearer ...` 接收原 token，服务端保存 SHA-256 摘要并做常量时间比较；无默认凭据。每个身份配置 `subject`、`roles`（`diagnose` / `read` / `approve`）和 `profiles`。普通身份仅访问自己在授权 profile 下的 run，审批身份可读取其授权 profile 下的运行；越权 run 返回 404。审批人从认证身份注入，body 中额外的 `reviewer` 会被拒绝。这是开发用身份机制，尚无 OAuth/SSO、TLS 或公开部署。
- 新增两张 PostgreSQL API 表，分别保存请求归属/幂等绑定和单调编号的 SSE 事件。`POST /diagnose` 必须提供 8–128 位字母、数字、下划线或连字符组成的 `Idempotency-Key`；同身份、同 key、同请求返回同 run，不重新诊断；同 key 不同请求返回 409。运行在独立后台任务中，客户端断线不会取消诊断。默认最多两个活动任务，容量满时新请求返回 429，且不注册第二个 run；这是单进程容量限制，不是全局分布式任务队列。PostgreSQL 活动锁仍约束同 run 的执行。
- 运行时通过真实 LangGraph `astream(stream_mode="updates", durability="sync")` 发送 `node` 进度，仅含节点名、轮数、工具调用数和临时报告状态；不转发原始节点输出。`accepted`、`node`、`awaiting_approval`、`completed`、`error` 持久化，可在重开服务后重放。事件游标格式为 `<run_id>:<sequence>`，支持 `Last-Event-ID`；错误 run 或超前游标返回 400，已读到最后终止事件返回 204。错误响应不回显输入、数据库连接或提供方异常；CLI 关闭 access log，避免误把 URL 中的敏感信息写入日志。
- `POST /runs/{id}/approve` 要求 `approve` 角色，并精确匹配展示草案的 `action_id` 与 `payload_sha256`，`approved` 必须是 JSON 布尔值。可信决定**先写入 SQLite，再调度图恢复**；worker 未启动或服务重开后，可从原决定恢复，并保留原审批人。相同决定重试不创建第二张工单，不同决定返回 409。拒绝路径保存决定且创建零工单。应用关闭会取消本服务管理的任务，检查点、预算与已保存决定可恢复；不会自动恢复所有历史中断 run。

**启动与配置**：可选依赖由 `requirements/api.in` 和 macOS arm64 / Python 3.11 锁文件 `requirements/api-macos-arm64.txt` 管理，FastAPI 0.162.0、Uvicorn 0.54.0、Starlette 1.10.0；Linux 未实测。先在环境或忽略的本机 `.env` 配置 `PVDIAG_POSTGRES_DSN` 和 `PVDIAG_API_TOKENS`。身份字段使用如下 JSON 结构，其中摘要由本机生成的高熵 token 计算，不能把占位符直接作为可用配置；原 token 仅留在受信客户端：

```json
[
  {
    "subject": "operator",
    "token_sha256": "<原 token 的 64 位小写 SHA-256>",
    "roles": ["diagnose", "read"],
    "profiles": ["dev-2107"]
  }
]
```

数据库及账号由操作者准备，表初始化是显式操作；`serve` 仅检查既有表，不自动迁移。使用项目虚拟环境：

```bash
.venv/bin/python -m pvdiag.api init --schema pvdiag
.venv/bin/python -m pvdiag.api serve --schema pvdiag --profiles config/api.example.json
```

默认监听 `127.0.0.1:8000`。授权客户端先调用 `GET /profiles` 获取可见窗口/设备，再以 `Authorization`、`Idempotency-Key` 请求头向 `POST /diagnose` 提交：

```json
{
  "profile_id": "dev-2107",
  "request": {
    "system_id": 2107,
    "device_ids": ["inv03", "inv04", "inv05"],
    "start": "2019-01-08T00:00:00Z",
    "end": "2019-01-15T00:00:00Z"
  }
}
```

响应头 `X-Run-ID` 给出运行 ID，`text/event-stream` 中 `id` 可用于重连。`GET /runs/{id}` 返回 JSON；`GET /runs/{id}/events` 重放事件。`POST /runs/{id}/resume` 仅诊断拥有者可用：执行恢复时返回 SSE，已完成或等待尚未决定的审批时返回 JSON；客户端应检查 `Content-Type`。审批先读取具体草案，再向 `POST /runs/{id}/approve` 提交 `approved`、`action_id`、`payload_sha256` 和可选 `reason`；执行恢复时返回 SSE，完成后相同决定重试返回原 JSON 结果。审批需要服务端显式配置工单台账和审批身份，示例不默认启用。

**本轮验证**：全套 **197 个测试通过**，含八个 API 检查（其中六个使用真实独立 PostgreSQL）和既有七个 PostgreSQL 集成检查。覆盖身份/归属/角色、路径及审批人注入拒绝、幂等冲突、容量限制、游标边界、错误脱敏、审批先落库及重启恢复；真实 Uvicorn/TCP HTTP 测试使用等待屏障确认断线后 worker 仍活动，释放后可重放至待审批。受控审批测试仅操作临时测试台账，没有替真实场景批准工单。评分器保持 v1.2，未修改评分规则。

真实 M1 `S2107-F3-0001` 使用示例 profile 完成 ASGI HTTP 冒烟：返回 16 个节点事件和最终报告，八次只读工具调用；根因、设备和时间 IoU 检查通过，同 key 重试保持同 run，服务重开后的结果与事件重放一致。收据在 `data/runs/m1_api_smoke.json`。该运行是 B0，零模型调用、工单关闭；真实 TCP 断线行为由上述独立受控测试验证，不把两者混同为真实 LLM 或公开服务成绩。

Gradio 后续实现见下节；Ragas、Langfuse、长期记忆、自进化及正式正常/留出基准和完整 LLM 消融仍待推进。2026-10-26 复查本机仍无 `.env` 或 `DEEPSEEK_API_KEY`，未发起真实模型请求，单场景授权未消耗。


### Gradio 单页与 API 客户端（2026-10-26）

- 新增 `src/pvdiag/ui/`：浏览器分别选择可见数据配置、设备、带时区的诊断窗口和问题，默认值来自认证后的 `GET /profiles`。支持开始/重试、读取运行、重连持久事件及恢复中断运行；刷新页面后可重新填入令牌与运行 ID 读取结果。SSE 节点进度显示在“流式过程”页签，界面最多保留最近 200 行，完整持久事件仍在 API 数据库中。清空页面仅准备新请求，不停止后台任务。
- 客户端为同一浏览器会话中的同一组输入保留幂等键，**在 HTTP 请求发出前**保存到会话状态，网络失败后的重试仍复用该键。变更输入或显式清空后再开始才创建新请求。客户端同时处理 SSE、恢复/重复审批返回的 JSON，以及已读到最后事件的 204；重连携带相应游标，拒绝错 run 的事件，不自行重新开始既有诊断。SSE 帧大小上限 4 MiB，拒绝无效/截断帧；连接和错误响应只显示固定的操作提示。
- “诊断结果”展示摘要、根因/设备/时间窗及完整核验结果；“证据图表”按具体证据 ID 展示 `get_timeseries` 采样点与 `peer_compare` 日间低值占比，并可核对原始证据、查询窗口、采样/截断标记。时序图使用 Plotly 日期轴，数据和 tooltip 均显式使用 UTC 字符串，保留证据点的 UTC 时间，避免浏览器本地时区导致故障时间偏移；图表标注实际指标单位，不补造缺失采样点。界面为窄屏顺序布局，表格可独立滚动。
- 审批区展示完整的 `PendingAction`（草案内容、证据、run ID、动作 ID 和内容摘要），确认框默认未勾选；批准和拒绝均须人工勾选确认。提交绑定**已展示的草案快照**，不使用可编辑的运行 ID 文本框替换审批目标，也不允许填写审批人。审批身份通过当前 API 令牌确定，后端仍独立核验角色、范围、内容摘要和既有决定。被拒绝或失败的请求可重新读取运行后处理，UI 不绕过 API 直接操作工单台账。
- API 地址仅由启动 CLI 指定，不提供浏览器 URL 输入；只接受无嵌入凭据/查询串的本机 HTTP origin 或 HTTPS origin，禁用重定向和环境代理。令牌由每个浏览器会话的密码框传入，不写入 Gradio 状态、诊断请求、收据或服务端共享身份；关闭 Gradio analytics、run history、公开分享及公共回调 API 展示。UI 默认固定监听 `127.0.0.1`，这是开发界面，尚非公开部署或企业认证系统。
- 页面实测发现并修复 API 的窗口边界：观测可能为五分钟采样，原先 `max(ts)+15min` 会返回 `00:10` 等无效默认结束时间。现在开始时间向上对齐十五分钟，结束时间使用最后一个十五分钟 bin 的右边界；`Profile.validate` 与公开范围一致，并增加非整刻原始数据的回归检查。

**独立环境启动**：Gradio **6.30.0** 需要的 Hugging Face Hub 版本与现有 Transformers/BGE 固定环境不兼容，因此 UI 锁文件独立于 API/知识依赖，不在 `.venv` 中安装。诊断环境的 Hub **0.36.2** 和 Pillow **13.0.0** 已恢复；两个环境的依赖兼容检查均通过。UI 使用 Gradio 6.30.0、Plotly 6.9.0，macOS arm64 / Python 3.11 完成实测，其他平台未验证。先按上一节启动 API，再在仓库根目录另开终端：

```bash
uv venv .venv-ui --python .venv/bin/python
uv pip install --python .venv-ui/bin/python -r requirements/ui-macos-arm64.txt
PYTHONPATH=src .venv-ui/bin/python -m pvdiag.ui --api-url http://127.0.0.1:8000
```

访问 `http://127.0.0.1:7860`，填入既有 API 身份的原 token，再点击“连接并读取配置”。UI 不需要数据库连接、模型凭据或观测路径；这些仍由 API 管理。诊断拥有者和审批者可分别用自己的令牌读取同一 run。默认示例是 B0、工单关闭，不产生外部模型费用；LLM profile 的启用和收费授权仍遵循既有边界。

**验证结果**：全套 **206 个测试通过**。新增九个检查覆盖 SSE 多行/UTF-8/心跳/截断/超量事件、错误脱敏与拒绝重定向、失败后的同键重试、会话隔离、游标和 JSON/204 恢复、已展示草案绑定与显式确认、证据点/缺失值/UTC 保留、输入及范围对齐，以及真实 PostgreSQL/FastAPI 下的请求归属、审批身份和幂等恢复；后续图表时区修改后，相关 UI/API **17 项检查再次通过**。

真实浏览器还完成了 M1 `S2107-F3-0001` 的 B0 页面诊断、刷新后读取和证据图表检查：八次只读工具调用，根因、设备与时间 IoU 检查通过，工单关闭，零模型调用。另在独立的受控合成 F3 profile 中核对待审草案，验证未确认不提交、使用受控审批身份确认后仅生成一张本地测试工单；没有替真实 M1 场景批准工单。原始收据为 `data/runs/gradio_browser_smoke.json`，实际页面截图为 `data/runs/gradio_evidence.png` 与 `data/runs/gradio_peer_evidence.png`。两类图表均在浏览器中核对，时序图使用 UTC 日期轴，同站对比使用连续的百分比数值轴；临时服务和数据库在验证后停止。

下一步仍为正式正常/留出基准、Ragas/Langfuse、真实 LLM 单例、长期记忆及自进化等；没有因此取得完整 A3 或正式泛化成绩。DeepSeek 本机凭据仍未配置，既有单场景授权未消耗。


### 原始基线人工审核与六类 B0 开发评测（2026-10-26）

- 新增 `eval/baseline_review.py`，提供 `prepare / render / review / release / status`。`prepare` 从清单绑定的原始源 Parquet 重建每个候选，与已落盘的观测逐值、逐字段精确比较，不放宽浮点容差；任一候选不一致时不创建审核包。80 个真实 M2 候选已全部通过，按原始查询窗口、设备队列和 POA 去重为 **20 份原始基线包**。
- 审核包仅含**注入前**的完整范围、十五分钟原始功率/辐照/经验同站比值、固定筛查结果和原始基线 Parquet，不含注入故障标签、诊断结论或场景 ID。包 ID 是内容 SHA-256，源 Parquet、元数据和基线快照均绑定指纹；各场景观测另行记录独立指纹。审核目录与原候选清单在本地 SQLite 中绑定，修改目录、审核数据或基线原文都会使后续校验失败。
- `render` 复用现有 UI 虚拟环境中的 Plotly，生成离线 HTML 与本地 `plotly.min.js`，不调用模型或外部报告服务。展示 UTC 功率、辐照和经验 P95 归一化同站比值，可缩放并查点；缺测保留空缺，不插值。报告与目录已在真实浏览器核对，预览在 `data/reviews/m2_dev_v1/preview.png`。图上的同站比值不等于物理性能比，人工接受只表示该范围内未观察到故障，不能作为生产健康认证。
- `review` 需显式 `--interactive`，展示审核范围与报告入口后，由本地操作者输入接受/拒绝及依据，身份由本机 OS 用户注入。接受时绑定完整核对清单；同一内容决定幂等，已有决定不能静默替换。只有全部原始基线都获接受，`release` 才派生可评测开发清单；原清单、正常候选的 `evaluation_eligible=false` 和原观测文件不改写。拒绝或未审核的范围会阻止发布，不自动移除失败候选凑数。上述接受路径仅在受控合成测试中验证，**本次没有代替用户接受任何真实基线**。
- `eval/run_bench.py` 现在校验审核清单与原目录/本地决定的一致性，并在第一次诊断前加载、检查并冻结全部观测与元数据快照。后续候选文件损坏或元数据被替换会在首次 reasoner 调用前失败；正常标签即使被手动改为 eligible，没有对应审核包仍拒绝评分。诊断图仍只接收查询范围与观测，审核身份、故障真值和注入参数不进入模型状态。
- 新增显式 `--faults-only`，可从原 80 个开发候选中排除 20 个未审核的未注入候选，运行 60 个故障场景；摘要记录完整输入清单指纹、每个观测快照、被排除的 ID 和逐类指标。逐类/整体均复用冻结评分器 v1.2，SHA-256 保持 `03acd7aafa7b0d0eb83ea4120a3fd887b38a7ef5bc8298903c99edb740fc0670`，没有调整规则、阈值或评分标准。没有正常样本时，正常准确率和误报率保持 null；全为正例的 precision 不用于推断正常识别能力。
- `validate_split_isolation` 补齐 F5 目标检查：即使逆变器和查询时间完全不交叉，共享被测 POA 的 F5 开发/留出组合仍拒绝。遵循用户确认，当前只有一个 POA 的 2107 不生成 F5 跨设备留出集。本轮仍只运行开发集，没有查看或评测正式留出集。

**六类故障开发子集 · B0 + domain 核验实跑**：使用原 `bench/m2_dev_candidates.json` 和 `data/m2_candidates/`，输出在 `data/eval/m2_b0_fault_subset_v1/`。每类十个、总计六十个，规则版本仍为 `rule-v1`：

| 故障类型 | 根因与设备联合命中 | 时间 IoU 达标 |
|---|---:|---:|
| F1 积灰 | 0/10 | 0/10 |
| F2 遮挡 | 0/10 | 0/10 |
| F3 停机 | 9/10 | 9/10 |
| F4 组串损失 | 8/10 | 7/10 |
| F5 辐照漂移 | 0/10 | 0/10 |
| F6 通讯丢失 | 0/10 | 0/10 |
| 合计 | **17/60（28.33%）** | **16/60（26.67%）** |

共 414 次只读工具调用，均成功；17 个 fault、39 个 inconclusive、4 个 needs_review，弃答率 71.67%，数字引用检查通过，模型调用与未经审批写入均为零。当前规则仅实现 F3/F4 判别，且在更广的开发样本上也未全检出；这次结果明确暴露四类未覆盖与短窗局限，不是完整六类规则或 LLM/A3 成绩。该集合仍是严格筛查、缺少真实人工接受的七天候选，慢变效应较弱，没有正式正常误报率、留出泛化或生产准确率结论。

**复现与人工审核入口**（在仓库根目录）：

```bash
# 准备必须使用新的目录；已存在的审核资料不会被覆盖。
.venv/bin/python -m pvdiag.eval.baseline_review prepare --bundle data/reviews/m2_dev_v1
PYTHONPATH=src .venv-ui/bin/python -m pvdiag.eval.baseline_review render --bundle data/reviews/m2_dev_v1
.venv/bin/python -m pvdiag.eval.baseline_review status --bundle data/reviews/m2_dev_v1

# 打开 data/reviews/m2_dev_v1/index.html，人工逐份核对后再运行：
.venv/bin/python -m pvdiag.eval.baseline_review review --bundle data/reviews/m2_dev_v1 --review-id "<页面上的64位审核ID>" --interactive
# 只有全部接受后才能派生开发清单；该命令不会自动接受基线：
.venv/bin/python -m pvdiag.eval.baseline_review release --bundle data/reviews/m2_dev_v1

# 无需模型、无需接受正常候选的故障子集评测：
.venv/bin/python -m pvdiag.eval.run_bench --manifest bench/m2_dev_candidates.json --scenario-root data/m2_candidates --faults-only --verification domain --output data/eval/m2_b0_fault_subset_v1
# 若使用人工审核后派生的清单，必须同时指定 --review-bundle 原审核目录。
```

实际审核目录已生成，状态为 **20 pending、0 accepted、0 rejected**，正常标签发布仍被阻止。M1 十场景也使用新预检流程重跑，根因/定位仍 10/10、时间窗 9/10，结果在 `data/eval/m1_benchmark_preflight_regression/`。

本轮全套 **218 个测试通过**，包含新增十二个审核/划分检查，以及原有 PostgreSQL、API 与 UI 检查；覆盖精确重建、盲化去重、源/包/目录/观测/元数据篡改拒绝、决定幂等与不可替换、未审核正常标签拒绝、审核信息不进入诊断状态、后续文件异常在首次调用前拦截、显式故障子集和 F5 传感器目标隔离。Ruff 与格式检查通过，没有新增依赖。独立临时 PostgreSQL 和报告浏览服务在验证后停止。

后续需要真实人工基线审核、可用长窗口慢变数据、六类规则覆盖与真实 LLM 对比；Ragas/Langfuse、正式正常/留出基准、长期记忆和自进化仍未完成。DeepSeek 单场景授权未扩大，本机尚缺凭据，本轮没有真实模型请求。


### B0 rule-v2：积灰恢复、遮挡与采集缺失（2026-10-26）

- `agent/rules.py` 从 `rule-v1` 升级为 **`rule-v2`**，继续只消费真实 MCP 观测摘要，不访问真值、基线审核结论或注入参数。保留三轮、最多三条候选和既有工具调用预算；版本随运行元数据保存，旧版产物不覆盖。
- **F1 积灰恢复候选**：检查已有形态证据中的恢复阶跃（至少 0.02）及恢复前多日相对斜率（不高于 −0.001/天）。取证先观察参考期之后的完整范围，再对可见低值包络重新画像；未触及通用低值阈值、但已有多日缓降和恢复证据时，也可提名该评估范围，避免只截取恢复前低值段而丢掉恢复证据。报告置信度为 low，明确时间是观测包络/评估范围，不是精确的真实积灰起止或已经确认的清洗记录。无恢复的持续缓降不直接归为积灰，也不因低 IQR 就归为组串阶跃。
- **F2 遮挡候选**：要求至少三天重复损失、当地小时集中在至多三个小时、零值占比不超过 0.05，且其他时段相对性能恢复到 `[0.9, 1.15]`。直接使用工具按站点时区生成的统计，不把固定 UTC 小时当成当地遮挡时段；与非零、覆盖更多小时的稳定组串损失区分。数值通过 JSON Pointer 引用原始工具字段。
- **F6 采集/通讯候选**：根据 `list_devices` 查询目标设备实际可见的全部测点质量，求各测点缺测段交集，当前 B0 最短候选为一小时；再用该交集的 `get_timeseries` 完整覆盖率/观测 bin 计数复核，并确认至少两台同期逆变器仍有记录。只缺交流功率而其他测点在线、全站同时缺失、缺测列表不完整时均不能据此认定整设备通讯中断。仅零值/跳变列表截断而缺测列表完整时不误丢证据。夜间和没有可用 POA 参考时也可检查采集缺失，返回 low 置信度，不进一步区分网络、采集器或设备供电原因。
- B0 在默认 structural 模式下也自行要求候选前参考期、足够高辐照有效观测、完整候选段功率质量与两台同期非零对照；不把可选 domain 核验当作分类器。F3/F4 继续使用原有可见段规则，F4 增加对长窗明显渐变的排除。规则与领域必要条件存在重叠，domain 仍不是独立的语义忠实度评测。
- **F5 和 normal 仍保守弃答**：稳定的同站相对关系、甚至有设备参数，也不足以把全站共同积灰/性能偏差归因为辐照传感器漂移。当前没有独立辐照参考/校准证据，不添加 F5 确诊规则，也不凭稳定同站比值给正常结论。保持用户确认的 F5 严格目标设备隔离，继续等待独立传感器数据。
- `mcp_servers/analytics_tools.py` 仅对至少两个有效同站对照的行计算中位数，其他行保留 NaN，修正全站缺测时的空集合统计警告；不插补缺测，不改变有效行的计算结果。

**同输入开发对比**：复用 `bench/m2_dev_candidates.json` 的六十个故障候选与原 metadata，两个版本均为 B0 + domain 核验，均显式排除二十个未审核的未注入候选。已核对清单指纹、六十份观测快照、排除列表及评分器 SHA-256 完全一致。412 份相同工具/相同请求的证据响应逐项一致；另外两条旧聚焦请求在新取证策略中改变了范围，已在比较收据中列明。评分器仍为冻结 v1.2，未修改真值、评分阈值或原故障参数。

| 故障类型（各十个） | rule-v1 联合命中 | rule-v2 联合命中 | rule-v2 时间 IoU 达标 |
|---|---:|---:|---:|
| F1 积灰 | 0/10 | 0/10 | 0/10 |
| F2 遮挡 | 0/10 | 6/10 | 6/10 |
| F3 停机 | 9/10 | 9/10 | 9/10 |
| F4 组串损失 | 8/10 | 8/10 | 7/10 |
| F5 辐照漂移 | 0/10 | 0/10 | 0/10 |
| F6 采集/通讯缺失 | 0/10 | 10/10 | 10/10 |
| 合计 | **17/60（28.33%）** | **33/60（55.00%）** | **32/60（53.33%）** |

rule-v2 共 484 次只读工具调用（平均 8.07 次/场景，rule-v1 为 6.90），均成功；33 个 fault、27 个 inconclusive，弃答率 45%，数字引用检查通过，真实模型调用及未经审批写入均为零。七天积灰候选仍因信号弱而未命中；F1 仅在更长、明确渐变及恢复的受控数据上验证了规则分支。新增收益来自遮挡与采集缺失，不能据此认定完整六类规则、正式正常误报率或留出泛化达标。

结果保存在 `data/eval/m2_b0_fault_subset_v2/`，同输入比较收据在 `data/eval/m2_b0_rule_v1_v2_comparison.json`，收录规则/工具源码指纹、逐类结果、共同证据核对及改变的聚焦请求。旧 `data/eval/m2_b0_fault_subset_v1/` 保留。M1 十个真实场景也已重跑，根因/定位仍 10/10、时间 IoU 达标 9/10、平均每场景 9.1 次只读工具调用，结果在 `data/eval/m1_rule_v2_regression/`。未为改善数字修改清单或独立评分器。

```bash
.venv/bin/python -m pvdiag.eval.run_bench --manifest bench/m2_dev_candidates.json --scenario-root data/m2_candidates --faults-only --verification domain --output data/eval/m2_b0_fault_subset_v2
.venv/bin/python -m pvdiag.eval.run_bench --verification domain --output data/eval/m1_rule_v2_regression
```

本轮全套 **241 个测试通过**，新增二十三项规则检查：F1/F2/F3/F4/F6 在 structural 与 domain 两种模式下的真实 MCP 分类与引用；仅功率缺测、全站缺测、无 POA 的夜间缺测、跨测点缺测交集、缺测段截断、无恢复缓降、未触及深度低值但有恢复的较弱积灰、无前序缓降的维修阶跃、正常/漂移证据不足、全站共同积灰、弱日间停机及不同设备双故障。已有 PostgreSQL、审批、API、UI 和评测检查同时通过，Ruff/格式检查通过，没有新增依赖。新建的独立临时 PostgreSQL 在验证后停止；原临时目录已不存在，没有重启或停止其他数据库。

历史报告仍可核对；持久化检查点继续按既有源码/依赖指纹绑定版本，旧未完成运行不能静默使用新规则续跑。实际基线审核仍是二十份待人工核对，真实 LLM、正常/留出评测、独立辐照与长窗数据、Ragas/Langfuse、长期记忆和自进化仍待推进。DeepSeek 单场景授权未消耗。

### A0–A3 消融执行边界（2026-10-26）

本批将 §9.3 的档位落实为可执行策略。单次诊断、开发集评测及 PostgreSQL start/resume CLI 均可用 `--ablation A0|A1|A2|A3` 选择；未指定档位时沿用原有 B0/LLM 及可选核验行为。A0–A3 要求显式 LLM 模式，评测过程不允许挂载工单。A4/A5 尚未实现，不能通过改标签冒充。

| 档位 | 模型可主动调用的工具 | 核验和循环 |
|---|---|---|
| A0 | 无；不调用规划模型 | 固定摘要后仅调用一次假设模型；最终结构/引用守卫保留，失败则降级，不发起修复循环 |
| A1 | `data_quality`、`get_timeseries` | 可按模型自己的计划继续取数；不向模型回传结构核验问题，也不因核验失败触发修复 |
| A2 | A1 + 全部五个分析工具 | 同 A1；禁止知识工具及领域核验反馈 |
| A3 | A2 + `search_fault_kb`、`get_playbook` | 固定为 `domain_llm`：领域规则 + 模型自评，问题可反馈至后续规划；必须显式挂载知识快照 |

- 新增 `agent/ablation.py`，策略版本为 `ablation-v1`。固定摘要协议为 `fixed-data-v1`：所有 LLM 档位都先取得同一观测快照的设备目录、站点信息、请求逆变器的交流功率，以及可用 POA/环境温度测点的质量和小时均值摘要。每组时序响应最多 96 个均匀采样点，质量工具保留原有有界缺测/零值/跳变摘要；完整覆盖率统计不以采样点计数替代。固定摘要只依赖公开请求范围，不使用故障真值、候选 ID、注入参数或故障条件化时间窗；不预计算分析画像，也不预载知识。
- 有 POA 和环境温度时共八次确定性取数（目录/站点两次 + 三组质量/时序共六次）；缺少传感器时省略对应组。**A0 的“不给工具”指不给模型主动调用能力，固定摘要仍需真实取数。**全部预取调用均以 `guard_in` / `fixed_summary` 节点如实进入 trace，并计入总工具预算；预算不足或摘要取数失败会在模型调用前终止，不能静默截掉一部分摘要。A0 的固定摘要不包含直流/组件温度等全部可见测点，不宜由此单独认定整设备通讯故障。
- A0/A1 不创建分析服务；A0–A2 不接受知识快照。网关同时过滤模型工具清单并在执行前按节点和档位检查白名单，即使误挂载了额外服务、模型仍输出禁用工具，也不会发送到 MCP。失败尝试如实计入 trace。两项目录工具仅供固定引导阶段使用，模型在调查阶段无需重复调用。
- 每档保留共同的输出 schema、设备/时间范围、数字及时间引用检查。A0–A2 的模型输入清除核验记录、critique 与前一轮报告，结构失败只用于本轮输出守卫/审计，不能把核验修复能力提前混入消融。A3 的领域规则和模型自评可以反馈；规划、假设与自评共用已有模型账本，默认最多六次调用，每次最多 4096 输出 token，不为自评另开预算。预算耗尽不视为核验通过。
- 单场景结果新增 `ablation`，包含档位、工具权限、核验反馈开关、摘要协议、工具/循环预算、固定证据 ID 与内容 SHA-256。开发集摘要新增档位策略与逐场景 `summary_snapshots`；同输入四档的固定摘要指纹应相同。知识指纹仍包含检索后端与模型配置，BM25 和 BGE 后端不能在比较中静默互换。正式主表仍需固定模型、检索后端与预算，再获得相应批量调用授权。
- 持久运行 contract 绑定档位策略及既有源码、模型、数据、知识和预算。真实 PostgreSQL 检查已验证：关闭连接后重建 A0 运行时可返回原始结果、摘要指纹和一次模型用量；尝试改成 A1 时，在执行图/工具/模型之前被拒绝。

**验证范围**：全套 **271 个测试通过**，包含新增 25 项消融边界检查、四档开发集入口检查及一项真实 PostgreSQL 恢复检查。模型传输用明确标记 `offline-ablation-fixture` 的离线 SDK，观测和知识调用走真实 FastMCP；覆盖固定证据一致、A0 不规划、允许工具实际执行、禁用工具执行前拒绝、结构问题不回流、A3 知识与领域问题回流、自评共享预算、配置冲突/不完整摘要在请求前拒绝、真值隔离与结果落盘。它们验证执行边界，**没有得到真实 LLM 准确率或 A0→A3 收益结论**。已有 API、UI、人工审批和 PostgreSQL 测试同时通过。Ruff、格式、依赖兼容、离线 lock 检查与差异空白检查通过，未增加依赖。

固定评分器 v1.2 未改，SHA-256 仍为 `03acd7aafa7b0d0eb83ea4120a3fd887b38a7ef5bc8298903c99edb740fc0670`。模型实际输入仍受既有 80,000 字符上限约束，复杂窗口可能因上下文预算被拒绝或降级；未提升预算、未隐式重试。A3 挂载知识只说明检索能力可用，是否实际检索、采用什么证据须查看 trace，不能仅凭档位推断知识贡献。

真实 M1 十场景 B0 + domain 回归保存在 `data/eval/m1_ablation_runtime_regression/`，仍为根因/定位 10/10、时间窗 9/10、共 91 次只读工具调用，模型请求和未经审批写入均为零；本批运行时调整没有改变这些既有规则成绩。隔离 PostgreSQL 在验证后停止并核实无服务运行，未操作其他数据库。

本机再次确认没有 `DEEPSEEK_API_KEY`，本批未调用真实模型。已有授权仅覆盖一个开发场景，不扩展为四档矩阵或整套开发集。正常基线仍待真实人工审核；F5 继续严格隔离目标设备，等待独立 POA 数据。Ragas/Langfuse、正式正常/留出评测、长期记忆与自进化仍待推进。
