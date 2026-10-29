# PV Diagnosis Agent

基于 LangGraph + MCP 的光伏电站时序故障诊断 Agent。在公开的 NREL PVDAQ 数据上注入已知故障，构造有标准答案的场景，用固定的纯代码评分器做可复现评测，并让评分器驱动提示词 / Playbook 的自进化。

完整设计、每一批的验证记录和局限见 [docs/DESIGN.md](docs/DESIGN.md)。本 README 只说明怎么跑起来。

## 现状（只写实际跑出来的结果）

| 项目 | 结果 |
|---|---|
| 规则基线 B0（不用大模型） | M1 十个冒烟场景：根因与设备 10/10，时间窗 IoU≥0.5 为 9/10；M2 六类各 10 个故障候选：根因+设备联合命中 33/60 |
| DeepSeek（单场景试跑） | 第三次运行给出了根因、设备、时间窗都正确的结论，但因一条数值引用路径越界被守卫降级为需人工复核；尚无通过守卫的 LLM 诊断，没有 A0–A5 成绩 |
| 双故障（B0） | 开发集 20 个 hard 双故障场景：17 个判出有故障，但两个根因与设备都答对的只有 F4+F6 的 3/5 |
| 正常场景 / 留出集 / 跨站 | 2107 留出候选 70 个、9069 跨站候选 40 个已生成并通过预检；开发集与留出集各 20 份原始基线待人工审核；尚无正式误报率、留出或跨站成绩 |

这些数字来自开发数据，不代表泛化能力。

## 架构

```
数据与仿真 (PVDAQ 2107 · 六类故障注入 · 失败驱动场景生成)
        │
MCP 工具：pv-data · pv-analytics · pv-knowledge · pv-ops(写，须人工审批)
        │  stdio / Streamable HTTP
诊断图 (LangGraph)：范围检查 → 规划 → 并行取证 → 假设 → 核验 →(审批)→ 报告
        │                      └── 证据不足回到规划，最多 3 轮
服务：PostgresSaver 断点续跑 · FastAPI/SSE · Gradio
评测：固定评分器 · A0–A5 消融 · Ragas · Langfuse(本地记录)
进化：长期记忆 (pgvector + BGE) · harness 自进化 (提议/试跑/合并/沉淀)
```

| 目录 | 内容 |
|---|---|
| `src/pvdiag/data` | 下载、清洗、数据卡 |
| `src/pvdiag/sim` | 故障原语、场景、基线筛查、候选与失败驱动生成 |
| `src/pvdiag/mcp_servers` | 四组 MCP 工具 |
| `src/pvdiag/agent` | 诊断图、规则基线、LLM 适配、守卫、核验、持久化 |
| `src/pvdiag/eval` | 评分器、基准运行、基线人工审核、Ragas |
| `src/pvdiag/memory`、`src/pvdiag/evolve` | 长期记忆（A4）、自进化（A5） |
| `src/pvdiag/api`、`src/pvdiag/ui` | FastAPI 服务、Gradio 页面 |
| `harness/`、`playbooks/`、`kb/` | 提示词、故障诊断 Playbook、许可核对过的知识语料 |
| `bench/` | 场景清单（真值）；观测数据在被忽略的 `data/` 下 |

## 快速开始（本地，B0，不调用大模型）

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
```

下载并清洗 2107 的数据（匿名访问 AWS 开放数据桶，这四个文件加元数据约 445 MB，不入库）：

```bash
uv run python -m pvdiag.data.download 2107 --include 2107_electrical_data.csv --include 2107_irradiance_data.csv --include 2107_environment_data.csv --include 2107_meter_15m_data.csv
```

```bash
uv run python -m pvdiag.data.normalize 2107 --start 2019-01-01 --end 2021-01-01
```

```bash
uv run python -m pvdiag.data.datacard 2107
```

生成 M1 十个冒烟场景，并用 B0 跑一遍评测：

```bash
uv run python -m pvdiag.sim.scenarios 2107 --start 2019-01-01 --end 2019-07-01 --devices inv01 inv02 inv03 inv04 inv05 inv06 inv07 inv08 inv09 inv10 inv11 inv12 --poa-device sensor_149574
```

```bash
uv run python -m pvdiag.eval.run_bench --verification domain --output data/eval/m1_b0
```

单个场景诊断：

```bash
uv run python -m pvdiag.agent 2107 --observations data/scenarios/S2107-F3-0001/timeseries.parquet --metadata data/raw/2107/metadata/2107_system_metadata.json --devices inv03 inv04 inv05 --start 2019-01-08 --end 2019-01-15 --verification domain
```

运行测试（需要真实 PostgreSQL 的用例会自动跳过，除非设置 `PVDIAG_TEST_POSTGRES_DSN`）：

```bash
uv run --group dev pytest -q
```

## 可选功能

各功能的依赖单独锁定在 `requirements/` 下（`*.in` 为来源，`*-macos-arm64.txt` 为 macOS arm64 锁文件）。

| 功能 | 依赖 | 入口 |
|---|---|---|
| 大模型诊断（DeepSeek 等 OpenAI 兼容接口） | 核心依赖 | `--mode llm --allow-model-calls`，凭据放在环境变量或被忽略的 `.env`（如 `DEEPSEEK_API_KEY`） |
| BGE 混合检索 | `requirements/knowledge.in` | `--knowledge-root kb --retrieval-mode hybrid_bge` |
| 断点续跑 | `requirements/persistence.in` | `python -m pvdiag.agent.sessions init/start/resume/show` |
| API 服务 | `requirements/api.in` | `python -m pvdiag.api init/serve` |
| Gradio 页面 | `requirements/ui.in`（单独的 `.venv-ui`） | `python -m pvdiag.ui --api-url http://127.0.0.1:8000` |
| 长期记忆 / A4 | persistence + knowledge | `python -m pvdiag.memory init/add-case/add-lesson/show`，诊断时 `--ablation A4 --memory-revision N` |
| 自进化 / A5 | 核心依赖 | `python -m pvdiag.evolve --proposer-file ...` 或 `--proposer-model ... --allow-proposer-calls` |
| 失败驱动场景生成 | 核心依赖 | `python -m pvdiag.sim.scenario_gen --manifest ... --summary ...` |
| Ragas | `requirements/ragas.in`（单独的 `.venv-eval`） | `python -m pvdiag.eval.ragas prepare/run` |

### 评测范围、留出集与跨站集

- `run_bench` 默认只跑开发集。`--scope heldout|cross_site --preflight-only` 只校验并冻结观测快照，不诊断、不消耗评测次数。
- 留出 / 跨站的正式评测需要 `--allow-final-evaluation --stage <阶段名>`，并写入一次性台账（默认 `data/eval/final_ledger.json`）；同一清单的同一阶段只能评测一次，失败也算用过。
- 生成器 `python -m pvdiag.sim.benchmark` 支持 `--split dev|test|cross_site`、`--fault-types` 子集、`--compound FA,FB --per-compound N`（hard 双故障）；两条带版本号的放宽筛查规则只在显式指定时生效：`--max-peer-outlier-fraction`（大电站同站离群容忍）与 `--daytime-poa-coverage 纬度 经度`（POA 覆盖率只统计日间）。清单会记录所用规则，审核与重建时自动沿用。

所有可能计费的调用都要显式开关（`--allow-model-calls`、`--allow-proposer-calls`、Ragas 的 `--allow-generation/--allow-judge`），并有调用次数上限；没有开关时会在发出任何请求之前拒绝。

## Docker Compose（PostgreSQL + API）

`docker-compose.yml` 启动带 pgvector 的 PostgreSQL 和诊断 API。API 端口只发布到宿主机的 `127.0.0.1:8000`；Gradio 页面仍在本机运行并连接这个地址。

1. 先按「快速开始」准备好 `data/` 下的场景数据（以只读方式挂进容器）。
2. 在被忽略的 `.env` 中设置 `PVDIAG_API_TOKENS`（身份与 token 的 SHA-256 摘要，格式见 `docs/DESIGN.md` 的 FastAPI 一节）和 `POSTGRES_PASSWORD`。
3. 启动：

```bash
docker compose up --build
```

首次启动会先运行一次显式的表初始化服务（`migrate`），成功后再启动 API。默认 profile 是 `config/api.example.json` 中的 B0，不调用大模型。

## 在 Claude Desktop 中使用诊断工具

`config/claude_desktop.example.json` 接入三个只读 MCP 服务（pv-data、pv-analytics、pv-knowledge，共 11 个工具），默认指向 M1 场景 `S2107-F3-0001`；写工单的 pv-ops 不接入。在仓库根目录运行下面的命令，输出就是填好本机路径的配置：

```bash
sed "s#REPO#$PWD#g" config/claude_desktop.example.json
```

把输出中的 `mcpServers` 合并进 `~/Library/Application Support/Claude/claude_desktop_config.json`，重启 Claude Desktop 即可在对话里调用这些工具。换场景时修改 `--observations` 指向的 Parquet 文件即可。

## 安全与数据边界

- 诊断只能读取观测快照，读不到真值、注入参数和基线审核信息；评分在诊断结束后才读取真值。
- 写操作（模拟工单）默认不加载，加载后也必须经人工审批，审批人身份由服务端绑定。
- 报告里的每个数字和时间都必须能沿 `evidence_id` + JSON Pointer 找回出处，否则降级为需人工复核。
- 评测入口只接受开发集；留出集保持关闭，长期记忆和自进化都不读取留出集。
- 凭据只从环境变量或被忽略的 `.env` 读取，不进入命令行、日志或结果文件。

## 数据与许可

- 观测数据：NREL PVDAQ 2023 Solar Data Prize（`oedi-data-lake` 公开桶），仓库不包含任何数据文件。
- 知识语料：pvlib 0.16.1 文档字符串（BSD-3-Clause）与项目自写的 Playbook，许可核对见 `kb/SOURCES.md`。
- 本地 BGE 模型（可选）：BAAI/bge-small-zh-v1.5 与 bge-reranker-base，许可见 `kb/licenses/`。
