import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from pvdiag.agent.__main__ import reasoner_from_args, run_diagnosis
from pvdiag.agent.gateway import MCPGateway
from pvdiag.agent.graph import build_graph
from pvdiag.agent.guards import check_report, resolve
from pvdiag.agent.llm import LLMReasoner
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.state import DiagnosisReport, DiagRequest, Plan, Scope, ToolCall
from pvdiag.mcp_servers.data_tools import DataTools
from pvdiag.mcp_servers.models import QualityRequest
from pvdiag.mcp_servers.pv_analytics import create_server as analytics_server
from pvdiag.mcp_servers.pv_data import create_server as data_server
from pvdiag.mcp_servers.store import ObservationStore


@pytest.fixture
def observation(tmp_path):
    ts = pd.date_range("2019-01-01", periods=7 * 96, freq="15min", tz="UTC")
    day = (ts.hour >= 16) & (ts.hour < 23)
    rows = []
    for device, metric, values in (
        ("inv01", "ac_power_kw", day.astype(float) * 10),
        ("inv02", "ac_power_kw", day.astype(float) * 10),
        ("inv03", "ac_power_kw", day.astype(float) * 10),
        ("poa", "poa_wm2", day.astype(float) * 600),
        ("ambient", "ambient_c", 20.0),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "ts": ts,
                    "device_id": device,
                    "metric": metric,
                    "system_id": 2107,
                    "value": values,
                }
            )
        )
    path = tmp_path / "observations.parquet"
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "System": {"system_id": 2107, "timezone_code": "PST8PDT"},
                "Metrics": {
                    f"inv_{i:02d}_power_{i}": {
                        "metric_id": i,
                        "common_name": "AC power",
                        "units": "kw",
                        "source_type": "INVERTER",
                        "source_id": None,
                    }
                    for i in range(1, 4)
                },
            }
        )
    )
    (tmp_path / "truth.json").write_text("poisoned truth: diagnosis must not open")
    return path, metadata


def request(**updates):
    return DiagRequest(
        **(
            {
                "system_id": 2107,
                "device_ids": ["inv01", "inv02", "inv03"],
                "start": "2019-01-01",
                "end": "2019-01-08",
            }
            | updates
        )
    )


def inject(observation, kind):
    data = pd.read_parquet(observation[0])
    mask = (data.device_id == "inv01") & (data.ts >= "2019-01-03")
    if kind == "F3":
        mask &= data.ts.between("2019-01-03 16:00Z", "2019-01-03 18:00Z", inclusive="left")
        data.loc[mask, "value"] = 0.0
    else:
        data.loc[mask, "value"] *= 0.75
    data.to_parquet(observation[0], index=False)


@pytest.mark.parametrize("kind", ["F3", "F4"])
def test_real_graph_generates_cited_report_and_respects_budgets(observation, kind):
    inject(observation, kind)
    result = asyncio.run(run_diagnosis(ObservationStore(*observation), request(), RuleReasoner()))
    assert result.report.status == "fault"
    assert result.report.findings[0].fault_type == kind
    assert result.report.findings[0].device_id == "inv01"
    assert result.loops == 3
    assert len(result.trace) <= 30 and all(t.success for t in result.trace)
    assert result.model["mode"] == "B0" and result.model_usage["model_calls"] == 0
    scope = Scope(
        source_sha256=next(iter(result.evidence.values()))["source_sha256"],
        device_ids=request().device_ids,
        poa_device_id="poa",
        ambient_device_id="ambient",
    )
    assert check_report(result.report, result.evidence, request(), scope) == []
    assert "poisoned truth" not in result.model_dump_json()


def test_output_guard_rejects_hallucinated_numbers_paths_times_and_scope(observation):
    inject(observation, "F3")
    result = asyncio.run(run_diagnosis(ObservationStore(*observation), request(), RuleReasoner()))
    scope = Scope(
        source_sha256=next(iter(result.evidence.values()))["source_sha256"],
        device_ids=request().device_ids,
        poa_device_id="poa",
        ambient_device_id="ambient",
    )
    tampered = result.report.model_copy(deep=True)
    tampered.findings[0].numbers[0].value = 999
    tampered.findings[0].device_id = "inv99"
    tampered.findings[0].window.start.value = pd.Timestamp("2018-01-01", tz="UTC")
    tampered.summary = "损失率为99"
    issues = check_report(tampered, result.evidence, request(), scope)
    assert {
        "invalid_numeric_citation",
        "invalid_time_citation",
        "finding_device_outside_scope",
        "finding_window_outside_scope",
        "uncited_number_in_summary",
    } <= set(issues)
    for path in ("/values/-1", "/values/01", "/bad~3"):
        with pytest.raises(ValueError):
            resolve({"values": [0]}, path)
    assert resolve({"a/b": {"~": [3]}}, "/a~1b/~0/0") == 3
    sensor_finding = result.report.model_copy(deep=True)
    sensor_finding.findings[0].fault_type = "F5"
    sensor_finding.findings[0].device_id = "poa"
    assert "finding_device_outside_scope" not in check_report(
        sensor_finding, result.evidence, request(), scope
    )
    sensor_finding.findings[0].device_id = "inv01"
    assert "finding_device_outside_scope" in check_report(
        sensor_finding, result.evidence, request(), scope
    )


class RepairReasoner(RuleReasoner):
    async def plan(self, state, tools, prompt):
        return Plan(calls=[])

    async def hypothesize(self, state, prompt):
        summary = "未经引用的42" if state["loops"] == 1 else "当前证据不足。"
        return DiagnosisReport(status="inconclusive", summary=summary)


def test_guard_retry_can_repair_and_does_not_keep_stale_critique(observation):
    result = asyncio.run(run_diagnosis(ObservationStore(*observation), request(), RepairReasoner()))
    assert result.loops == 2
    assert result.report.status == "inconclusive"
    assert result.report.summary == "当前证据不足。"


def test_invalid_plan_and_exhausted_budget_never_execute_writes(observation):
    class Malicious(RuleReasoner):
        async def plan(self, state, tools, prompt):
            return {"calls": [{"name": "create_work_order", "arguments": {}}]}

    result = asyncio.run(run_diagnosis(ObservationStore(*observation), request(), Malicious()))
    assert result.report.status == "needs_review" and result.loops == 3
    assert {t.name for t in result.trace} == {"list_devices", "get_site_info"}
    limited = asyncio.run(
        run_diagnosis(ObservationStore(*observation), request(), RuleReasoner(), max_calls=3)
    )
    assert len(limited.trace) == 3
    assert limited.report.status == "needs_review"


def test_input_guard_rejects_foreign_devices_and_out_of_domain_questions(observation):
    store = ObservationStore(*observation)
    for bad in (request(device_ids=["inv99"]), request(question="帮我订一份晚餐")):
        with pytest.raises(ValueError):
            asyncio.run(run_diagnosis(store, bad, RuleReasoner()))


def test_gateway_denies_cross_scope_calls_and_mixed_snapshots(observation, tmp_path):
    async def verify():
        store = ObservationStore(*observation)
        async with MCPGateway.connect(data_server(store), analytics_server(store)) as gateway:
            scope = Scope(
                source_sha256=store.source_sha256,
                device_ids=request().device_ids,
                poa_device_id="poa",
                ambient_device_id="ambient",
            )
            bad = ToolCall(
                name="data_quality",
                arguments=request().model_dump(mode="json", exclude={"question"})
                | {"system_id": 9069},
            )
            remote_calls = []

            async def forbidden(*args, **kwargs):
                remote_calls.append(args)
                raise RuntimeError("must not be reached")

            gateway.clients["data_quality"] = SimpleNamespace(call_tool=forbidden)
            result, trace = await gateway.call(bad, request(), scope, "investigate")
            assert result is None and not trace.success
            assert remote_calls == []
            arguments = request().model_dump(mode="json", exclude={"question"})
            forged = (
                DataTools(store).data_quality(QualityRequest(**arguments)).model_dump(mode="json")
            )
            forged["result"]["series"][0]["coverage"] = 0.5

            async def tampered(*args, **kwargs):
                return SimpleNamespace(structured_content=forged)

            gateway.clients["data_quality"] = SimpleNamespace(call_tool=tampered)
            result, trace = await gateway.call(
                ToolCall(name="data_quality", arguments=arguments), request(), scope, "investigate"
            )
            assert result is None and trace.error == "ValueError"
        other_path = tmp_path / "other.parquet"
        altered = pd.read_parquet(observation[0])
        altered.loc[0, "value"] += 1
        altered.to_parquet(other_path, index=False)
        other = ObservationStore(other_path, observation[1])
        async with MCPGateway.connect(data_server(store), analytics_server(other)) as gateway:
            arguments = request().model_dump(mode="json")
            arguments.pop("question")
            arguments |= {
                "poa_device_id": "poa",
                "reference_start": "2019-01-01",
                "reference_end": "2019-01-02",
            }
            result, trace = await gateway.call(
                ToolCall(name="peer_compare", arguments=arguments), request(), scope, "investigate"
            )
            assert result is None and trace.error == "ValueError"

    asyncio.run(verify())


def test_checkpoint_persists_completed_graph_state(observation):
    async def verify():
        store = ObservationStore(*observation)
        async with MCPGateway.connect(data_server(store), analytics_server(store)) as gateway:
            graph = build_graph(gateway, RepairReasoner())
            config = {"configurable": {"thread_id": "checkpoint-test"}}
            first = await graph.ainvoke({"request": request()}, config=config)
            snapshot = await graph.aget_state(config)
            assert snapshot.next == ()
            assert len(snapshot.values["trace"]) == len(first["trace"]) == 2
            assert snapshot.values["result"].report.status == "inconclusive"

    asyncio.run(verify())


def test_llm_adapter_uses_function_calling_and_records_usage_without_api(monkeypatch, observation):
    import pvdiag.agent.llm as module

    configured = {}

    class Fake:
        def with_structured_output(self, schema, **kwargs):
            assert kwargs == {"include_raw": True, "method": "function_calling"}
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            assert messages[0][0] == "system"
            return {
                "raw": AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
                    response_metadata={"model_name": "offline-fixture"},
                ),
                "parsed": Plan(calls=[])
                if self.schema is Plan
                else DiagnosisReport(status="inconclusive", summary="离线模型测试证据不足。"),
                "parsing_error": None,
            }

    def initialize(model, **kwargs):
        configured.update(kwargs)
        return Fake()

    monkeypatch.setattr(module, "init_chat_model", initialize)
    reasoner = LLMReasoner(
        "deepseek-chat", base_url="https://api.deepseek.com", api_key="TEST_SECRET"
    )
    asyncio.run(reasoner._call(Plan, {"task": "offline"}, "system prompt"))
    assert configured["max_tokens"] == 4096 and configured["max_retries"] == 0
    assert reasoner.usage["input_tokens"] == 11
    assert reasoner.metadata["reported_models"] == ["offline-fixture"]
    assert "TEST_SECRET" not in json.dumps(reasoner.metadata | reasoner.usage)
    plugged = LLMReasoner("offline-structured-model")
    run = asyncio.run(run_diagnosis(ObservationStore(*observation), request(), plugged))
    assert run.model["mode"] == "LLM" and run.model_usage["model_calls"] == 2
    assert run.report.status == "inconclusive"
    with pytest.raises(ValueError, match="--allow-model-calls"):
        reasoner_from_args(SimpleNamespace(mode="llm", allow_model_calls=False))


def test_failed_model_calls_mark_usage_unknown_and_strip_provider_error(monkeypatch):
    import pvdiag.agent.llm as module

    class Broken:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            raise ValueError("PRIVATE_API_KEY_SHOULD_NOT_LEAK")

    monkeypatch.setattr(module, "init_chat_model", lambda *args, **kwargs: Broken())
    reasoner = LLMReasoner("offline")
    with pytest.raises(RuntimeError, match="model_call_failed") as error:
        asyncio.run(reasoner._call(Plan, {}, "prompt"))
    assert "PRIVATE_API_KEY" not in str(error.value)
    assert reasoner.usage["usage_missing_calls"] == 1


def test_real_langchain_openai_adapter_accepts_structured_schemas_without_network():
    adapter = LLMReasoner("deepseek-chat", base_url="http://127.0.0.1:1", api_key="offline-dummy")
    for schema in (Plan, DiagnosisReport):
        runnable = adapter.model.with_structured_output(
            schema, method="function_calling", include_raw=True
        )
        assert callable(runnable.ainvoke)
    assert adapter.usage["model_calls"] == 0


def test_planner_tool_schemas_are_the_inner_request_contracts_the_gateway_validates(observation):
    # The first real DeepSeek run planned against FastMCP's {"request": ...} wrapper and
    # every investigate call failed validation; planners must see the inner schema.
    from pvdiag.agent.gateway import CONTRACTS

    async def verify():
        store = ObservationStore(*observation)
        async with MCPGateway.connect(data_server(store), analytics_server(store)) as gateway:
            for name, tool in gateway.tools.items():
                assert tool["input_schema"] == CONTRACTS[name][0].model_json_schema()
                assert "request" not in tool["input_schema"].get("properties", {})
            scope = Scope(
                source_sha256=store.source_sha256,
                device_ids=request().device_ids,
                poa_device_id="poa",
                ambient_device_id="ambient",
            )
            inner = request().model_dump(mode="json", exclude={"question"})
            ok = await gateway.call(
                ToolCall(name="data_quality", arguments=inner), request(), scope, "investigate"
            )
            wrapped = await gateway.call(
                ToolCall(name="data_quality", arguments={"request": inner}),
                request(),
                scope,
                "investigate",
            )
            return ok[1], wrapped[1]

    ok, wrapped = asyncio.run(verify())
    assert ok.success and not wrapped.success and wrapped.error == "ValidationError"
