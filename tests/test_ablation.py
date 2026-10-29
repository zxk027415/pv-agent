"""Offline model transport, real MCP tools: these are boundary tests, not LLM scores."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from test_agent import observation as observation  # noqa: PLC0414 -- re-export pytest fixture
from test_agent import request

from pvdiag.agent.__main__ import reasoner_from_args, run_diagnosis
from pvdiag.agent.ablation import CORE_LEVELS as LEVELS
from pvdiag.agent.ablation import resolve_ablation
from pvdiag.agent.gateway import MCPGateway
from pvdiag.agent.llm import LLMReasoner
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.runtime import open_runtime
from pvdiag.agent.state import DiagnosisReport, ModelReview, Plan, ToolCall
from pvdiag.knowledge.store import KnowledgeStore
from pvdiag.mcp_servers.pv_analytics import create_server as analytics_server
from pvdiag.mcp_servers.pv_data import create_server as data_server
from pvdiag.mcp_servers.pv_knowledge import create_server as knowledge_server
from pvdiag.mcp_servers.store import ObservationStore


def offline_model(monkeypatch, handler):
    import pvdiag.agent.llm as module

    attempts = []

    class OfflineSDK:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            payload = json.loads(messages[1][1])
            attempts.append(payload)
            parsed = handler(payload, self.schema)
            return {
                "raw": AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                    response_metadata={"model_name": "offline-ablation-fixture"},
                ),
                "parsed": parsed,
                "parsing_error": None,
            }

    monkeypatch.setattr(module, "init_chat_model", lambda *args, **kwargs: OfflineSDK())
    return attempts


def inconclusive(payload, schema):
    if schema is Plan:
        return Plan(calls=[])
    assert schema is DiagnosisReport
    return DiagnosisReport(status="inconclusive", summary="现有观测不足以区分故障原因。")


def test_all_levels_receive_identical_fixed_evidence_and_a0_never_plans(observation, monkeypatch):
    attempts = offline_model(monkeypatch, inconclusive)
    results, first_inputs = [], []
    store = ObservationStore(*observation)

    async def verify():
        for level in LEVELS:
            before = len(attempts)
            result = await run_diagnosis(
                store,
                request(),
                LLMReasoner("offline"),
                ablation=level,
                knowledge_store=KnowledgeStore() if level == "A3" else None,
            )
            results.append(result)
            first_inputs.append(attempts[before])
            assert result.ablation["level"] == level
            assert result.ablation["verification_feedback"] == (level == "A3")
            assert len(result.trace) == 8
            assert all(t.success and t.node in {"guard_in", "fixed_summary"} for t in result.trace)
            assert result.report.status == "inconclusive"
            assert result.model_usage["model_calls"] == (1 if level == "A0" else 2)
            assert result.model["reported_models"] == ["offline-ablation-fixture"]
            assert "poisoned truth" not in json.dumps(attempts)

    asyncio.run(verify())
    assert first_inputs[0]["task"] == "hypothesize" and "tools" not in first_inputs[0]
    assert len({r.ablation["summary_sha256"] for r in results}) == 1
    assert all(p["evidence"] == first_inputs[0]["evidence"] for p in first_inputs)
    for level, payload in zip(LEVELS[1:], first_inputs[1:], strict=True):
        policy, _ = resolve_ablation(level)
        assert {t["name"] for t in payload["tools"]} == policy.model_tools


@pytest.mark.parametrize("level", ["A0", "A1"])
def test_early_levels_do_not_construct_analytics_server(observation, monkeypatch, level):
    from pvdiag.agent import runtime

    def forbidden(*args):
        pytest.fail("analytics must not be mounted for A0/A1")

    monkeypatch.setattr(runtime, "analytics_server", forbidden)
    offline_model(monkeypatch, inconclusive)
    asyncio.run(
        run_diagnosis(
            ObservationStore(*observation), request(), LLMReasoner("offline"), ablation=level
        )
    )


@pytest.mark.parametrize(
    "level,tool,arguments",
    [
        ("A0", "data_quality", {}),
        ("A1", "peer_compare", {}),
        ("A2", "get_playbook", {"fault_type": "F3"}),
    ],
)
def test_gateway_rejects_forbidden_tools_before_remote_dispatch(
    observation, level, tool, arguments
):
    store, kb = ObservationStore(*observation), KnowledgeStore()

    async def verify():
        policy, _ = resolve_ablation(level)
        # Even if a forbidden server is accidentally mounted, the execution
        # boundary must still refuse it independently of the model's tool list.
        async with MCPGateway.connect(
            data_server(store),
            analytics_server(store),
            knowledge_server(kb),
            knowledge_source_sha256=kb.source_sha256,
            allowed_tools=policy.model_tools,
        ) as gateway:

            async def forbidden(*args, **kwargs):
                pytest.fail("forbidden calls must not reach MCP")

            gateway.clients[tool] = SimpleNamespace(call_tool=forbidden)
            evidence, trace = await gateway.call(
                ToolCall(name=tool, arguments=arguments), request(), None, "investigate"
            )
            assert evidence is None and not trace.success and trace.error == "ValueError"
            assert tool not in gateway.tools

    asyncio.run(verify())


def test_model_proposed_forbidden_call_is_recorded_without_leaking_evidence(
    observation, monkeypatch
):
    def handler(payload, schema):
        if schema is Plan:
            return Plan(calls=[ToolCall(name="get_playbook", arguments={"fault_type": "F3"})])
        assert all(ev["tool"] != "get_playbook" for ev in payload["evidence"].values())
        return inconclusive(payload, schema)

    offline_model(monkeypatch, handler)
    result = asyncio.run(
        run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline"),
            ablation="A1",
        )
    )
    assert result.trace[-1].name == "get_playbook" and not result.trace[-1].success
    assert result.trace[-1].node == "investigate"
    assert result.model_usage["model_calls"] == 2


@pytest.mark.parametrize("level", ["A1", "A2"])
def test_structural_findings_never_reach_early_level_planning(observation, monkeypatch, level):
    plans = []

    def handler(payload, schema):
        if schema is Plan:
            plans.append(payload)
            assert payload["critique"] == payload["verification_issues"] == []
            return Plan(calls=[], ready_for_verdict=len(plans) > 1)
        return DiagnosisReport(status="inconclusive", summary="未经引用的数字 123。")

    offline_model(monkeypatch, handler)
    result = asyncio.run(
        run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline"),
            ablation=level,
        )
    )
    assert len(plans) == result.loops == 2
    assert all(not item.passed for item in result.verification)
    assert result.report.status == "needs_review"


@pytest.mark.parametrize("level", ["A0", "A1", "A2"])
def test_failed_structural_check_cannot_trigger_early_level_repair_loop(
    observation, monkeypatch, level
):
    def handler(payload, schema):
        return (
            Plan(calls=[])
            if schema is Plan
            else DiagnosisReport(status="inconclusive", summary="123")
        )

    attempts = offline_model(monkeypatch, handler)
    result = asyncio.run(
        run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline"),
            ablation=level,
        )
    )
    assert result.report.status == "needs_review" and result.loops == 1
    assert len(attempts) == (1 if level == "A0" else 2)


def test_a3_uses_knowledge_domain_feedback_and_model_review_within_shared_budget(
    observation, monkeypatch
):
    plans = []

    def handler(payload, schema):
        if schema is Plan:
            plans.append(payload)
            if len(plans) == 1:
                return Plan(calls=[ToolCall(name="get_playbook", arguments={"fault_type": "F3"})])
            assert any(
                i["code"] == "normal_peers_unverified" for i in payload["verification_issues"]
            )
            assert payload["critique"]
            return Plan(calls=[])
        if schema is ModelReview:
            assert payload["task"] == "verify" and payload["rule_issues"]
            return ModelReview(passed=True, issues=[])
        assert schema is DiagnosisReport
        if len(plans) == 1:
            return DiagnosisReport(
                status="normal", summary="设备观测稳定。", evidence_ids=list(payload["evidence"])
            )
        return DiagnosisReport(status="inconclusive", summary="缺少充分的独立佐证。")

    attempts = offline_model(monkeypatch, handler)
    result = asyncio.run(
        run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline", max_model_calls=5),
            ablation="A3",
            knowledge_store=KnowledgeStore(),
        )
    )
    assert [p["task"] for p in attempts] == ["plan", "hypothesize", "verify", "plan", "hypothesize"]
    assert result.model_usage == {
        "model_calls": 5,
        "input_tokens": 35,
        "output_tokens": 15,
        "usage_missing_calls": 0,
    }
    assert result.verification[0].model_reviewed and not result.verification[0].passed
    assert result.report.status == "inconclusive" and result.verification_mode == "domain_llm"
    assert any(t.name == "get_playbook" and t.success for t in result.trace)


@pytest.mark.parametrize(
    "case", ["b0", "knowledge_missing", "knowledge_extra", "ops", "verification", "level"]
)
def test_bad_configuration_fails_before_tools_or_model_requests(observation, monkeypatch, case):
    from pvdiag.agent import runtime

    attempts = offline_model(monkeypatch, inconclusive)
    options = {"ablation": "A1"}
    reasoner = LLMReasoner("offline")
    if case == "b0":
        reasoner = RuleReasoner()
    elif case == "knowledge_missing":
        options["ablation"] = "A3"
    elif case == "knowledge_extra":
        options["knowledge_store"] = KnowledgeStore()
    elif case == "ops":
        options["ops_store"] = object()
    elif case == "verification":
        options["verification_mode"] = "domain"
    elif case == "level":
        options["ablation"] = "A5"

    def forbidden(*args):
        pytest.fail("invalid configuration must fail before mounting tools")

    monkeypatch.setattr(runtime, "data_server", forbidden)
    with pytest.raises(ValueError):
        asyncio.run(run_diagnosis(ObservationStore(*observation), request(), reasoner, **options))
    assert attempts == []


def test_incomplete_fixed_summary_never_reaches_model(observation, monkeypatch):
    attempts = offline_model(monkeypatch, inconclusive)
    with pytest.raises(ValueError, match="complete fixed summary"):
        asyncio.run(
            run_diagnosis(
                ObservationStore(*observation),
                request(),
                LLMReasoner("offline"),
                ablation="A0",
                max_calls=7,
            )
        )
    assert attempts == []


def test_cli_refuses_ablation_with_b0_before_credentials():
    with pytest.raises(ValueError, match="--mode llm"):
        reasoner_from_args(SimpleNamespace(mode="b0", ablation="A1"))


@pytest.mark.parametrize("level,tool", [("A1", "data_quality"), ("A2", "profile_pattern")])
def test_allowed_tools_reach_real_mcp_and_return_citable_evidence(
    observation, monkeypatch, level, tool
):
    arguments = request().model_dump(mode="json", exclude={"question"})
    if level == "A2":
        arguments |= {
            "poa_device_id": "poa",
            "reference_start": "2019-01-01",
            "reference_end": "2019-01-02",
        }

    def handler(payload, schema):
        if schema is Plan:
            return Plan(calls=[ToolCall(name=tool, arguments=arguments)])
        assert any(
            ev["tool"] == tool and ev["request"]["device_ids"] == request().device_ids
            for ev in payload["evidence"].values()
        )
        return inconclusive(payload, schema)

    offline_model(monkeypatch, handler)
    result = asyncio.run(
        run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline"),
            ablation=level,
        )
    )
    assert result.trace[-1].success and result.trace[-1].name == tool
    assert result.trace[-1].node == "investigate"
    assert result.evidence[result.trace[-1].evidence_id]["tool"] == tool


def test_a3_review_cannot_exceed_the_shared_model_budget(observation, monkeypatch):
    def handler(payload, schema):
        if schema is Plan:
            return Plan(calls=[])
        assert schema is DiagnosisReport
        return DiagnosisReport(
            status="normal", summary="设备观测稳定。", evidence_ids=list(payload["evidence"])
        )

    attempts = offline_model(monkeypatch, handler)
    result = asyncio.run(
        run_diagnosis(
            ObservationStore(*observation),
            request(),
            LLMReasoner("offline", max_model_calls=2),
            ablation="A3",
            knowledge_store=KnowledgeStore(),
            max_loops=1,
        )
    )
    assert len(attempts) == result.model_usage["model_calls"] == 2
    assert result.model_usage["usage_missing_calls"] == 0
    assert result.report.status == "needs_review"
    assert not result.verification[-1].model_reviewed
    assert "model_review_failed" in {i.code for i in result.verification[-1].issues}


def test_persistent_contract_binds_ablation_configuration(observation, monkeypatch):
    offline_model(monkeypatch, inconclusive)

    async def verify():
        contracts = []
        for level in ("A1", "A2"):
            async with open_runtime(
                ObservationStore(*observation), LLMReasoner("offline"), ablation=level
            ) as runtime:
                contracts.append(runtime.contract)
        assert contracts[0]["ablation"]["level"] == "A1"
        assert contracts[1]["ablation"]["level"] == "A2"
        assert contracts[0] != contracts[1]

    asyncio.run(verify())


def test_report_author_sees_verification_feedback_except_in_feedback_free_ablations(
    observation, monkeypatch
):
    calls = {"n": 0}

    def handler(payload, schema):
        if schema is Plan:
            return Plan(calls=[])
        calls["n"] += 1
        summary = "第 1 轮结论" if calls["n"] == 1 else "现有观测不足以区分故障原因。"
        return DiagnosisReport(status="inconclusive", summary=summary)

    attempts = offline_model(monkeypatch, handler)
    asyncio.run(run_diagnosis(ObservationStore(*observation), request(), LLMReasoner("offline")))
    hypotheses = [a for a in attempts if a["task"] == "hypothesize"]
    assert hypotheses[0]["verification_issues"] == []
    second = hypotheses[1]["verification_issues"]
    assert [i["code"] for i in second] == ["uncited_number_in_summary"]
    assert "数字字符" in second[0]["message"]

    calls["n"] = 0
    attempts.clear()
    asyncio.run(
        run_diagnosis(ObservationStore(*observation), request(), LLMReasoner("offline"), ablation="A1")
    )
    hypotheses = [a for a in attempts if a["task"] == "hypothesize"]
    assert all(h["verification_issues"] == [] and h["critique"] == [] for h in hypotheses)


def test_schema_failures_are_reported_by_field_to_the_next_plan_and_hypothesis(
    observation, monkeypatch
):
    import pvdiag.agent.llm as module

    try:
        DiagnosisReport.model_validate({"status": "fault", "summary": "SECRET_OUTPUT_TEXT"})
    except ValueError as error:
        invalid = error
    attempts = []

    class SDK:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            payload = json.loads(messages[1][1])
            attempts.append(payload)
            first_hypothesis = self.schema is DiagnosisReport and sum(a["task"] == "hypothesize" for a in attempts) == 1
            parsed = (
                None
                if first_hypothesis
                else Plan(calls=[])
                if self.schema is Plan
                else DiagnosisReport(status="inconclusive", summary="现有观测不足以区分故障原因。")
            )
            return {
                "raw": AIMessage(content="", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
                "parsed": parsed,
                "parsing_error": invalid if first_hypothesis else None,
            }

    monkeypatch.setattr(module, "init_chat_model", lambda *a, **k: SDK())
    asyncio.run(run_diagnosis(ObservationStore(*observation), request(), LLMReasoner("offline")))
    plans = [a for a in attempts if a["task"] == "plan"]
    hypotheses = [a for a in attempts if a["task"] == "hypothesize"]
    hints = [c for c in plans[1]["critique"] if c.startswith("上一轮报告未通过结构化 schema 校验")]
    assert hints and any("findings" in h or "evidence_ids" in h for h in hints)
    carried = [i for i in hypotheses[1]["verification_issues"] if i["code"] == "invalid_structured_output"]
    assert carried and carried[0]["message"].startswith("上一轮报告未通过结构化 schema 校验")
    assert "SECRET_OUTPUT_TEXT" not in json.dumps(attempts[2:], ensure_ascii=False)
