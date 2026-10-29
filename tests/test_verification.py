import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from pvdiag.agent.__main__ import reasoner_from_args, run_diagnosis
from pvdiag.agent.guards import check_report
from pvdiag.agent.llm import LLMReasoner
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.state import (
    DiagnosisReport,
    DiagRequest,
    Finding,
    ModelReview,
    Plan,
    Scope,
    ToolCall,
    VerificationIssue,
)
from pvdiag.agent.verification import check_domain
from pvdiag.mcp_servers.analytics_tools import AnalyticsTools
from pvdiag.mcp_servers.data_tools import DataTools
from pvdiag.mcp_servers.models import (
    DetectRequest,
    ExpectedRequest,
    PeerRequest,
    QualityRequest,
    SiteRequest,
    TimeseriesRequest,
)
from pvdiag.mcp_servers.store import ObservationStore
from pvdiag.sim.faults import FaultSpec, inject_fault

DEVICES = ["inv01", "inv02", "inv03"]


def context(tmp_path, kind=None, *, parameters=False):
    """Controlled data goes through real tools; no tool result fields are invented."""
    ts = pd.date_range("2019-01-01", periods=42 * 96, freq="15min", tz="UTC")
    sun = ts.hour >= 16
    parts = []
    for device, metric, values in [
        *[(d, "ac_power_kw", sun.astype(float) * 10) for d in DEVICES],
        ("poa", "poa_wm2", sun.astype(float) * 600),
        ("ambient", "ambient_c", 20.0),
        ("inv01", "module_c", 30.0),
    ]:
        parts.append(
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
    frame = pd.concat(parts, ignore_index=True)
    start, end = "2019-01-04T00:00:00Z", "2019-02-05T00:00:00Z"
    if kind in {"F3", "F6"}:
        start, end = "2019-01-04T16:00:00Z", "2019-01-04T18:00:00Z"
    specs = {
        "F1": {"soiling_rate_per_day": 0.005, "cleaning_times": ("2019-01-28T00:00:00Z",)},
        "F2": {"severity": 0.5, "daily_window": ("15:00", "17:00"), "timezone": "PST8PDT"},
        "F3": {"poa_device_id": "poa"},
        "F4": {"loss_fraction": 0.25},
        "F5": {"drift_per_day": 0.005},
        "F6": {},
    }
    if kind:
        fault = FaultSpec(kind, "poa" if kind == "F5" else "inv01", start, end, **specs[kind])
        frame, _ = inject_fault(frame, 2107, fault)
    else:
        start, end = "2019-01-01T00:00:00Z", "2019-02-12T00:00:00Z"
    observation, metadata = tmp_path / "observations.parquet", tmp_path / "metadata.json"
    frame.to_parquet(observation, index=False)
    metadata.write_text(
        json.dumps(
            {
                "System": {"system_id": 2107, "timezone_code": "PST8PDT"},
                "Metrics": {
                    f"inv_{i:02d}_ac_power_{i}": {
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
    parameter_file = tmp_path / "parameters.json"
    if parameters:
        parameter_file.write_text(
            json.dumps(
                {
                    d: {
                        "rated_dc_kw": 20,
                        "rated_ac_kw": 15,
                        "gamma_pdc": -0.004,
                        "source": "synthetic fixture only",
                    }
                    for d in DEVICES
                }
            )
        )
    store = ObservationStore(observation, metadata, parameter_file if parameters else None)
    request = DiagRequest(system_id=2107, device_ids=DEVICES, start="2019-01-01", end="2019-02-12")
    scope = Scope(
        source_sha256=store.source_sha256,
        device_ids=DEVICES,
        poa_device_id="poa",
        ambient_device_id="ambient",
    )
    data, analytics = DataTools(store), AnalyticsTools(store)
    args = {"system_id": 2107, "device_ids": DEVICES, "start": start, "end": end}
    peer = args | {
        "poa_device_id": "poa",
        "reference_start": "2019-01-01",
        "reference_end": "2019-01-03",
    }
    outputs = [
        data.list_devices(SiteRequest(system_id=2107)),
        data.data_quality(QualityRequest(**args, metrics=["ac_power_kw", "module_c"])),
        data.data_quality(QualityRequest(**(args | {"device_ids": ["poa"]}), metrics=["poa_wm2"])),
        data.get_timeseries(TimeseriesRequest(**args)),
        analytics.peer_compare(PeerRequest(**peer)),
        analytics.profile_pattern(PeerRequest(**peer)),
        analytics.detect_anomalies(DetectRequest(**peer)),
    ]
    expected = args | {"poa_device_id": "poa", "ambient_device_id": "ambient"}
    outputs += [
        analytics.expected_power(ExpectedRequest(**expected)),
        analytics.expected_power(
            ExpectedRequest(**(expected | {"start": "2019-01-01", "end": "2019-01-03"}))
        ),
    ]
    evidence = {item.evidence_id: item.model_dump(mode="json") for item in outputs}
    citations = list(evidence)
    if kind:
        timing = outputs[3].evidence_id
        finding = Finding(
            fault_type=kind,
            device_id="poa" if kind == "F5" else "inv01",
            confidence="medium",
            window={
                "start": {"value": start, "evidence_id": timing, "path": "/request/start"},
                "end": {"value": end, "evidence_id": timing, "path": "/request/end"},
            },
            rationale="受控数据用于测试核验规则。",
            evidence_ids=citations,
        )
        report = DiagnosisReport(
            status="fault", summary="受控假设。", findings=[finding], evidence_ids=citations
        )
    else:
        report = DiagnosisReport(
            status="normal", summary="受控数据未见异常。", evidence_ids=citations
        )
    return store, request, scope, evidence, report


@pytest.mark.parametrize("kind", ["F1", "F2", "F3", "F4", "F5", "F6"])
def test_six_domain_signatures_are_supported_by_real_tool_outputs(tmp_path, kind):
    _, request, scope, evidence, report = context(tmp_path, kind, parameters=kind == "F5")
    assert check_report(report, evidence, request, scope) == []
    assert check_domain(report, evidence, request, scope) == []


@pytest.mark.parametrize(
    "actual,claimed,expected",
    [
        ("F6", "F3", "power_quality_unverified"),
        ("F3", "F6", "communication_gap_unverified"),
        ("F2", "F4", "shading_not_excluded"),
        ("F4", "F2", "shading_repeat_unverified"),
    ],
)
def test_true_citations_do_not_validate_a_contradictory_fault_label(
    tmp_path, actual, claimed, expected
):
    _, request, scope, evidence, report = context(tmp_path, actual)
    report.findings[0].fault_type = claimed
    assert check_report(report, evidence, request, scope) == []
    assert expected in {issue.code for issue in check_domain(report, evidence, request, scope)}


def test_sensor_drift_and_normal_need_physics_not_just_stable_peer_ratios(tmp_path):
    for kind in ("F5", None):
        _, request, scope, evidence, report = context(tmp_path, kind)
        codes = {issue.code for issue in check_domain(report, evidence, request, scope)}
        assert ("sensor_drift_physics_unverified" if kind else "common_mode_not_excluded") in codes
    _, request, scope, evidence, report = context(tmp_path, None, parameters=True)
    assert check_domain(report, evidence, request, scope) == []


def test_profile_from_other_window_or_source_cannot_support_a_finding(tmp_path):
    _, request, scope, evidence, report = context(tmp_path, "F4")
    altered = deepcopy(evidence)
    for item in altered.values():
        if item["tool"] == "profile_pattern":
            item["request"]["start"] = "2019-01-05T00:00:00Z"
    assert "fault_profile_missing" in {
        i.code for i in check_domain(report, altered, request, scope)
    }
    for item in altered.values():
        item["source_sha256"] = "different_snapshot"
    assert "power_quality_unverified" in {
        i.code for i in check_domain(report, altered, request, scope)
    }


def test_normal_verdict_cannot_omit_a_known_daylight_outage(tmp_path):
    store, request, scope, evidence, _ = context(tmp_path, "F3", parameters=True)
    args = request.model_dump(exclude={"question"})
    peers = args | {
        "poa_device_id": "poa",
        "reference_start": "2019-01-01",
        "reference_end": "2019-01-03",
    }
    analytics = AnalyticsTools(store)
    for item in [
        analytics.detect_anomalies(DetectRequest(**peers)),
        analytics.peer_compare(PeerRequest(**peers)),
    ]:
        evidence[item.evidence_id] = item.model_dump(mode="json")
    # Even omitting the anomaly receipt from the report must not hide it from verification.
    cited = [key for key, value in evidence.items() if value["tool"] != "detect_anomalies"]
    report = DiagnosisReport(status="normal", summary="未见异常。", evidence_ids=cited)
    assert "normal_anomaly_unresolved" in {
        i.code for i in check_domain(report, evidence, request, scope)
    }


def test_model_approval_cannot_override_domain_failures(tmp_path):
    store, request, _, _, _ = context(tmp_path, "F3")

    class WrongRule(RuleReasoner):
        async def hypothesize(self, state, prompt):
            result = await super().hypothesize(state, prompt)
            for finding in result.findings:
                finding.fault_type = "F4"
            return result

        async def review(self, state, issues, prompt):
            return ModelReview(passed=True)

    result = asyncio.run(run_diagnosis(store, request, WrongRule(), verification_mode="domain_llm"))
    assert result.report.status == "needs_review"
    assert result.verification[-1].model_reviewed
    assert "string_step_unverified" in {i.code for i in result.verification[-1].issues}
    assert result.verification_mode == "domain_llm"


def test_reviewer_failure_and_invalid_citations_are_not_a_pass(tmp_path):
    store, request, _, _, _ = context(tmp_path, "F3")

    class BrokenReview(RuleReasoner):
        async def review(self, state, issues, prompt):
            return ModelReview(
                passed=False,
                issues=[
                    VerificationIssue(
                        code="unsupported", message="测试问题。", evidence_ids=["invented_evidence"]
                    )
                ],
            )

    result = asyncio.run(
        run_diagnosis(store, request, BrokenReview(), verification_mode="domain_llm")
    )
    assert result.report.status == "needs_review"
    assert not result.verification[-1].model_reviewed
    assert "model_review_failed" in {i.code for i in result.verification[-1].issues}


def test_review_feedback_reaches_replanning_and_a_later_pass_clears_it(tmp_path):
    store, request, _, _, _ = context(tmp_path, "F3")

    class EagerReview(RuleReasoner):
        async def plan(self, state, tools, prompt):
            if state["loops"]:
                assert state["verification"][-1].issues[0].code == "confirm_reference"
                assert state["critique"]
            args = {
                "system_id": 2107,
                "device_ids": DEVICES,
                "start": "2019-01-04T16:00:00Z",
                "end": "2019-01-04T18:00:00Z",
            }
            peer = args | {
                "poa_device_id": "poa",
                "reference_start": "2019-01-01",
                "reference_end": "2019-01-03",
            }
            return Plan(
                calls=[
                    ToolCall(name="data_quality", arguments=args),
                    *[
                        ToolCall(name=name, arguments=peer)
                        for name in ("peer_compare", "profile_pattern", "detect_anomalies")
                    ],
                ]
            )

        async def review(self, state, issues, prompt):
            if state["loops"] == 1:
                key = next(
                    k for k, item in state["evidence"].items() if item["tool"] == "peer_compare"
                )
                return ModelReview(
                    passed=False,
                    issues=[
                        VerificationIssue(
                            code="confirm_reference",
                            message="测试要求重新核对参考期。",
                            finding_index=0,
                            evidence_ids=[key],
                            suggested_tools=["peer_compare"],
                        )
                    ],
                )
            return ModelReview(passed=True)

    result = asyncio.run(
        run_diagnosis(store, request, EagerReview(), verification_mode="domain_llm")
    )
    assert result.report.status == "fault" and result.loops == 2
    assert [record.passed for record in result.verification] == [False, True]
    assert all(record.model_reviewed for record in result.verification)
    assert len(result.verification_policy_sha256) == 64


def test_model_budget_is_shared_with_review_and_prevents_extra_network_attempts(monkeypatch):
    import pvdiag.agent.llm as module

    attempts = []

    class Fake:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            attempts.append(messages)
            return {
                "raw": AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                ),
                "parsed": Plan(calls=[]),
                "parsing_error": None,
            }

    monkeypatch.setattr(module, "init_chat_model", lambda *args, **kwargs: Fake())
    reasoner = LLMReasoner("offline", max_model_calls=1)
    asyncio.run(reasoner._call(Plan, {}, "prompt"))
    with pytest.raises(ValueError, match="model_call_budget_exhausted"):
        asyncio.run(reasoner._call(ModelReview, {}, "prompt"))
    assert len(attempts) == reasoner.usage["model_calls"] == 1


def test_review_schema_requires_specific_issues_for_rejection():
    with pytest.raises(ValueError):
        ModelReview(passed=False, issues=[])
    with pytest.raises(ValueError):
        ModelReview(passed=True, issues=[VerificationIssue(code="unsupported", message="矛盾。")])


def test_endpoint_credentials_are_not_implicitly_forwarded_to_another_provider(
    monkeypatch, tmp_path
):
    import pvdiag.agent.llm as module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fixture-only")
    configured = []
    monkeypatch.setattr(module, "LLMReasoner", lambda *args, **kwargs: configured.append(kwargs))
    args = SimpleNamespace(
        mode="llm",
        allow_model_calls=True,
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
    )
    with pytest.raises(ValueError, match="set DEEPSEEK_API_KEY"):
        reasoner_from_args(args)
    assert configured == []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-fixture-only")
    reasoner_from_args(args)
    assert configured[-1]["api_key"] == "deepseek-fixture-only"
    with pytest.raises(ValueError, match="explicit API key"):
        LLMReasoner("offline", base_url="https://api.deepseek.com")


def test_every_structural_guard_code_has_specific_repair_guidance():
    import re
    from pathlib import Path

    from pvdiag.agent import guards
    from pvdiag.agent.graph import STRUCTURAL_GUIDANCE

    codes = set(re.findall(r'issues\.append\("([a-z_]+)"\)', Path(guards.__file__).read_text()))
    assert codes and codes == set(STRUCTURAL_GUIDANCE)
    assert all("结构化输出或引用校验失败" not in text for text in STRUCTURAL_GUIDANCE.values())


def test_only_exact_in_scope_identifiers_are_exempt_from_the_digit_guard():
    from pvdiag.agent.guards import has_uncited_digits
    from pvdiag.agent.state import Scope

    scope = Scope(
        source_sha256="0" * 64,
        device_ids=["inv03", "inv04", "inv05"],
        poa_device_id="sensor_149574",
        ambient_device_id=None,
    )
    for allowed in ("inv03在午后持续零功率", "inv03 与 inv04、inv05 对比", "sensor_149574 读数完整", "无数字"):
        assert not has_uncited_digits(allowed, scope), allowed
    for blocked in (
        "inv03 在 1 月 12 日停机",
        "inv99 异常",
        "inv035 异常",
        "持续约 3 小时",
        "判定为 F3",
        "比值为 0.44",
        "19:30 起",
    ):
        assert has_uncited_digits(blocked, scope), blocked
