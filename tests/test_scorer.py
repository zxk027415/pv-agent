from copy import deepcopy

import pytest

from pvdiag.agent.state import DiagnosisReport, RunResult, ToolTrace
from pvdiag.eval.run_bench import run_benchmark
from pvdiag.eval.scorer import aggregate, maximum_matching, numeric_value, score_case, temporal_iou


def truth(faults=None, identifier="case"):
    return {
        "scenario_id": identifier,
        "system_id": 2107,
        "query_window": ["2019-01-01", "2019-01-08"],
        "faults": [{"type": "F3", "device_id": "inv01", "start": "2019-01-03", "end": "2019-01-04"}]
        if faults is None
        else faults,
    }


def finding(kind="F3", device="inv01", start="2019-01-03T00:00:00Z", end="2019-01-04T00:00:00Z"):
    return {
        "fault_type": kind,
        "device_id": device,
        "confidence": "medium",
        "window": {
            "start": {"value": start, "evidence_id": "ev_a", "path": "/start"},
            "end": {"value": end, "evidence_id": "ev_a", "path": "/end"},
        },
        "rationale": "测试证据。",
        "evidence_ids": ["ev_a", "ev_b"],
        "numbers": [{"label": "占比", "value": 0.5, "evidence_id": "ev_a", "path": "/value"}],
    }


def run(findings=None, status=None):
    findings = [finding()] if findings is None else findings
    return RunResult(
        request={
            "system_id": 2107,
            "device_ids": ["inv01", "inv02", "inv03"],
            "start": "2019-01-01",
            "end": "2019-01-08",
        },
        report=DiagnosisReport(
            status=status or ("fault" if findings else "normal"),
            summary="测试报告。",
            findings=findings,
            evidence_ids=["ev_a", "ev_b"],
        ),
        evidence={"ev_a": {"value": 0.5}},
        trace=[],
        loops=1,
        model={"mode": "offline-fixture"},
        model_usage={"input_tokens": 0, "output_tokens": 0},
        harness_sha256="fixture",
        duration_seconds=1.0,
    )


@pytest.mark.parametrize(
    "a,b,c,d,expected",
    [
        ("2019-01-01", "2019-01-03", "2019-01-01", "2019-01-03", 1),
        ("2019-01-01", "2019-01-02", "2019-01-02", "2019-01-03", 0),
        ("2019-01-01", "2019-01-03", "2019-01-02", "2019-01-04", 1 / 3),
        (
            "2019-01-01T00:00-08:00",
            "2019-01-02T00:00-08:00",
            "2019-01-01T08:00Z",
            "2019-01-02T08:00Z",
            1,
        ),
    ],
)
def test_temporal_iou_hand_calculated_cases(a, b, c, d, expected):
    assert temporal_iou(a, b, c, d) == pytest.approx(expected)


def test_matching_is_one_to_one_and_can_reassign_earlier_matches():
    assert maximum_matching([[0, 1], [0]]) == 2
    assert maximum_matching([[0], [0]]) == 1
    assert maximum_matching([]) == 0


def test_extra_findings_cannot_game_root_device_or_temporal_metrics():
    correct = score_case(truth(), run())
    assert correct["root_top1"] and correct["device_correct"]
    assert correct["temporal_hits"] == correct["temporal_denominator"] == 1
    extra = score_case(truth(), run([finding(), finding()]))
    assert not extra["root_top1"] and not extra["device_correct"]
    assert not extra["root_top3"]
    assert extra["temporal_hits"] == 1 and extra["temporal_denominator"] == 2


def test_root_alternatives_and_device_mapping_are_scored_separately():
    predicted = finding("F4")
    predicted["alternatives"] = ["F3"]
    case = score_case(truth(), run([predicted]))
    assert not case["root_top1"] and case["root_top3"]
    assert case["temporal_hits"] == 0
    faults = truth()["faults"] + [
        {"type": "F4", "device_id": "inv02", "start": "2019-01-03", "end": "2019-01-04"}
    ]
    swapped = score_case(truth(faults), run([finding("F3", "inv02"), finding("F4", "inv01")]))
    assert swapped["root_top1"] and swapped["device_correct"]
    assert not swapped["joint_root_device_correct"]
    assert swapped["temporal_hits"] == 0


def test_abstention_and_normal_false_alarm_denominators_are_not_hidden():
    cases = [
        score_case(truth(identifier="tp"), run()),
        score_case(truth(identifier="fn"), run([], "inconclusive")),
        score_case(truth([], "fp"), run()),
        score_case(truth([], "normal-abstain"), run([], "inconclusive")),
        score_case(truth([], "tn"), run([])),
    ]
    metrics = aggregate(cases)
    assert metrics["detection_precision"] == 0.5
    assert metrics["detection_recall"] == 0.5
    assert metrics["detection_f1"] == 0.5
    assert metrics["normal_false_alarm_rate"] == pytest.approx(1 / 3)
    assert metrics["normal_accuracy"] == pytest.approx(1 / 3)
    assert metrics["abstention_rate"] == 0.4
    missing_normals = aggregate(cases[:2])
    assert missing_normals["normal_false_alarm_rate"] is None
    assert missing_normals["evidence_faithfulness_judge"] is None
    assert missing_normals["cost_usd"] is None


def test_invalid_numeric_claims_unknown_usage_and_write_traces_are_visible():
    response = run()
    response.report.findings[0].numbers[0].value = 999
    response.model_usage["usage_missing_calls"] = 1
    response.trace = [
        ToolTrace(name="create_work_order", node="act", success=True, duration_seconds=0.1)
    ]
    case = score_case(truth(), response)
    metrics = aggregate([case])
    assert metrics["numeric_citation_accuracy"] == 0
    assert metrics["unapproved_write_executions"] == 1
    assert not metrics["usage_complete"] and metrics["input_tokens"] is None
    assert metrics["known_input_tokens"] == 0
    for path in ("/array/-1", "/array/01", "/bad~2"):
        with pytest.raises(ValueError):
            numeric_value({"array": [0]}, path)


def test_scorer_rejects_mismatched_truth_and_duplicate_case_ids():
    wrong = deepcopy(truth())
    wrong["system_id"] = 9069
    with pytest.raises(ValueError, match="system differ"):
        score_case(wrong, run())
    wrong = deepcopy(truth())
    wrong["faults"][0]["device_id"] = "inv99"
    with pytest.raises(ValueError, match="outside request"):
        score_case(wrong, run())
    case = score_case(truth(), run())
    with pytest.raises(ValueError, match="unique"):
        aggregate([case, case])


def test_runner_refuses_heldout_before_opening_any_observations(tmp_path):
    import asyncio

    scenario = truth()
    scenario["split"] = "test"
    with pytest.raises(ValueError, match="only permits development"):
        asyncio.run(
            run_benchmark(
                {"scenarios": [scenario]},
                tmp_path,
                tmp_path / "missing",
                tmp_path / "out",
                lambda: None,
            )
        )


def test_irradiance_fault_localization_uses_observed_poa_sensor_not_inverter_scope():
    actual = truth([{"type": "F5", "device_id": "poa", "start": "2019-01-03", "end": "2019-01-04"}])
    result = run([finding("F5", "poa")])
    result.evidence["catalog"] = {
        "tool": "list_devices",
        "result": {"devices": [{"device_id": "poa", "metrics": ["poa_wm2"]}]},
    }
    scored = score_case(actual, result)
    assert scored["device_correct"] and scored["joint_root_device_correct"]
    assert aggregate([scored])["scorer_version"] == "v1.1"
    actual["faults"][0]["device_id"] = "other_sensor"
    with pytest.raises(ValueError, match="outside request"):
        score_case(actual, result)
