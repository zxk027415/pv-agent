import asyncio

import pandas as pd
import pytest
from test_verification import context

from pvdiag.agent.__main__ import run_diagnosis
from pvdiag.agent.guards import check_report
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.state import Scope
from pvdiag.agent.verification import check_domain
from pvdiag.mcp_servers.data_tools import DataTools
from pvdiag.mcp_servers.models import QualityRequest
from pvdiag.mcp_servers.store import ObservationStore
from pvdiag.sim.faults import FaultSpec, inject_fault


def run(store, request, mode="domain"):
    result = asyncio.run(run_diagnosis(store, request, RuleReasoner(), verification_mode=mode))
    assert len(result.trace) <= 30 and all(t.success for t in result.trace)
    assert result.model["model"] == "rule-v2" and result.model_usage["model_calls"] == 0
    scope = Scope(
        source_sha256=store.source_sha256,
        device_ids=request.device_ids,
        poa_device_id="poa" if "poa" in store.catalog else None,
        ambient_device_id="ambient",
    )
    assert check_report(result.report, result.evidence, request, scope) == []
    assert check_domain(result.report, result.evidence, request, scope) == []
    return result


@pytest.mark.parametrize("kind", ["F1", "F2", "F3", "F4", "F6"])
@pytest.mark.parametrize("mode", ["structural", "domain"])
def test_five_signatures_are_classified_from_real_mcp_evidence(tmp_path, kind, mode):
    store, request, _, _, _ = context(tmp_path, kind)
    result = run(store, request, mode=mode)
    assert result.report.status == "fault"
    assert [(f.fault_type, f.device_id) for f in result.report.findings] == [(kind, "inv01")]
    assert result.report.findings[0].numbers
    if kind == "F1":
        assert "不是精确" in result.report.findings[0].rationale


def rewrite(tmp_path, transform):
    _, request, _, _, _ = context(tmp_path, None)
    path = tmp_path / "observations.parquet"
    frame = pd.read_parquet(path)
    transform(frame).to_parquet(path, index=False)
    return ObservationStore(path, tmp_path / "metadata.json"), request


def test_only_power_gap_does_not_prove_whole_device_communication_loss(tmp_path):
    def transform(frame):
        frame.loc[
            (frame.device_id == "inv01")
            & (frame.metric == "ac_power_kw")
            & frame.ts.between("2019-01-04T03:00Z", "2019-01-04T09:00Z", inclusive="left"),
            "value",
        ] = float("nan")
        return frame

    result = run(*rewrite(tmp_path, transform))
    assert result.report.status == "inconclusive" and not result.report.findings


def test_whole_site_gap_is_not_attributed_to_one_device(tmp_path):
    def transform(frame):
        frame.loc[
            frame.device_id.isin(["inv01", "inv02", "inv03"])
            & frame.ts.between("2019-01-04T03:00Z", "2019-01-04T09:00Z", inclusive="left"),
            "value",
        ] = float("nan")
        return frame

    assert run(*rewrite(tmp_path, transform)).report.status == "inconclusive"


def test_night_gap_works_without_poa_or_daylight_reference(tmp_path):
    def transform(frame):
        frame.loc[
            (frame.device_id == "inv01")
            & frame.ts.between("2019-01-04T03:00Z", "2019-01-04T09:00Z", inclusive="left"),
            "value",
        ] = float("nan")
        return frame[frame.device_id != "poa"]

    result = run(*rewrite(tmp_path, transform))
    assert result.report.status == "fault" and result.loops == 2
    finding = result.report.findings[0]
    assert finding.fault_type == "F6"
    assert finding.window.start.value == pd.Timestamp("2019-01-04T03:00Z")
    assert finding.window.end.value == pd.Timestamp("2019-01-04T09:00Z")
    assert not any(t.name == "peer_compare" for t in result.trace)


def test_gap_window_is_intersection_of_all_visible_metric_gaps(tmp_path):
    def transform(frame):
        target = frame.device_id == "inv01"
        frame.loc[
            target
            & (frame.metric == "ac_power_kw")
            & frame.ts.between("2019-01-04T03:00Z", "2019-01-04T09:00Z", inclusive="left"),
            "value",
        ] = float("nan")
        frame.loc[
            target
            & (frame.metric == "module_c")
            & frame.ts.between("2019-01-04T05:00Z", "2019-01-04T07:00Z", inclusive="left"),
            "value",
        ] = float("nan")
        return frame

    result = run(*rewrite(tmp_path, transform))
    assert len(result.report.findings) == 1
    finding = result.report.findings[0]
    assert finding.fault_type == "F6"
    assert finding.window.start.value == pd.Timestamp("2019-01-04T05:00Z")
    assert finding.window.end.value == pd.Timestamp("2019-01-04T07:00Z")


def test_incomplete_missing_spans_are_not_treated_as_complete_gap_proof(tmp_path):
    def transform(frame):
        missing = (frame.ts >= "2019-01-04T03:00Z") & (frame.ts < "2019-01-04T09:00Z")
        missing |= (frame.ts >= "2019-01-05") & (frame.ts.dt.hour >= 3) & (frame.ts.dt.hour < 5)
        frame.loc[(frame.device_id == "inv01") & missing, "value"] = float("nan")
        return frame

    store, request = rewrite(tmp_path, transform)
    quality = DataTools(store).data_quality(
        QualityRequest(**request.model_dump(exclude={"question"}))
    )
    row = quality.result.series[0]
    assert sum(s.bins for s in row.missing_spans) < row.missing_bins
    result = run(store, request)
    assert not any(f.fault_type == "F6" for f in result.report.findings)


def test_slow_decline_without_recovery_is_not_relabelled_as_string_failure(tmp_path):
    def transform(frame):
        return inject_fault(
            frame,
            2107,
            FaultSpec("F1", "inv01", "2019-01-04", "2019-02-12", soiling_rate_per_day=0.005),
        )[0]

    assert run(*rewrite(tmp_path, transform)).report.status == "inconclusive"


def test_small_soiling_with_recovery_does_not_require_a_deep_loss_span(tmp_path):
    def transform(frame):
        return inject_fault(
            frame,
            2107,
            FaultSpec(
                "F1",
                "inv01",
                "2019-01-04",
                "2019-02-05",
                soiling_rate_per_day=0.0015,
                cleaning_times=("2019-01-28",),
            ),
        )[0]

    result = run(*rewrite(tmp_path, transform))
    assert [(f.fault_type, f.device_id) for f in result.report.findings] == [("F1", "inv01")]
    assert all(
        not row["low_spans"]
        for item in result.evidence.values()
        if item["tool"] == "peer_compare"
        for row in item["result"]["devices"]
    )


def test_static_step_with_repair_is_not_soiling_without_preceding_decline(tmp_path):
    def transform(frame):
        target = (frame.device_id == "inv01") & (frame.metric == "ac_power_kw")
        frame.loc[
            target & frame.ts.between("2019-01-04", "2019-01-28", inclusive="left"), "value"
        ] *= 0.75
        return frame

    result = run(*rewrite(tmp_path, transform))
    assert not any(f.fault_type == "F1" for f in result.report.findings)


@pytest.mark.parametrize("kind", [None, "F5"])
def test_stable_peers_or_common_physical_deviation_do_not_prove_normal_or_drift(tmp_path, kind):
    store, request, _, _, _ = context(tmp_path, kind, parameters=True)
    assert run(store, request).report.status == "inconclusive"


def test_common_soiling_of_all_inverters_is_not_misidentified_as_sensor_drift(tmp_path):
    def transform(frame):
        for device in ["inv01", "inv02", "inv03"]:
            frame = inject_fault(
                frame,
                2107,
                FaultSpec("F1", device, "2019-01-04", "2019-02-12", soiling_rate_per_day=0.003),
            )[0]
        return frame

    assert run(*rewrite(tmp_path, transform)).report.status == "inconclusive"


def test_weak_outage_without_enough_daylight_support_is_not_a_structural_only_fault(tmp_path):
    def transform(frame):
        frame.loc[
            (frame.device_id == "inv01")
            & (frame.metric == "ac_power_kw")
            & frame.ts.between("2019-01-04T16:00Z", "2019-01-04T16:30Z", inclusive="left"),
            "value",
        ] = 0.0
        return frame

    assert run(*rewrite(tmp_path, transform), mode="structural").report.status == "inconclusive"


def test_separate_device_outage_and_night_gap_keep_both_cited_findings(tmp_path):
    def transform(frame):
        faults = [
            FaultSpec("F6", "inv01", "2019-01-04T03:00Z", "2019-01-04T09:00Z"),
            FaultSpec("F3", "inv02", "2019-01-05T16:00Z", "2019-01-05T18:00Z", poa_device_id="poa"),
        ]
        for fault in faults:
            frame = inject_fault(frame, 2107, fault)[0]
        return frame

    result = run(*rewrite(tmp_path, transform))
    assert {(f.fault_type, f.device_id) for f in result.report.findings} == {
        ("F6", "inv01"),
        ("F3", "inv02"),
    }
