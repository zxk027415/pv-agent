"""B0 reference implementation: evidence-only deterministic F3/F4 rules, no LLM."""

from __future__ import annotations

from datetime import timedelta

from pvdiag.agent.state import DiagnosisReport, Finding, Plan, ToolCall
from pvdiag.data.normalize import utc_bound


def latest(evidence: dict, name: str):
    return next((item for item in reversed(list(evidence.values())) if item["tool"] == name), None)


def reference(state):
    observed = latest(state["evidence"], "get_timeseries")
    if not observed:
        return None
    points = observed["result"]["series"][0]["points"]
    high = [p for p in points if p["value"] is not None and p["value"] > 200]
    if len(high) < 4:
        return None
    return {
        "reference_start": state["request"].start.isoformat(),
        "reference_end": (utc_bound(high[3]["ts"]) + timedelta(minutes=15)).isoformat(),
    }


def candidates(evidence):
    """Derive candidate envelopes from observed segments, never injected truth."""
    result = []
    anomalies = latest(evidence, "detect_anomalies")
    peers = latest(evidence, "peer_compare")
    devices_with_zeros = set()
    if anomalies:
        devices = sorted({a["device_id"] for a in anomalies["result"]["anomalies"]})
        for device in devices:
            events = [
                (i, a["span"])
                for i, a in enumerate(anomalies["result"]["anomalies"])
                if a["device_id"] == device and a["kind"] == "zero_power" and a["span"]["bins"] >= 2
            ]
            if events:
                first, last = events[0], events[-1]
                result.append(
                    {
                        "type": "F3",
                        "device_id": device,
                        "start": first[1]["start"],
                        "end": last[1]["end"],
                        "evidence": anomalies,
                        "start_path": f"/result/anomalies/{first[0]}/span/start",
                        "end_path": f"/result/anomalies/{last[0]}/span/end",
                    }
                )
                devices_with_zeros.add(device)
    if peers:
        for i, device in enumerate(peers["result"]["devices"]):
            if device["device_id"] in devices_with_zeros:
                continue
            spans = [(j, span) for j, span in enumerate(device["low_spans"]) if span["bins"] >= 4]
            if spans:
                first, last = spans[0], spans[-1]
                result.append(
                    {
                        "type": "F4",
                        "device_id": device["device_id"],
                        "start": first[1]["start"],
                        "end": last[1]["end"],
                        "evidence": peers,
                        "start_path": f"/result/devices/{i}/low_spans/{first[0]}/start",
                        "end_path": f"/result/devices/{i}/low_spans/{last[0]}/end",
                    }
                )
    return result[:3]


class RuleReasoner:
    def __init__(self):
        self.metadata = {"mode": "B0", "model": "rule-v1", "provider": "none", "temperature": None}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}

    async def plan(self, state, tools, prompt):
        request, scope = state["request"], state["scope"]
        base = {
            "system_id": request.system_id,
            "device_ids": scope.device_ids,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
        }
        if state["loops"] == 0:
            calls = [ToolCall(name="data_quality", arguments=base)]
            if scope.poa_device_id:
                calls.append(
                    ToolCall(
                        name="get_timeseries",
                        arguments=base
                        | {
                            "device_ids": [scope.poa_device_id],
                            "metrics": ["poa_wm2"],
                            "resample": "15min",
                            "end": min(request.end, request.start + timedelta(days=3)).isoformat(),
                            "max_points": 240,
                        },
                    )
                )
            return Plan(calls=calls, ready_for_verdict=False)
        ref = reference(state)
        if ref is None or len(scope.device_ids) < 3:
            return Plan(calls=[])
        common = base | ref | {"poa_device_id": scope.poa_device_id}
        if latest(state["evidence"], "peer_compare") is None:
            return Plan(
                calls=[
                    ToolCall(name=name, arguments=common)
                    for name in ("peer_compare", "detect_anomalies")
                ],
                ready_for_verdict=False,
            )
        calls = []
        for candidate in candidates(state["evidence"]):
            arguments = common | {"start": candidate["start"], "end": candidate["end"]}
            calls.append(ToolCall(name="profile_pattern", arguments=arguments))
            if utc_bound(candidate["start"]) >= utc_bound(ref["reference_end"]):
                calls.append(ToolCall(name="compare_periods", arguments=arguments))
        return Plan(calls=calls)

    async def hypothesize(self, state, prompt):
        findings = []
        for candidate in candidates(state["evidence"]):
            profile = next(
                (
                    e
                    for e in reversed(list(state["evidence"].values()))
                    if e["tool"] == "profile_pattern"
                    and utc_bound(e["request"]["start"]) == utc_bound(candidate["start"])
                    and utc_bound(e["request"]["end"]) == utc_bound(candidate["end"])
                ),
                None,
            )
            if profile is None:
                continue
            index = next(
                i
                for i, p in enumerate(profile["result"]["devices"])
                if p["device_id"] == candidate["device_id"]
            )
            pattern = profile["result"]["devices"][index]
            wanted = "zero_output" if candidate["type"] == "F3" else "level_shift"
            if pattern["shape"] != wanted:
                continue
            observed = candidate["evidence"]
            field = "zero_fraction" if wanted == "zero_output" else "median_peer_ratio"
            findings.append(
                Finding(
                    fault_type=candidate["type"],
                    device_id=candidate["device_id"],
                    confidence="medium",
                    window={
                        "start": {
                            "value": candidate["start"],
                            "evidence_id": observed["evidence_id"],
                            "path": candidate["start_path"],
                        },
                        "end": {
                            "value": candidate["end"],
                            "evidence_id": observed["evidence_id"],
                            "path": candidate["end_path"],
                        },
                    },
                    rationale="高辐照下出现零输出，同站对照与形态支持停机假设。"
                    if wanted == "zero_output"
                    else "相对同站设备出现持续稳定的阶跃损失，支持组串失效假设。",
                    evidence_ids=[observed["evidence_id"], profile["evidence_id"]],
                    numbers=[
                        {
                            "label": "日间零值占比" if wanted == "zero_output" else "相对性能比",
                            "value": pattern[field],
                            "evidence_id": profile["evidence_id"],
                            "path": f"/result/devices/{index}/{field}",
                        }
                    ],
                )
            )
        if findings:
            citations = list(dict.fromkeys(key for f in findings for key in f.evidence_ids))
            return DiagnosisReport(
                status="fault",
                summary="发现有证据支持的设备故障假设。",
                findings=findings,
                evidence_ids=citations,
            )
        return DiagnosisReport(
            status="inconclusive",
            summary="当前规则证据不足，无法确定故障；不能据此认定正常。",
            evidence_ids=list(state["evidence"])[:30],
        )
