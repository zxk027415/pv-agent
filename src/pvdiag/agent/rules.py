"""B0 rules: cited outage, step, shading, soiling-recovery and acquisition gaps.

All decisions use bounded observation evidence. Drift and normal operation remain
unresolved without independent physical support; no benchmark truth is accessed.
"""

from __future__ import annotations

from datetime import timedelta

from pvdiag.agent.state import DiagnosisReport, Finding, Plan, ToolCall
from pvdiag.data.normalize import utc_bound


def latest(evidence: dict, name: str):
    return next((item for item in reversed(list(evidence.values())) if item["tool"] == name), None)


def reference(state):
    series = next(
        (
            series
            for item in state["evidence"].values()
            if item["tool"] == "get_timeseries"
            for series in item["result"]["series"]
            if series["metric"] == "poa_wm2"
        ),
        None,
    )
    if series is None:
        return None
    points = series["points"]
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


def _soiling(pattern):
    return (
        pattern["zero_fraction"] == 0
        and pattern["largest_daily_recovery"] is not None
        and pattern["largest_daily_recovery"] >= 0.02
        and pattern["pre_recovery_slope_per_day"] is not None
        and pattern["pre_recovery_slope_per_day"] <= -0.001
    )


def _physical_candidates(evidence):
    selected = candidates(evidence)
    peers = latest(evidence, "peer_compare")
    if peers is None:
        return selected
    analysis = next(
        (
            item
            for item in evidence.values()
            if item["tool"] == "profile_pattern"
            and item["request"]["start"] == peers["request"]["reference_end"]
            and item["request"]["end"] == peers["request"]["end"]
        ),
        None,
    )
    if analysis:
        present = {c["device_id"] for c in selected}
        for candidate in selected:
            row = next(
                (
                    p
                    for p in analysis["result"]["devices"]
                    if p["device_id"] == candidate["device_id"]
                ),
                None,
            )
            if candidate["type"] == "F4" and row and _soiling(row):
                # A slow-loss envelope must include the observed recovery, not
                # stop at the last pre-cleaning low span. This is an assessed
                # interval, not a claim that injection onset/end are known.
                candidate["end"] = analysis["request"]["end"]
                candidate["end_evidence"] = analysis
                candidate["end_path"] = "/request/end"
        # Small but sustained losses can recover before ever crossing the
        # generic low-span threshold. The recovery/trend evidence itself may
        # nominate an assessed interval; it still needs focused validation.
        for row in analysis["result"]["devices"]:
            if row["device_id"] not in present and row["anomalous_days"] == 0 and _soiling(row):
                selected.append(
                    {
                        "type": "F1",
                        "device_id": row["device_id"],
                        "start": analysis["request"]["start"],
                        "end": analysis["request"]["end"],
                        "evidence": analysis,
                        "support_evidence": peers,
                        "start_path": "/request/start",
                        "end_path": "/request/end",
                    }
                )
    return selected[:3]


def _catalog(evidence, request):
    item = latest(evidence, "list_devices")
    return (
        {
            row["device_id"]: row
            for row in item["result"]["devices"]
            if row["device_id"] in request.device_ids
        }
        if item
        else {}
    )


def _quality(evidence, device, metric):
    result = []
    for item in evidence.values():
        if item["tool"] == "data_quality":
            for index, row in enumerate(item["result"]["series"]):
                # Truncated zero/jump lists do not invalidate a complete
                # missing list. Incomplete missing lists are not used.
                if (
                    row["device_id"] == device
                    and row["metric"] == metric
                    and sum(s["bins"] for s in row["missing_spans"]) == row["missing_bins"]
                ):
                    result.append((item, index, row))
    return result


def _gap_candidates(evidence, request):
    result = []
    catalog = _catalog(evidence, request)
    for device in request.device_ids:
        for item, index, row in _quality(evidence, device, "ac_power_kw"):
            intervals = []
            for j, span in enumerate(row["missing_spans"]):
                intervals.append(
                    {
                        "start": span["start"],
                        "end": span["end"],
                        "evidence": item,
                        "start_evidence": item,
                        "end_evidence": item,
                        "start_path": f"/result/series/{index}/missing_spans/{j}/start",
                        "end_path": f"/result/series/{index}/missing_spans/{j}/end",
                        "quality_ids": [item["evidence_id"]],
                    }
                )
            for metric in catalog.get(device, {}).get("metrics", []):
                if metric == "ac_power_kw":
                    continue
                intersections = []
                for other, other_index, other_row in _quality(evidence, device, metric):
                    for j, span in enumerate(other_row["missing_spans"]):
                        for current in intervals:
                            left = max(utc_bound(current["start"]), utc_bound(span["start"]))
                            right = min(utc_bound(current["end"]), utc_bound(span["end"]))
                            if left >= right:
                                continue
                            found = dict(current)
                            if left == utc_bound(span["start"]):
                                found.update(
                                    start=span["start"],
                                    start_evidence=other,
                                    start_path=f"/result/series/{other_index}/missing_spans/{j}/start",
                                )
                            if right == utc_bound(span["end"]):
                                found.update(
                                    end=span["end"],
                                    end_evidence=other,
                                    end_path=f"/result/series/{other_index}/missing_spans/{j}/end",
                                )
                            found["quality_ids"] = list(
                                dict.fromkeys([*current["quality_ids"], other["evidence_id"]])
                            )
                            intersections.append(found)
                intervals = intersections
            for found in intervals:
                if utc_bound(found["end"]) - utc_bound(found["start"]) >= timedelta(hours=1):
                    found.update(type="F6", device_id=device)
                    result.append(found)
    unique = {}
    for candidate in result:
        unique[(candidate["device_id"], candidate["start"], candidate["end"])] = candidate
    return sorted(unique.values(), key=lambda c: (c["start"], c["device_id"]))[:3]


def _peer_online(evidence, device, start, end):
    for item, _, row in _quality(evidence, device, "ac_power_kw"):
        if (
            utc_bound(item["request"]["start"]) <= start
            and utc_bound(item["request"]["end"]) >= end
            and row["negative_bins"] == 0
            and not any(
                utc_bound(s["start"]) < end and utc_bound(s["end"]) > start
                for s in row["missing_spans"]
            )
        ):
            return True
    return False


def _finding(candidate, kind, rationale, evidence_ids, numbers, confidence="medium"):
    start_item = candidate.get("start_evidence", candidate["evidence"])
    end_item = candidate.get("end_evidence", candidate["evidence"])
    return Finding(
        fault_type=kind,
        device_id=candidate["device_id"],
        confidence=confidence,
        window={
            "start": {
                "value": candidate["start"],
                "evidence_id": start_item["evidence_id"],
                "path": candidate["start_path"],
            },
            "end": {
                "value": candidate["end"],
                "evidence_id": end_item["evidence_id"],
                "path": candidate["end_path"],
            },
        },
        rationale=rationale,
        evidence_ids=list(
            dict.fromkeys([*evidence_ids, start_item["evidence_id"], end_item["evidence_id"]])
        ),
        numbers=numbers,
    )


def _gap_findings(state):
    evidence, request = state["evidence"], state["request"]
    findings = []
    for candidate in _gap_candidates(evidence, request):
        start, end = utc_bound(candidate["start"]), utc_bound(candidate["end"])
        peers = [d for d in request.device_ids if d != candidate["device_id"]]
        if sum(_peer_online(evidence, d, start, end) for d in peers) < 2:
            continue
        for item in evidence.values():
            if item["tool"] != "get_timeseries" or (
                utc_bound(item["request"]["start"]) != start
                or utc_bound(item["request"]["end"]) != end
            ):
                continue
            for index, row in enumerate(item["result"]["series"]):
                if (
                    row["device_id"] != candidate["device_id"]
                    or row["metric"] != "ac_power_kw"
                    or row["observed_bins"] != 0
                ):
                    continue
                findings.append(
                    _finding(
                        candidate,
                        "F6",
                        "全部可见测点同期缺失，同站对照仍有记录，支持采集或通讯异常候选。",
                        [*candidate["quality_ids"], item["evidence_id"]],
                        [
                            {
                                "label": "故障段功率覆盖率",
                                "value": row["coverage"],
                                "evidence_id": item["evidence_id"],
                                "path": f"/result/series/{index}/coverage",
                            }
                        ],
                        confidence="low",
                    )
                )
                break
            else:
                continue
            break
    return findings


def _pattern_kind(candidate, pattern):
    if candidate["type"] == "F3":
        return (
            "F3"
            if pattern["zero_fraction"] is not None
            and pattern["zero_fraction"] >= 0.95
            and pattern["median_peer_ratio"] is not None
            and pattern["median_peer_ratio"] <= 0.05
            else None
        )
    if _soiling(pattern):
        return "F1"
    if (
        pattern["anomalous_days"] >= 3
        and 0 < pattern["loss_hour_count"] <= 3
        and pattern["zero_fraction"] is not None
        and pattern["zero_fraction"] <= 0.05
        and pattern["off_loss_median_ratio"] is not None
        and 0.9 <= pattern["off_loss_median_ratio"] <= 1.15
    ):
        return "F2"
    gradual = (
        utc_bound(candidate["end"]) - utc_bound(candidate["start"]) >= timedelta(days=7)
        and pattern["slope_per_day"] is not None
        and pattern["slope_per_day"] <= -0.001
    )
    if (
        pattern["shape"] == "level_shift"
        and pattern["zero_fraction"] == 0
        and pattern["median_peer_ratio"] is not None
        and 0.05 < pattern["median_peer_ratio"] < 0.9
        and pattern["loss_hour_count"] > 3
        and not gradual
    ):
        return "F4"
    return None


class RuleReasoner:
    def __init__(self):
        self.metadata = {"mode": "B0", "model": "rule-v2", "provider": "none", "temperature": None}
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
            catalog = _catalog(state["evidence"], request)
            for metric in sorted(
                {metric for row in catalog.values() for metric in row["metrics"]} - {"ac_power_kw"}
            ):
                devices = [
                    d for d in scope.device_ids if metric in catalog.get(d, {}).get("metrics", [])
                ]
                calls.append(
                    ToolCall(
                        name="data_quality",
                        arguments=base | {"device_ids": devices, "metrics": [metric]},
                    )
                )
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
        gaps = _gap_candidates(state["evidence"], request)
        common = base | (ref or {}) | {"poa_device_id": scope.poa_device_id}
        physical = (
            ref is not None
            and len(scope.device_ids) >= 3
            and utc_bound(ref["reference_end"]) < request.end
        )
        if state["loops"] == 1:
            calls = [
                ToolCall(
                    name="get_timeseries",
                    arguments=base
                    | {
                        "device_ids": [c["device_id"]],
                        "start": c["start"],
                        "end": c["end"],
                        "metrics": ["ac_power_kw"],
                        "resample": "15min",
                        "max_points": 48,
                    },
                )
                for c in gaps
            ]
            if physical:
                calls += [
                    ToolCall(name=name, arguments=common)
                    for name in ("peer_compare", "detect_anomalies")
                ]
                calls.append(
                    ToolCall(
                        name="profile_pattern", arguments=common | {"start": ref["reference_end"]}
                    )
                )
            return Plan(calls=calls, ready_for_verdict=not physical)
        calls = []
        if not physical:
            return Plan(calls=[])
        for candidate in _physical_candidates(state["evidence"])[: max(0, 3 - len(gaps))]:
            arguments = common | {"start": candidate["start"], "end": candidate["end"]}
            already_profiled = any(
                item["tool"] == "profile_pattern"
                and utc_bound(item["request"]["start"]) == utc_bound(candidate["start"])
                and utc_bound(item["request"]["end"]) == utc_bound(candidate["end"])
                for item in state["evidence"].values()
            )
            if not already_profiled:
                calls.append(ToolCall(name="profile_pattern", arguments=arguments))
            if utc_bound(candidate["start"]) >= utc_bound(ref["reference_end"]):
                calls.append(ToolCall(name="compare_periods", arguments=arguments))
        return Plan(calls=calls)

    async def hypothesize(self, state, prompt):
        findings = _gap_findings(state)
        for candidate in _physical_candidates(state["evidence"]):
            if len(findings) >= 3:
                break
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
            # B0 also requires basic observation support in structural-only
            # mode. The graph's optional domain gate is not its classifier.
            start = utc_bound(candidate["start"])
            end = utc_bound(candidate["end"])
            supported_peers = sum(
                row["device_id"] != candidate["device_id"]
                and row["observed_daylight_bins"] >= 4
                and row["zero_fraction"] == 0
                and row["median_peer_ratio"] is not None
                and row["median_peer_ratio"] >= 0.5
                for row in profile["result"]["devices"]
            )
            if (
                utc_bound(profile["request"]["reference_end"]) > start
                or pattern["daylight_bins"] < 4
                or pattern["observed_daylight_bins"] != pattern["daylight_bins"]
                or supported_peers < 2
                or not _peer_online(state["evidence"], candidate["device_id"], start, end)
            ):
                continue
            kind = _pattern_kind(candidate, pattern)
            if kind is None:
                continue
            observed = candidate["evidence"]
            fields = {
                "F1": [
                    ("恢复前相对日斜率", "pre_recovery_slope_per_day"),
                    ("最大日尺度恢复阶跃", "largest_daily_recovery"),
                ],
                "F2": [
                    ("重复异常日数", "anomalous_days"),
                    ("异常当地小时数", "loss_hour_count"),
                    ("其他时段相对性能", "off_loss_median_ratio"),
                ],
                "F3": [("日间零值占比", "zero_fraction")],
                "F4": [("相对性能比", "median_peer_ratio")],
            }[kind]
            rationale = {
                "F1": "观测包络内存在多日相对缓降及明显恢复阶跃，支持积灰与恢复候选，起止不是精确机理时刻。",
                "F2": "损失跨日重复集中于少数当地小时，其他时段恢复相对正常，支持固定时段遮挡候选。",
                "F3": "高辐照下出现零输出，同站对照与形态支持停机假设。",
                "F4": "相对同站设备出现持续稳定的阶跃损失，支持组串失效假设。",
            }[kind]
            findings.append(
                _finding(
                    candidate,
                    kind,
                    rationale,
                    [
                        observed["evidence_id"],
                        profile["evidence_id"],
                        candidate.get("support_evidence", observed)["evidence_id"],
                    ],
                    [
                        {
                            "label": label,
                            "value": pattern[field],
                            "evidence_id": profile["evidence_id"],
                            "path": f"/result/devices/{index}/{field}",
                        }
                        for label, field in fields
                    ],
                    confidence="low" if kind == "F1" else "medium",
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
