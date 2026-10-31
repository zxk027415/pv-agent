"""Domain evidence gates, without raw data, benchmark truth or model assertions.

These are conservative necessary checks, not a substitute for held-out scoring
or the independent faithfulness judge. Missing support remains unverified.
"""

from __future__ import annotations

from statistics import median

from pvdiag.agent.state import DiagnosisReport, DiagRequest, Scope, VerificationIssue
from pvdiag.data.normalize import utc_bound


def _bounds(item):
    request = item["request"]
    return utc_bound(request["start"]), utc_bound(request["end"])


def _covers(item, start, end, *, exact=False):
    if "start" not in item.get("request", {}) or "end" not in item["request"]:
        return False
    left, right = _bounds(item)
    return (left == start and right == end) if exact else left <= start and right >= end


def _row(item, device, metric=None):
    for row in item["result"].get("devices", item["result"].get("series", [])):
        if row["device_id"] == device and (metric is None or row.get("metric") == metric):
            return row
    return None


def _span_fraction(spans, start, end):
    intervals = []
    for span in spans:
        left, right = max(start, utc_bound(span["start"])), min(end, utc_bound(span["end"]))
        if left < right:
            intervals.append((left, right))
    total = 0.0
    cursor = start
    for left, right in sorted(intervals):
        left = max(left, cursor)
        if left < right:
            total += (right - left).total_seconds()
            cursor = right
    return total / (end - start).total_seconds()


def _quality_clear(items, device, start, end, metric="ac_power_kw"):
    for item in items:
        row = _row(item, device, metric)
        if row is None or not _covers(item, start, end):
            continue
        if metric.endswith("power_kw") and row["negative_bins"]:
            continue
        if row["missing_bins"] == 0 and row["coverage"] >= 0.99:
            return True
        complete = (
            not item["result"]["spans_truncated"]
            and sum(s["bins"] for s in row["missing_spans"]) == row["missing_bins"]
        )
        if complete and _span_fraction(row["missing_spans"], start, end) <= 0.01:
            return True
    return False


def _physical_rows(item, devices):
    result = {}
    for device in devices:
        row = _row(item, device)
        if not row or row.get("status") != "available" or not row.get("parameters"):
            return None
        value = row.get("median_performance_ratio")
        if value is None or value <= 0 or row.get("comparable_bins", 0) < 4:
            return None
        result[device] = row
    return result


def check_domain(report: DiagnosisReport, evidence: dict, request: DiagRequest, scope: Scope):
    issues = []

    def add(code, message, tools=(), index=None, ids=()):
        issues.append(
            VerificationIssue(
                code=code,
                message=message,
                finding_index=index,
                suggested_tools=list(tools),
                evidence_ids=list(ids)[:30],
            )
        )

    def select(tool, *, start=None, end=None, exact=False, citations=None):
        return [
            item
            for key, item in evidence.items()
            if item["tool"] == tool
            and item["source_sha256"] == scope.source_sha256
            and (citations is None or key in citations)
            and (start is None or _covers(item, start, end, exact=exact))
        ]

    quality = select("data_quality")
    if report.status in {"inconclusive", "needs_review"}:
        return issues
    if report.status == "normal":
        if not all(
            _quality_clear(quality, d, request.start, request.end) for d in scope.device_ids
        ):
            add(
                "normal_quality_unverified",
                "正常结论缺少完整查询窗的逐设备质量检查。",
                ("data_quality",),
            )
        peers = select(
            "peer_compare",
            start=request.start,
            end=request.end,
            exact=True,
            citations=report.evidence_ids,
        )
        healthy = any(
            all(
                (row := _row(item, d))
                and row["reference_bins"] >= 4
                and row["comparable_daylight_bins"] >= 16
                and row["median_ratio"] is not None
                and 0.9 <= row["median_ratio"] <= 1.15
                and row["low_ratio_fraction"] == 0
                and not row["low_spans"]
                for d in scope.device_ids
            )
            for item in peers
        )
        if not healthy:
            add(
                "normal_peers_unverified",
                "未充分核对所有设备的同站对比，不能认定正常。",
                ("peer_compare",),
            )
        detection = select("detect_anomalies", start=request.start, end=request.end, exact=True)
        cleared = any(
            not item["result"]["truncated"]
            and not item["result"]["anomalies"]
            and set(scope.device_ids) <= set(item["request"]["device_ids"])
            for item in detection
        )
        if not cleared:
            add(
                "normal_anomaly_unresolved",
                "缺少完整异常排查，或仍有未解释的零值、缺测和统计异常。",
                ("detect_anomalies",),
            )
        physical = select(
            "expected_power",
            start=request.start,
            end=request.end,
            exact=True,
            citations=report.evidence_ids,
        )
        supported = any(
            (rows := _physical_rows(item, scope.device_ids))
            and all(0.9 <= r["median_performance_ratio"] <= 1.1 for r in rows.values())
            for item in physical
        )
        if not supported:
            add(
                "common_mode_not_excluded",
                "同站比值正常不能排除全站漂移，需独立物理性能证据。",
                ("expected_power",),
            )
        declining = False
        for item in physical:
            rows = _physical_rows(item, scope.device_ids)
            if not rows:
                continue
            trends = []
            for row in rows.values():
                points = sorted(
                    (utc_bound(p["ts"]), p["value"])
                    for p in row["performance_ratio_points"]
                    if p["value"] is not None
                )
                trends.append(
                    len(points) >= 6
                    and (points[-1][0] - points[0][0]).total_seconds() >= 2 * 86400
                    and median(v for _, v in points[-3:]) < 0.99 * median(v for _, v in points[:3])
                )
            declining |= all(trends)
        if declining:
            add(
                "normal_common_trend_unresolved",
                "正常平均性能比不能覆盖多日全站共同下降趋势。",
                ("expected_power",),
            )
        return issues

    for index, finding in enumerate(report.findings):
        start, end = utc_bound(finding.window.start.value), utc_bound(finding.window.end.value)
        citations = set(finding.evidence_ids)
        device, kind = finding.device_id, finding.fault_type
        if kind == "F6":
            metrics = {"ac_power_kw"}
            for catalog in select("list_devices"):
                descriptor = _row(catalog, device)
                if descriptor:
                    metrics.update(descriptor["metrics"])
            absent = all(
                any(
                    (row := _row(item, device, metric))
                    and _covers(item, start, end)
                    and _span_fraction(row["missing_spans"], start, end) >= 0.95
                    for item in quality
                    if item["evidence_id"] in citations
                )
                for metric in metrics
            )
            if not absent:
                add(
                    "communication_gap_unverified",
                    "需逐测点缺测段证明整台设备缺失；零功率不等于通讯中断。",
                    ("data_quality",),
                    index,
                )
            others = [d for d in scope.device_ids if d != device]
            if sum(_quality_clear(quality, d, start, end) for d in others) < 2:
                add(
                    "communication_peers_unverified",
                    "需确认至少两台同期设备仍有有效数据，排除全站采集中断。",
                    ("data_quality",),
                    index,
                )
            continue

        if kind == "F5":
            if not _quality_clear(quality, device, start, end, metric="poa_wm2"):
                add(
                    "sensor_quality_unverified",
                    "需先排除辐照测点缺测，避免把不连续数据解释为漂移。",
                    ("data_quality",),
                    index,
                )
            peers = select("peer_compare", start=start, end=end, citations=citations)
            unchanged = any(
                all(
                    (row := _row(item, d))
                    and row["reference_bins"] >= 4
                    and row["comparable_daylight_bins"] >= 16
                    and row["median_ratio"] is not None
                    and 0.95 <= row["median_ratio"] <= 1.05
                    and row["low_ratio_fraction"] is not None
                    and row["low_ratio_fraction"] <= 0.05
                    for d in scope.device_ids
                )
                for item in peers
            )
            if not unchanged:
                add(
                    "sensor_drift_peers_unverified",
                    "需证明可见逆变器间相对关系稳定，而非单机性能损失。",
                    ("peer_compare",),
                    index,
                )
            current = select(
                "expected_power", start=start, end=end, exact=True, citations=citations
            )
            before = [
                item
                for item in select("expected_power", citations=citations)
                if _bounds(item)[1] <= start
            ]
            common_drop = False
            for after in current:
                after_rows = _physical_rows(after, scope.device_ids)
                if not after_rows:
                    continue
                for previous in before:
                    before_rows = _physical_rows(previous, scope.device_ids)
                    if before_rows and all(
                        before_rows[d]["parameters"] == after_rows[d]["parameters"]
                        and after_rows[d]["median_performance_ratio"]
                        < 0.95 * before_rows[d]["median_performance_ratio"]
                        for d in scope.device_ids
                    ):
                        common_drop = True
            if not common_drop:
                add(
                    "sensor_drift_physics_unverified",
                    "缺少故障前后全站物理性能下降证据；模型 unavailable 不能作为漂移证明。",
                    ("expected_power",),
                    index,
                )
            gradual = False
            for item in current:
                rows = _physical_rows(item, scope.device_ids)
                if not rows:
                    continue
                trajectories = []
                for row in rows.values():
                    points = sorted(
                        (utc_bound(p["ts"]), p["value"])
                        for p in row["performance_ratio_points"]
                        if p["value"] is not None
                    )
                    trajectories.append(
                        len(points) >= 6
                        and (points[-1][0] - points[0][0]).total_seconds() >= 2 * 86400
                        and median(v for _, v in points[-3:])
                        < 0.99 * median(v for _, v in points[:3])
                    )
                gradual |= all(trajectories)
            if not gradual:
                add(
                    "sensor_drift_trend_unverified",
                    "需连续多日共同渐变，单个静态低性能比不足以证明缓慢漂移。",
                    ("expected_power",),
                    index,
                )
            continue

        if not _quality_clear(quality, device, start, end):
            add(
                "power_quality_unverified",
                "功率故障需排除候选时间段内缺测和异常负值。",
                ("data_quality",),
                index,
            )
        profiles = select("profile_pattern", start=start, end=end, exact=True, citations=citations)
        profile = next((item for item in reversed(profiles) if _row(item, device)), None)
        if profile is None:
            add(
                "fault_profile_missing",
                "缺少针对该设备、该故障时间段的形态证据：需调用 profile_pattern，device_ids 含"
                + f" {device} 及至少两台同站对照，start={start:%Y-%m-%dT%H:%M:%SZ}、"
                + f"end={end:%Y-%m-%dT%H:%M:%SZ}（与本条结论时间窗完全一致），reference_end"
                + " 不晚于该 start，并把该证据列入本条 finding 的 evidence_ids。",
                ("profile_pattern",),
                index,
            )
            continue
        target = _row(profile, device)
        enough = (
            target["daylight_bins"] >= 4
            and target["observed_daylight_bins"] == target["daylight_bins"]
        )
        if not enough:
            add(
                "daylight_support_missing",
                "高辐照有效观测不足，无法区分阴雨、缺测与设备故障。",
                ("profile_pattern",),
                index,
            )
        peers = [_row(profile, d) for d in scope.device_ids if d != device]
        comparable = sum(
            bool(
                row
                and row["observed_daylight_bins"] >= 4
                and row["zero_fraction"] == 0
                and row["median_peer_ratio"] is not None
                and row["median_peer_ratio"] >= 0.5
            )
            for row in peers
        )
        if comparable < 2:
            add(
                "healthy_peers_missing",
                "至少需两台同期有效、非零输出的同站对照，排除全站影响。",
                ("profile_pattern",),
                index,
            )
        if (
            profile["request"].get("reference_end") is None
            or utc_bound(profile["request"]["reference_end"]) > start
        ):
            add(
                "reference_not_before_fault",
                "参考期必须早于所声称故障，避免故障后重新定标：所引 profile_pattern 的"
                + f" reference_end 不得晚于 {start:%Y-%m-%dT%H:%M:%SZ}。",
                ("profile_pattern",),
                index,
            )
        if kind == "F3":
            zero = target["zero_fraction"] is not None and target["zero_fraction"] >= 0.95
            ratio = target["median_peer_ratio"] is not None and target["median_peer_ratio"] <= 0.05
            measured = any(
                any(
                    a["device_id"] == device
                    and a["kind"] == "zero_power"
                    and _span_fraction([a["span"]], start, end) > 0
                    for a in item["result"]["anomalies"]
                )
                for item in select("detect_anomalies", start=start, end=end, citations=citations)
            )
            if not (zero and ratio and measured):
                add(
                    "outage_evidence_missing",
                    "停机必须有高辐照零值段及相对同站输出接近零的独立支持。",
                    ("detect_anomalies", "profile_pattern"),
                    index,
                )
        elif kind == "F4":
            stable = (
                target["median_peer_ratio"] is not None
                and 0.05 < target["median_peer_ratio"] < 0.9
                and target["ratio_iqr"] is not None
                and target["ratio_iqr"] < 0.08
                and target["zero_fraction"] == 0
            )
            if not stable:
                add(
                    "string_step_unverified",
                    "组串失效需非零、稳定比例的损失，不能用任意异常替代。",
                    ("profile_pattern", "compare_periods"),
                    index,
                )
            if target["loss_hour_count"] <= 3:
                add(
                    "shading_not_excluded",
                    "损失仅覆盖少数当地小时，尚未排除固定时段遮挡。",
                    ("profile_pattern",),
                    index,
                )
        elif kind == "F2":
            localized = (
                target["anomalous_days"] >= 3
                and 0 < target["loss_hour_count"] <= 3
                and target["zero_fraction"] is not None
                and target["zero_fraction"] <= 0.05
                and target["off_loss_median_ratio"] is not None
                and 0.9 <= target["off_loss_median_ratio"] <= 1.15
            )
            if not localized:
                add(
                    "shading_repeat_unverified",
                    "遮挡需多日重复的窄时段损失、时段外恢复正常，并排除零值停机。",
                    ("profile_pattern",),
                    index,
                )
        elif kind == "F1":
            recovery = (
                target["largest_daily_recovery"] is not None
                and target["largest_daily_recovery"] >= 0.02
                and target["pre_recovery_slope_per_day"] is not None
                and target["pre_recovery_slope_per_day"] <= -0.001
            )
            if not recovery:
                add(
                    "soiling_ramp_recovery_unverified",
                    "积灰需清洗恢复前的多日缓降及恢复阶跃，短窗小幅变化不足。",
                    ("profile_pattern",),
                    index,
                )
    return issues
