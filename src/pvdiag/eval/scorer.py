"""Fixed scorer v1.1: scene detection and exact multiset root/device correctness.

Multi-fault root correctness requires one-to-one matches with no extra findings.
Temporal score uses same-device/type matching and IoU >=0.5, penalizing extras.
Normal abstentions are reported separately and do not count as correct normal
verdicts. LLM-as-Judge faithfulness and monetary price are not invented here.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from pvdiag.agent.state import RunResult
from pvdiag.data.normalize import utc_bound

SCORER_VERSION = "v1.1"
READ_ONLY_TOOLS = frozenset(
    {
        "list_devices",
        "get_site_info",
        "get_timeseries",
        "data_quality",
        "peer_compare",
        "detect_anomalies",
        "profile_pattern",
        "compare_periods",
        "expected_power",
    }
)


def numeric_value(document, path):
    if not path.startswith("/"):
        raise ValueError("invalid numeric citation pointer")
    for token in path[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("invalid pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(document, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("invalid array index")
            document = document[int(token)]
        else:
            document = document[token]
    return document


def temporal_iou(start, end, true_start, true_end) -> float:
    a, b, c, d = map(utc_bound, (start, end, true_start, true_end))
    if a >= b or c >= d:
        raise ValueError("IoU requires increasing intervals")
    intersection = max(0.0, (min(b, d) - max(a, c)).total_seconds())
    union = (b - a).total_seconds() + (d - c).total_seconds() - intersection
    return intersection / union


def maximum_matching(edges: list[list[int]]) -> int:
    matched = {}

    def augment(left, seen):
        for right in edges[left]:
            if right in seen:
                continue
            seen.add(right)
            if right not in matched or augment(matched[right], seen):
                matched[right] = left
                return True
        return False

    return sum(augment(i, set()) for i in range(len(edges)))


def score_case(truth: dict, run: RunResult) -> dict:
    if truth["system_id"] != run.request.system_id:
        raise ValueError("truth and request system differ")
    if tuple(map(utc_bound, truth["query_window"])) != (
        utc_bound(run.request.start),
        utc_bound(run.request.end),
    ):
        raise ValueError("truth and query windows differ")
    faults, predicted = truth["faults"], run.report.findings
    for fault in faults:
        if fault["type"] not in {"F1", "F2", "F3", "F4", "F5", "F6"}:
            raise ValueError("unknown truth fault type")
        poa_devices = {
            d["device_id"]
            for ev in run.evidence.values()
            if ev.get("tool") == "list_devices"
            for d in ev["result"]["devices"]
            if "poa_wm2" in d["metrics"]
        }
        allowed = poa_devices if fault["type"] == "F5" else set(run.request.device_ids)
        if fault["device_id"] not in allowed:
            raise ValueError("truth device outside request")
        start, end = map(utc_bound, (fault["start"], fault["end"]))
        if not run.request.start <= start < end <= run.request.end:
            raise ValueError("truth interval outside request")
    actual_fault, predicted_fault = bool(faults), run.report.status == "fault"
    kinds = [p.fault_type for p in predicted]
    top3_edges = [
        [i for i, p in enumerate(predicted) if f["type"] in {p.fault_type, *p.alternatives}]
        for f in faults
    ]
    temporal_edges = [
        [
            i
            for i, p in enumerate(predicted)
            if p.fault_type == f["type"]
            and p.device_id == f["device_id"]
            and temporal_iou(p.window.start.value, p.window.end.value, f["start"], f["end"]) >= 0.5
        ]
        for f in faults
    ]
    valid_numbers = total_numbers = 0
    for finding in predicted:
        for claim in finding.numbers:
            total_numbers += 1
            try:
                value = numeric_value(run.evidence[claim.evidence_id], claim.path)
                valid_numbers += (
                    claim.evidence_id in finding.evidence_ids
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isclose(value, claim.value, rel_tol=1e-9, abs_tol=1e-9)
                )
            except (KeyError, IndexError, ValueError, TypeError):
                pass
    return {
        "scenario_id": truth["scenario_id"],
        "actual_fault": actual_fault,
        "predicted_fault": predicted_fault,
        "status": run.report.status,
        "root_top1": (Counter(f["type"] for f in faults) == Counter(kinds))
        if actual_fault
        else None,
        "root_top3": (len(predicted) == len(faults) and maximum_matching(top3_edges) == len(faults))
        if actual_fault
        else None,
        "device_correct": (
            Counter(f["device_id"] for f in faults) == Counter(p.device_id for p in predicted)
        )
        if actual_fault
        else None,
        "joint_root_device_correct": (
            Counter((f["type"], f["device_id"]) for f in faults)
            == Counter((p.fault_type, p.device_id) for p in predicted)
        )
        if actual_fault
        else None,
        "temporal_hits": maximum_matching(temporal_edges),
        "temporal_denominator": max(len(faults), len(predicted)) if actual_fault else 0,
        "normal_correct": run.report.status == "normal" if not actual_fault else None,
        "abstained": run.report.status in {"inconclusive", "needs_review"},
        "valid_numeric_claims": int(valid_numbers),
        "numeric_claims": total_numbers,
        "tool_calls": len(run.trace),
        "successful_tool_calls": sum(t.success for t in run.trace),
        "unapproved_write_executions": sum(
            t.success and t.name not in READ_ONLY_TOOLS for t in run.trace
        ),
        "input_tokens": run.model_usage.get("input_tokens", 0),
        "output_tokens": run.model_usage.get("output_tokens", 0),
        "usage_missing_calls": run.model_usage.get("usage_missing_calls", 0),
        "duration_seconds": run.duration_seconds,
    }


def aggregate(cases: list[dict]) -> dict:
    if not cases or len({c["scenario_id"] for c in cases}) != len(cases):
        raise ValueError("scoring requires nonempty unique scenarios")
    positives = [c for c in cases if c["actual_fault"]]
    negatives = [c for c in cases if not c["actual_fault"]]
    tp = sum(c["actual_fault"] and c["predicted_fault"] for c in cases)
    fp = sum(not c["actual_fault"] and c["predicted_fault"] for c in cases)
    fn = sum(c["actual_fault"] and not c["predicted_fault"] for c in cases)
    precision = tp / (tp + fp) if tp + fp else (0.0 if positives else None)
    recall = tp / (tp + fn) if positives else None
    f1 = (
        (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if positives
        else None
    )
    calls = sum(c["tool_calls"] for c in cases)
    claims = sum(c["numeric_claims"] for c in cases)
    temporal = sum(c["temporal_denominator"] for c in positives)
    return {
        "scorer_version": SCORER_VERSION,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scenarios": len(cases),
        "fault_scenarios": len(positives),
        "normal_scenarios": len(negatives),
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "root_top1_accuracy": sum(c["root_top1"] for c in positives) / len(positives)
        if positives
        else None,
        "root_top3_accuracy": sum(c["root_top3"] for c in positives) / len(positives)
        if positives
        else None,
        "device_accuracy": sum(c["device_correct"] for c in positives) / len(positives)
        if positives
        else None,
        "joint_root_device_accuracy": sum(c["joint_root_device_correct"] for c in positives)
        / len(positives)
        if positives
        else None,
        "temporal_iou_hit_rate": sum(c["temporal_hits"] for c in positives) / temporal
        if temporal
        else None,
        "normal_false_alarm_rate": fp / len(negatives) if negatives else None,
        "normal_accuracy": sum(c["normal_correct"] for c in negatives) / len(negatives)
        if negatives
        else None,
        "abstention_rate": sum(c["abstained"] for c in cases) / len(cases),
        "numeric_citation_accuracy": sum(c["valid_numeric_claims"] for c in cases) / claims
        if claims
        else None,
        "evidence_faithfulness_judge": None,
        "tool_success_rate": sum(c["successful_tool_calls"] for c in cases) / calls
        if calls
        else None,
        "mean_tool_calls": calls / len(cases),
        "unapproved_write_executions": sum(c["unapproved_write_executions"] for c in cases),
        "input_tokens": None
        if any(c["usage_missing_calls"] for c in cases)
        else sum(c["input_tokens"] for c in cases),
        "output_tokens": None
        if any(c["usage_missing_calls"] for c in cases)
        else sum(c["output_tokens"] for c in cases),
        "known_input_tokens": sum(c["input_tokens"] for c in cases),
        "known_output_tokens": sum(c["output_tokens"] for c in cases),
        "usage_complete": not any(c["usage_missing_calls"] for c in cases),
        "mean_duration_seconds": sum(c["duration_seconds"] for c in cases) / len(cases),
        "cost_usd": None,
    }
