"""Structural evidence checks; semantic faithfulness remains a separate evaluation."""

from __future__ import annotations

import math
import re

from pvdiag.agent.state import DiagnosisReport, DiagRequest, Scope
from pvdiag.data.normalize import utc_bound
from pvdiag.knowledge.models import KNOWLEDGE_TOOLS


def resolve(document: dict, path: str):
    """Resolve an RFC 6901 pointer, including escaped object keys."""
    if not path.startswith("/"):
        raise ValueError("citation must be a JSON pointer")
    current = document
    for token in path[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("invalid JSON pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("invalid array index")
            current = current[int(token)]
        else:
            current = current[token]
    return current


def has_uncited_digits(text: str, scope: Scope) -> bool:
    """Digits outside the exact in-scope device/sensor identifiers; dates stay forbidden."""
    identifiers = [
        value
        for value in (*scope.device_ids, scope.poa_device_id, scope.ambient_device_id)
        if value
    ]
    for identifier in sorted(identifiers, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])", "", text)
    return bool(re.search(r"\d", text))


def check_report(
    report: DiagnosisReport, evidence: dict, request: DiagRequest, scope: Scope
) -> list[str]:
    issues = []
    referenced = set(report.evidence_ids)
    if referenced - evidence.keys():
        issues.append("unknown_report_evidence")
    if has_uncited_digits(report.summary, scope):
        issues.append("uncited_number_in_summary")
    if report.status == "normal":
        measurements = {evidence[key]["tool"] for key in referenced if key in evidence}
        if len(measurements - {"list_devices", "get_site_info"} - KNOWLEDGE_TOOLS) < 2:
            issues.append("normal_verdict_requires_measurements")
    for finding in report.findings:
        citations = set(finding.evidence_ids)
        if citations - evidence.keys():
            issues.append("unknown_finding_evidence")
        if not citations <= referenced:
            issues.append("finding_missing_from_report_citations")
        tools = {evidence[key]["tool"] for key in citations if key in evidence}
        if len(tools - {"list_devices", "get_site_info"} - KNOWLEDGE_TOOLS) < 2:
            issues.append("insufficient_distinct_tools")
        allowed = {scope.poa_device_id} if finding.fault_type == "F5" else set(scope.device_ids)
        if finding.device_id not in allowed:
            issues.append("finding_device_outside_scope")
        if finding.window.start.value < request.start or finding.window.end.value > request.end:
            issues.append("finding_window_outside_scope")
        if has_uncited_digits(finding.rationale, scope):
            issues.append("uncited_number_in_rationale")
        if any(number_problem(c, citations, evidence, scope) for c in finding.numbers):
            issues.append("invalid_numeric_citation")
        if any(
            time_problem(b, citations, evidence)
            for b in (finding.window.start, finding.window.end)
        ):
            issues.append("invalid_time_citation")
    return sorted(set(issues))


def pointer_hint(document, path: str) -> str:
    """Where an RFC 6901 pointer stops matching, phrased for a model to repair it."""
    current, walked = document, ""
    for token in path[1:].split("/") if path.startswith("/") else []:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token) or int(token) >= len(current):
                return f"{walked or '/'} 只有 {len(current)} 个元素（下标 0 至 {len(current) - 1}）"
            current = current[int(token)]
        elif isinstance(current, dict):
            if token not in current:
                keys = "、".join(sorted(map(str, current))[:12])
                return f"{walked or '/'} 下没有键 {token}，可用键：{keys}"
            current = current[token]
        else:
            return f"{walked} 已是标量，不能继续向下取 {token}"
        walked += "/" + token
    return "路径不是合法的 JSON Pointer"


def number_problem(claim, citations, evidence: dict, scope: Scope) -> str | None:
    if claim.evidence_id not in evidence:
        return "evidence_id 不存在"
    try:
        actual = resolve(evidence[claim.evidence_id], claim.path)
    except (KeyError, IndexError, TypeError, ValueError):
        return "路径不存在：" + pointer_hint(evidence[claim.evidence_id], claim.path)
    if claim.evidence_id not in citations:
        return "该证据未列入本条结论的 evidence_ids"
    if evidence[claim.evidence_id]["tool"] in KNOWLEDGE_TOOLS:
        return "知识库内容不能作为测量数值"
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return "路径指向的不是数值"
    if not math.isclose(actual, claim.value, rel_tol=1e-9, abs_tol=1e-9):
        return f"数值与证据不一致，证据中的值为 {actual!r}"
    if has_uncited_digits(claim.label, scope):
        return "label 含有范围内设备编号以外的数字字符"
    return None


def time_problem(bound, citations, evidence: dict) -> str | None:
    if bound.evidence_id not in evidence:
        return "evidence_id 不存在"
    try:
        actual = resolve(evidence[bound.evidence_id], bound.path)
    except (KeyError, IndexError, TypeError, ValueError):
        return "路径不存在：" + pointer_hint(evidence[bound.evidence_id], bound.path)
    if bound.evidence_id not in citations:
        return "该证据未列入本条结论的 evidence_ids"
    if not isinstance(actual, str):
        return "路径指向的不是时间戳"
    if evidence[bound.evidence_id]["tool"] in KNOWLEDGE_TOOLS:
        return "知识库文本不能作为观测故障时间"
    try:
        if utc_bound(actual) != utc_bound(bound.value):
            return f"时间与证据不一致，证据中的值为 {actual}"
    except (TypeError, ValueError):
        return "路径指向的文本不是可解析的时间戳"
    return None


def citation_details(report: DiagnosisReport, evidence: dict, scope: Scope) -> list[dict]:
    """Per-entry reasons behind invalid numeric/time citations, for repair feedback only."""
    details = []
    for index, finding in enumerate(report.findings):
        citations = set(finding.evidence_ids)
        for position, claim in enumerate(finding.numbers):
            if reason := number_problem(claim, citations, evidence, scope):
                details.append(
                    {
                        "code": "invalid_numeric_citation",
                        "finding_index": index,
                        "evidence_id": claim.evidence_id,
                        "text": f"numbers[{position}]（{claim.label}，{claim.path}）：{reason}",
                    }
                )
        for name, bound in (("start", finding.window.start), ("end", finding.window.end)):
            if reason := time_problem(bound, citations, evidence):
                details.append(
                    {
                        "code": "invalid_time_citation",
                        "finding_index": index,
                        "evidence_id": bound.evidence_id,
                        "text": f"window.{name}（{bound.path}）：{reason}",
                    }
                )
    return details


def degrade(evidence: dict) -> DiagnosisReport:
    return DiagnosisReport(
        status="needs_review",
        summary="证据或输出校验未通过，需要人工复核。",
        evidence_ids=list(evidence)[:30],
    )
