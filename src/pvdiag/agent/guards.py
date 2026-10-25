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


def check_report(
    report: DiagnosisReport, evidence: dict, request: DiagRequest, scope: Scope
) -> list[str]:
    issues = []
    referenced = set(report.evidence_ids)
    if referenced - evidence.keys():
        issues.append("unknown_report_evidence")
    if re.search(r"\d", report.summary):
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
        if re.search(r"\d", finding.rationale):
            issues.append("uncited_number_in_rationale")
        for claim in finding.numbers:
            try:
                actual = resolve(evidence[claim.evidence_id], claim.path)
                if (
                    claim.evidence_id not in citations
                    or evidence[claim.evidence_id]["tool"] in KNOWLEDGE_TOOLS
                    or isinstance(actual, bool)
                    or not isinstance(actual, (int, float))
                ):
                    raise ValueError("not a cited numeric measurement")
                if not math.isclose(actual, claim.value, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("value mismatch")
                if re.search(r"\d", claim.label):
                    raise ValueError("uncited number in label")
            except (KeyError, IndexError, TypeError, ValueError):
                issues.append("invalid_numeric_citation")
        for bound in (finding.window.start, finding.window.end):
            try:
                actual = resolve(evidence[bound.evidence_id], bound.path)
                if bound.evidence_id not in citations or not isinstance(actual, str):
                    raise ValueError("not a cited timestamp")
                if evidence[bound.evidence_id]["tool"] in KNOWLEDGE_TOOLS:
                    raise ValueError("knowledge text is not an observed fault timestamp")
                if utc_bound(actual) != utc_bound(bound.value):
                    raise ValueError("timestamp mismatch")
            except (KeyError, IndexError, TypeError, ValueError):
                issues.append("invalid_time_citation")
    return sorted(set(issues))


def degrade(evidence: dict) -> DiagnosisReport:
    return DiagnosisReport(
        status="needs_review",
        summary="证据或输出校验未通过，需要人工复核。",
        evidence_ids=list(evidence)[:30],
    )
