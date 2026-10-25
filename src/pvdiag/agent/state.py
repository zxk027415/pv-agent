"""Structured diagnosis state and citation-bearing reports."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

from pydantic import Field, field_validator, model_validator

from pvdiag.mcp_servers.models import Model, WindowRequest

FaultType = Literal["F1", "F2", "F3", "F4", "F5", "F6"]
ToolName = Literal[
    "list_devices",
    "get_site_info",
    "get_timeseries",
    "data_quality",
    "peer_compare",
    "detect_anomalies",
    "profile_pattern",
    "compare_periods",
    "expected_power",
]


class DiagRequest(WindowRequest):
    question: str = Field(
        default="检查光伏逆变器的故障、时间范围及证据。", min_length=1, max_length=2000
    )


class ToolCall(Model):
    name: ToolName
    arguments: dict


class Plan(Model):
    calls: list[ToolCall] = Field(max_length=8)
    ready_for_verdict: bool = True


class NumericClaim(Model):
    label: str
    value: float
    evidence_id: str
    path: str = Field(pattern=r"^/")


class TimeBound(Model):
    value: datetime
    evidence_id: str
    path: str = Field(pattern=r"^/")


class FaultWindow(Model):
    start: TimeBound
    end: TimeBound

    @model_validator(mode="after")
    def positive_window(self):
        if self.start.value.tzinfo is None or self.end.value.tzinfo is None:
            raise ValueError("fault timestamps must be timezone aware")
        if self.start.value >= self.end.value:
            raise ValueError("fault window must increase")
        return self


class Finding(Model):
    fault_type: FaultType
    device_id: str
    window: FaultWindow
    confidence: Literal["low", "medium", "high"]
    rationale: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=2, max_length=12)
    numbers: list[NumericClaim] = Field(default_factory=list, max_length=12)
    alternatives: list[FaultType] = Field(default_factory=list, max_length=2)

    @field_validator("evidence_ids")
    @classmethod
    def distinct_citations(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("duplicate citations")
        return value


class DiagnosisReport(Model):
    status: Literal["fault", "normal", "inconclusive", "needs_review"]
    summary: str = Field(min_length=1, max_length=2000)
    findings: list[Finding] = Field(default_factory=list, max_length=10)
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "fault") != bool(self.findings):
            raise ValueError("only fault reports may contain findings, and require at least one")
        return self


class ToolTrace(Model):
    name: str
    node: str
    success: bool
    duration_seconds: float
    evidence_id: str | None = None
    error: str | None = None


class Scope(Model):
    source_sha256: str
    device_ids: list[str]
    poa_device_id: str | None
    ambient_device_id: str | None


class VerificationIssue(Model):
    code: str = Field(pattern=r"^[a-z_]+$")
    message: str
    finding_index: int | None = Field(default=None, ge=0)
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)
    suggested_tools: list[ToolName] = Field(default_factory=list, max_length=3)


class ModelReview(Model):
    passed: bool
    issues: list[VerificationIssue] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def consistent_review(self):
        if self.passed == bool(self.issues):
            raise ValueError("a passing review has no issues; a failed review requires issues")
        return self


class VerificationRecord(Model):
    loop: int
    mode: Literal["structural", "domain", "domain_llm"]
    version: str = "m2-v1"
    passed: bool
    issues: list[VerificationIssue]
    model_reviewed: bool = False
    checked_evidence_ids: list[str]
    report_status: Literal["fault", "normal", "inconclusive", "needs_review"]
    ready_for_verdict: bool


class RunResult(Model):
    request: DiagRequest
    report: DiagnosisReport
    evidence: dict[str, dict]
    trace: list[ToolTrace]
    loops: int
    model: dict
    model_usage: dict
    harness_sha256: str
    duration_seconds: float
    verification_mode: Literal["structural", "domain", "domain_llm"] = "structural"
    verification: list[VerificationRecord] = Field(default_factory=list)
    verification_policy_sha256: str | None = None


class DiagState(TypedDict, total=False):
    request: DiagRequest
    scope: Scope
    plan: Plan
    evidence: dict[str, dict]
    trace: list[ToolTrace]
    critique: list[str]
    loops: int
    verdict: DiagnosisReport
    result: RunResult
    started_at: float
    verification: list[VerificationRecord]
