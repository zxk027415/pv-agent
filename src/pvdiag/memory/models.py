"""Memory provenance and admission; benchmark truth is never stored in recalled cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from pvdiag.agent.gateway import CONTRACTS
from pvdiag.agent.guards import check_report
from pvdiag.agent.state import DiagnosisReport, DiagRequest, RunResult, Scope
from pvdiag.eval import scorer
from pvdiag.eval.outcome import case_passed
from pvdiag.knowledge.models import KNOWLEDGE_TOOLS
from pvdiag.mcp_servers.models import Evidence, Model
from pvdiag.observability.store import digest


class MemoryCase(Model):
    version: Literal["development-case-v1"] = "development-case-v1"
    split: Literal["dev"] = "dev"
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_id: str = Field(min_length=1, max_length=100)
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    harness_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: DiagRequest
    report: DiagnosisReport
    model: dict
    is_correct: bool
    scorer_version: str
    scorer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def identity(self):
        return "case_" + digest(self.model_dump(mode="json"))

    @property
    def search_text(self):
        return (
            "；".join(f.fault_type + " " + f.rationale for f in self.report.findings)
            + " "
            + self.report.summary
        )[:3000]

    def reference(self):
        # No scenario IDs, expected labels, old evidence IDs or measurement claims
        # are exposed as if they described the current observation.
        return {
            "case_id": self.identity,
            "kind": "historical_model_prediction",
            "system_id": self.request.system_id,
            "query_window": [self.request.start.isoformat(), self.request.end.isoformat()],
            "summary": self.report.summary,
            "predicted_findings": [
                {"fault_type": f.fault_type, "rationale": f.rationale, "confidence": f.confidence}
                for f in self.report.findings
            ],
            "matched_development_reference": self.is_correct,
            "not_independent_field_confirmation": True,
        }


class MemoryLesson(Model):
    version: Literal["development-lesson-v1"] = "development-lesson-v1"
    split: Literal["dev"] = "dev"
    title: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=2000)
    source_case_ids: list[str] = Field(min_length=1, max_length=20)
    outcome: Literal["accepted", "rejected"]
    reviewer: str = Field(min_length=1, max_length=80)
    review_reason: str = Field(min_length=1, max_length=1000)
    review_method: Literal["local_interactive", "controlled_test"]

    @model_validator(mode="after")
    def distinct_sources(self):
        if len(set(self.source_case_ids)) != len(self.source_case_ids):
            raise ValueError("duplicate lesson support cases")
        if not all(
            value.strip() for value in (self.title, self.text, self.reviewer, self.review_reason)
        ):
            raise ValueError("lesson review fields cannot be blank")
        return self

    @property
    def identity(self):
        return "lesson_" + digest(self.model_dump(mode="json"))

    @property
    def search_text(self):
        return self.title + "\n" + self.text

    def reference(self):
        return {
            "lesson_id": self.identity,
            "kind": "reviewed_development_lesson",
            "title": self.title,
            "text": self.text,
            "outcome": self.outcome,
            "source_case_ids": self.source_case_ids,
            "not_current_measurement_or_action_authorization": True,
        }


class MemorySnapshot(Model):
    version: Literal["pgvector-memory-v1"] = "pgvector-memory-v1"
    store_id: str
    revision: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedder: dict
    backend: dict
    dimensions: int = Field(ge=1, le=2000)
    cases: int = Field(ge=0)
    lessons: int = Field(ge=0)

    @property
    def sha256(self):
        return digest(self.model_dump(mode="json"))


def scored_development_case(manifest, scenario_id, run, *, review_bundle=None):
    scenes = manifest.get("scenarios", [])
    if not scenes or any(s["split"] != "dev" for s in scenes):
        raise ValueError("memory writes only accept development manifests")
    if len({s["scenario_id"] for s in scenes}) != len(scenes):
        raise ValueError("duplicate memory source scenario IDs")
    scenario = next((s for s in scenes if s["scenario_id"] == scenario_id), None)
    if (
        scenario is None
        or scenario.get("evaluation_eligible") is False
        or scenario.get("truth_scope") == "unmodified_baseline_candidate"
    ):
        raise ValueError("unreviewed or unknown memory source scenario")
    if not scenario["faults"] or manifest.get("review_catalog_sha256") is not None:
        if review_bundle is None:
            raise ValueError("normal/reviewed memories require the original baseline review bundle")
        from pvdiag.eval.baseline_review import verify_release

        verify_release(manifest, review_bundle)
    run = RunResult.model_validate(
        run.model_dump(mode="json") if isinstance(run, RunResult) else run
    )
    sources = set()
    catalog = None
    for key, raw in run.evidence.items():
        if key != raw.get("evidence_id"):
            raise ValueError("memory evidence map identity mismatch")
        if raw["tool"] not in CONTRACTS:
            raise ValueError("unknown memory evidence tool")
        evidence = Evidence[CONTRACTS[raw["tool"]][1]].model_validate(raw)
        payload = evidence.model_dump(mode="json", exclude={"evidence_id"})
        if (
            evidence.evidence_id
            != "ev_"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()[:24]
        ):
            raise ValueError("invalid memory evidence identity")
        if evidence.tool not in KNOWLEDGE_TOOLS:
            sources.add(evidence.source_sha256)
            if evidence.request["system_id"] != run.request.system_id:
                raise ValueError("memory evidence outside request system")
        if evidence.tool == "list_devices":
            catalog = evidence.result.devices
    if len(sources) != 1 or catalog is None:
        raise ValueError("memory requires one bound observation snapshot and device catalog")
    if not set(run.request.device_ids) <= {d.device_id for d in catalog if d.kind == "inverter"}:
        raise ValueError("memory request devices outside its observation catalog")
    scope = Scope(
        source_sha256=next(iter(sources)),
        device_ids=run.request.device_ids,
        poa_device_id=next((d.device_id for d in catalog if "poa_wm2" in d.metrics), None),
        ambient_device_id=next((d.device_id for d in catalog if "ambient_c" in d.metrics), None),
    )
    if check_report(run.report, run.evidence, run.request, scope):
        raise ValueError("memory report failed citation or scope checks")
    case = scorer.score_case(scenario, run)
    correct = case_passed(case)
    return MemoryCase(
        manifest_sha256=digest(manifest),
        scenario_id=scenario_id,
        run_sha256=digest(run.model_dump(mode="json")),
        observation_source_sha256=scope.source_sha256,
        harness_sha256=run.harness_sha256,
        request=run.request,
        report=run.report,
        model=run.model,
        is_correct=correct,
        scorer_version=scorer.SCORER_VERSION,
        scorer_sha256=hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest(),
        score_sha256=digest(case),
    )
