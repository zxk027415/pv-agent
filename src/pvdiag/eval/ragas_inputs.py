"""Frozen retrieval inputs for Ragas; preparation never calls a generator or judge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from pvdiag.eval.retrieval import QASet, Question
from pvdiag.knowledge.models import Manifest, SearchRequest, SearchResult
from pvdiag.mcp_servers.models import Evidence, Model
from pvdiag.observability.store import digest

RAGAS_VERSION = "0.4.3"


class RagasCase(Model):
    question: Question
    evidence: Evidence[SearchResult]


class RagasPlan(Model):
    version: Literal["ragas-input-v1"] = "ragas-input-v1"
    purpose: Literal["development_retrieval"] = "development_retrieval"
    annotation_status: str
    qa_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_config: dict
    top_k: int = Field(ge=1, le=5)
    ragas_version: Literal["0.4.3"] = RAGAS_VERSION
    cases: list[RagasCase] = Field(min_length=1, max_length=100)

    @property
    def sha256(self):
        return digest(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def consistent_evidence(self):
        ids = [c.question.question_id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate Ragas question IDs")
        for case in self.cases:
            evidence = case.evidence
            expected = SearchRequest(query=case.question.question, top_k=self.top_k)
            payload = evidence.model_dump(mode="json", exclude={"evidence_id"})
            identity = hashlib.sha256(
                json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if (
                evidence.evidence_id != "ev_" + identity[:24]
                or evidence.tool != "search_fault_kb"
                or evidence.source_sha256 != self.knowledge_source_sha256
                or evidence.request != expected.model_dump(mode="json")
                or evidence.result.retrieval_config != self.retrieval_config
                or evidence.result.retrieval_mode != self.retrieval_config.get("mode")
                or len(evidence.result.hits) > self.top_k
                or len({h.document.document_id for h in evidence.result.hits})
                != len(evidence.result.hits)
            ):
                raise ValueError("Ragas retrieval evidence identity or configuration mismatch")
        return self

    def required_calls(self):
        eligible = [c for c in self.cases if c.evidence.result.hits]
        return {
            "answer": len(eligible),
            "judge": sum(len(c.evidence.result.hits) + 3 for c in eligible),
        }


def prepare_plan(dataset, store, *, top_k=3, question_ids=None):
    known = {d.document_id for d in store.documents}
    if any(set(q.relevant_document_ids) - known for q in dataset.questions):
        raise ValueError("QA labels reference documents absent from this corpus")
    selected = question_ids or [q.question_id for q in dataset.questions]
    if len(selected) != len(set(selected)) or set(selected) - {
        q.question_id for q in dataset.questions
    }:
        raise ValueError("unknown or duplicate selected question IDs")
    if not 1 <= top_k <= 5:
        raise ValueError("invalid retrieval top_k")
    cases = [
        RagasCase(question=q, evidence=store.search(SearchRequest(query=q.question, top_k=top_k)))
        for q in dataset.questions
        if q.question_id in selected
    ]
    return RagasPlan(
        annotation_status=dataset.annotation_status,
        qa_sha256=digest(dataset.model_dump(mode="json")),
        corpus_manifest_sha256=hashlib.sha256(
            (store.root / "manifest.json").read_bytes()
        ).hexdigest(),
        knowledge_source_sha256=store.source_sha256,
        retrieval_config=store.retriever.config,
        top_k=top_k,
        cases=cases,
    )


def validate_sources(plan, dataset: QASet, corpus):
    """Check frozen contexts against live hash-checked documents; no neural reload needed."""
    manifest_bytes = (corpus.root / "manifest.json").read_bytes()
    if (
        plan.qa_sha256 != digest(dataset.model_dump(mode="json"))
        or plan.annotation_status != dataset.annotation_status
        or plan.corpus_manifest_sha256 != hashlib.sha256(manifest_bytes).hexdigest()
    ):
        raise ValueError("Ragas QA or corpus snapshot changed")
    source_payload = {
        "manifest": Manifest.model_validate_json(manifest_bytes).model_dump(),
        "retrieval": plan.retrieval_config,
    }
    expected_source = hashlib.sha256(
        json.dumps(source_payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if plan.knowledge_source_sha256 != expected_source:
        raise ValueError("Ragas retrieval configuration does not match its source fingerprint")
    questions = {q.question_id: q for q in dataset.questions}
    documents = {
        d.document_id: (d, text) for d, text in zip(corpus.documents, corpus.texts, strict=True)
    }
    for case in plan.cases:
        if case.question != questions.get(case.question.question_id):
            raise ValueError("frozen Ragas question differs from its QA source")
        if set(case.question.relevant_document_ids) - documents.keys():
            raise ValueError("reference document no longer exists")
        for hit in case.evidence.result.hits:
            if documents.get(hit.document.document_id) != (hit.document, hit.text):
                raise ValueError("frozen Ragas passage differs from its licensed corpus source")


def load_plan(path):
    return RagasPlan.model_validate_json(Path(path).read_bytes())
