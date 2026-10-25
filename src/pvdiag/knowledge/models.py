"""Bounded knowledge contracts; references are never plant observations."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from pvdiag.mcp_servers.models import Model

FaultType = Literal["F1", "F2", "F3", "F4", "F5", "F6"]
KNOWLEDGE_TOOLS = {"search_fault_kb", "get_playbook"}


class SearchRequest(Model):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)
    fault_type: FaultType | None = None

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if len(value.strip()) < 2:
            raise ValueError("query must contain at least two nonblank characters")
        return value.strip()


class PlaybookRequest(Model):
    fault_type: FaultType


class Document(Model):
    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")
    title: str = Field(min_length=1, max_length=200)
    kind: Literal["upstream_excerpt", "project_playbook"]
    fault_types: list[FaultType] = Field(min_length=1, max_length=6)
    path: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str
    source_version: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: Literal["BSD-3-Clause", "project-authored"]
    license_path: str | None = None
    license_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attribution: str

    @model_validator(mode="after")
    def licensed(self):
        if self.kind == "upstream_excerpt":
            if self.license != "BSD-3-Clause" or not self.license_path or not self.license_sha256:
                raise ValueError("upstream excerpts require the reviewed BSD license snapshot")
            if not self.source_url.startswith("https://github.com/pvlib/pvlib-python/"):
                raise ValueError("unreviewed upstream source")
        elif self.license != "project-authored":
            raise ValueError("playbooks must identify project authorship")
        return self


class Manifest(Model):
    version: Literal["kb-v1"]
    documents: list[Document] = Field(min_length=1, max_length=200)


class Passage(Model):
    document: Document
    text: str = Field(min_length=1, max_length=7000)
    score: float | None = None


class SearchResult(Model):
    evidence_kind: Literal["knowledge_reference"] = "knowledge_reference"
    retrieval_mode: Literal["bm25", "hybrid_bge"]
    retrieval_config: dict
    hits: list[Passage] = Field(max_length=5)
    candidate_count: int


class PlaybookResult(Model):
    evidence_kind: Literal["knowledge_reference"] = "knowledge_reference"
    playbook: Passage
