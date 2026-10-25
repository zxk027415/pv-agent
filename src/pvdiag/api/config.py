"""Server-owned observation profiles and hashed local API credentials."""

import hashlib
import json
import secrets
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import Field, model_validator

from pvdiag.agent.__main__ import reasoner_from_args
from pvdiag.mcp_servers.models import Model
from pvdiag.mcp_servers.pv_knowledge import knowledge_from_args
from pvdiag.mcp_servers.store import ObservationStore


class Identity(Model):
    subject: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")
    token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roles: set[Literal["diagnose", "read", "approve"]] = Field(min_length=1)
    profiles: set[str] = Field(min_length=1)


class Authentication:
    def __init__(self, identities):
        self.identities = [Identity.model_validate(item) for item in identities]
        if (
            not self.identities
            or len({i.subject for i in self.identities}) != len(self.identities)
            or len({i.token_sha256 for i in self.identities}) != len(self.identities)
        ):
            raise ValueError("configure unique named API identities and token digests")

    def authenticate(self, authorization):
        if not authorization or not authorization.startswith("Bearer "):
            return None
        token = authorization[7:]
        if not token or len(token) > 1024:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        found = None
        for identity in self.identities:
            if secrets.compare_digest(digest, identity.token_sha256):
                found = identity
        return found


class ProfileConfig(Model):
    profile_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    observations: Path
    metadata: Path
    parameters: Path | None = None
    allowed_device_ids: list[str] = Field(min_length=1, max_length=24)
    mode: Literal["b0", "llm"] = "b0"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    credential_env: str | None = None
    max_model_calls: int = Field(default=6, ge=1, le=6)
    verification: Literal["structural", "domain", "domain_llm"] = "domain"
    knowledge_root: Path | None = None
    retrieval_mode: Literal["bm25", "hybrid_bge"] = "bm25"
    embedding_model_path: Path | None = None
    reranker_model_path: Path | None = None
    ops_db: Path | None = None

    @model_validator(mode="after")
    def model_review_requires_llm(self):
        if self.verification == "domain_llm" and self.mode != "llm":
            raise ValueError("model review requires a model-backed profile")
        if len(self.allowed_device_ids) != len(set(self.allowed_device_ids)):
            raise ValueError("profile device scope must be unique")
        return self


class Profile:
    def __init__(self, config, *, allow_model_calls=False, reasoner_factory=None):
        self.config = ProfileConfig.model_validate(config)
        if self.config.mode == "llm" and not allow_model_calls:
            raise ValueError("LLM profiles require explicit server --allow-model-calls")
        self.observations = ObservationStore(
            self.config.observations, self.config.metadata, self.config.parameters
        )
        if (
            not set(self.config.allowed_device_ids)
            <= self.observations.inverters & self.observations.catalog.keys()
        ):
            raise ValueError("profile device scope is outside its observation snapshot")
        self.knowledge = knowledge_from_args(SimpleNamespace(**self.config.model_dump()))
        self.ops = None
        if self.config.ops_db is not None:
            from pvdiag.ops.store import WorkOrderStore

            self.ops = WorkOrderStore(self.config.ops_db)
        self._factory = reasoner_factory or (
            lambda: reasoner_from_args(
                SimpleNamespace(**self.config.model_dump(), allow_model_calls=allow_model_calls)
            )
        )

    def reasoner(self):
        return self._factory()

    def validate(self, request):
        frame = self.observations.frame
        if request.system_id != self.observations.system_id or not set(request.device_ids) <= set(
            self.config.allowed_device_ids
        ):
            raise ValueError("diagnosis request is outside profile scope")
        if request.start < frame.ts.min() or request.end > frame.ts.max() + timedelta(minutes=15):
            raise ValueError("diagnosis window is outside profile scope")

    def public(self):
        return {
            "profile_id": self.config.profile_id,
            "system_id": self.observations.system_id,
            "device_ids": self.config.allowed_device_ids,
            "start": self.observations.frame.ts.min().isoformat(),
            "end": (self.observations.frame.ts.max() + timedelta(minutes=15)).isoformat(),
            "mode": self.config.mode,
            "max_model_calls": self.config.max_model_calls if self.config.mode == "llm" else 0,
            "work_orders_enabled": self.ops is not None,
        }


def load_profiles(path, *, allow_model_calls=False):
    configs = json.loads(Path(path).read_text())
    profiles = [Profile(config, allow_model_calls=allow_model_calls) for config in configs]
    if not profiles or len({p.config.profile_id for p in profiles}) != len(profiles):
        raise ValueError("configure unique server-owned observation profiles")
    return {p.config.profile_id: p for p in profiles}
