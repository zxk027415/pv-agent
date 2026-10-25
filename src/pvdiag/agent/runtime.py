"""Managed graph lifetime and version-bound persistent start/resume operations."""

from __future__ import annotations

import hashlib
import platform
import re
from contextlib import AsyncExitStack, asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from langgraph.types import Command

from pvdiag.agent.gateway import MCPGateway
from pvdiag.agent.graph import build_graph
from pvdiag.agent.state import DiagRequest, RunResult
from pvdiag.mcp_servers.models import Model
from pvdiag.mcp_servers.pv_analytics import create_server as analytics_server
from pvdiag.mcp_servers.pv_data import create_server as data_server
from pvdiag.mcp_servers.pv_knowledge import create_server as knowledge_server


class SessionReply(Model):
    run_id: str
    status: Literal["completed", "awaiting_approval", "interrupted", "running"]
    result: RunResult | None = None
    approval_request: dict | None = None


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        path
        for folder in ("agent", "mcp_servers", "ops", "knowledge")
        for path in (root / folder).glob("*.py")
    )
    paths.append(root / "data" / "normalize.py")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(
            path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
        )
    for name in ("system_prompt.md", "verification_prompt.md"):
        digest.update(name.encode() + b"\0" + (root.parents[1] / "harness" / name).read_bytes())
    return digest.hexdigest()


def validate_run_id(run_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", run_id):
        raise ValueError("invalid persistent run ID")


@asynccontextmanager
async def open_runtime(
    store,
    reasoner,
    *,
    knowledge_store=None,
    ops_store=None,
    persistence=None,
    max_calls=30,
    max_loops=3,
    verification_mode="structural",
):
    async with AsyncExitStack() as stack:
        reader = await stack.enter_async_context(
            MCPGateway.connect(
                data_server(store),
                analytics_server(store),
                knowledge_server(knowledge_store) if knowledge_store else None,
                knowledge_source_sha256=knowledge_store.source_sha256 if knowledge_store else None,
            )
        )
        writer = None
        if ops_store is not None:
            from pvdiag.mcp_servers.pv_ops import create_server as ops_server
            from pvdiag.ops.gateway import OpsGateway

            writer = await stack.enter_async_context(
                OpsGateway.connect(ops_server(ops_store), ops_store)
            )
        graph = build_graph(
            reader,
            reasoner,
            max_calls=max_calls,
            max_loops=max_loops,
            checkpointer=persistence.saver if persistence else None,
            verification_mode=verification_mode,
            ops_gateway=writer,
        )
        packages = {}
        for name in (
            "langgraph",
            "langgraph-checkpoint",
            "langgraph-checkpoint-postgres",
            "fastmcp",
            "pydantic",
            "pandas",
            "numpy",
            "pyarrow",
            "pvlib",
            "langchain",
            "langchain-core",
            "langchain-openai",
            "openai",
            "psycopg",
            "psycopg-binary",
        ):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = "not_installed"
        contract = {
            "version": "persistent-run-v1",
            "source_sha256": source_fingerprint(),
            "observation_source_sha256": store.source_sha256,
            "knowledge_source_sha256": reader.knowledge_source_sha256,
            "ops_store_sha256": hashlib.sha256(str(ops_store.path.resolve()).encode()).hexdigest()
            if ops_store
            else None,
            "model": {
                key: value for key, value in reasoner.metadata.items() if key != "reported_models"
            },
            "reasoner_class": f"{type(reasoner).__module__}.{type(reasoner).__qualname__}",
            "verification_mode": verification_mode,
            "max_calls": max_calls,
            "max_loops": max_loops,
            "packages": packages,
            "python_version": platform.python_version(),
        }
        yield DiagnosisRuntime(graph, reasoner, contract, persistence)


class DiagnosisRuntime:
    def __init__(self, graph, reasoner, contract, persistence):
        self.graph, self.reasoner = graph, reasoner
        self.contract, self.persistence = contract, persistence

    @staticmethod
    def config(run_id):
        validate_run_id(run_id)
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 40}

    async def view(self, run_id):
        snapshot = await self.graph.aget_state(self.config(run_id))
        if not snapshot.next and snapshot.values.get("result"):
            return SessionReply(
                run_id=run_id,
                status="completed",
                result=RunResult.model_validate(snapshot.values["result"]),
            )
        interruptions = [item.value for task in snapshot.tasks for item in task.interrupts]
        if interruptions:
            return SessionReply(
                run_id=run_id, status="awaiting_approval", approval_request=interruptions[0]
            )
        return SessionReply(run_id=run_id, status="interrupted")

    async def _accounting(self, run_id):
        if self.reasoner.metadata.get("mode") == "LLM":
            if not hasattr(self.reasoner, "call_tracker"):
                raise ValueError("persistent LLM runs require durable model accounting")
            self.reasoner.call_tracker = self.persistence.tracker(run_id)
            await self.reasoner.sync_accounting()

    async def start(self, run_id, request):
        if self.persistence is None:
            raise ValueError("persistent start requires a PostgreSQL store")
        config = self.config(run_id)
        request = DiagRequest.model_validate(request)
        async with self.persistence.exclusive_run(run_id):
            await self.persistence.create_run(
                run_id,
                request,
                self.contract,
                self.reasoner.max_model_calls if self.reasoner.metadata.get("mode") == "LLM" else 0,
            )
            await self._accounting(run_id)
            await self.graph.ainvoke({"request": request}, config=config, durability="sync")
            return await self.view(run_id)

    async def resume(self, run_id, decision=None):
        if self.persistence is None:
            raise ValueError("persistent resume requires a PostgreSQL store")
        config = self.config(run_id)
        async with self.persistence.exclusive_run(run_id):
            run = await self.persistence.get_run(run_id)
            if run["contract"] != self.contract:
                raise ValueError(
                    "persistent run context changed; restore the original data, code and configuration"
                )
            await self._accounting(run_id)
            reply = await self.view(run_id)
            if (
                reply.status == "completed"
                or reply.status == "awaiting_approval"
                and decision is None
            ):
                return reply
            snapshot = await self.graph.aget_state(config)
            if decision is not None:
                if reply.status != "awaiting_approval":
                    raise ValueError("run is not awaiting human approval")
                command = Command(resume=decision)
            elif snapshot.values:
                command = None
            else:
                command = {"request": DiagRequest.model_validate(run["request"])}
            await self.graph.ainvoke(command, config=config, durability="sync")
            return await self.view(run_id)
