"""Native FastMCP gateway: SDK v2 is incompatible with adapters 0.3.2's SDK v1."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import AsyncExitStack, asynccontextmanager
from time import perf_counter

from fastmcp import Client

from pvdiag.agent.ablation import ANALYTICS_TOOLS, DATA_TOOLS, SUMMARY_TOOLS
from pvdiag.agent.state import DiagRequest, Scope, ToolCall, ToolTrace
from pvdiag.knowledge.models import (
    KNOWLEDGE_TOOLS,
    PlaybookRequest,
    PlaybookResult,
    SearchRequest,
    SearchResult,
)
from pvdiag.mcp_servers.models import (
    AnomaliesResult,
    CompareRequest,
    DetectRequest,
    DevicesResult,
    Evidence,
    ExpectedRequest,
    ExpectedResult,
    PatternsResult,
    PeerRequest,
    PeersResult,
    PeriodsResult,
    QualityRequest,
    QualityResult,
    SiteRequest,
    SiteResult,
    TimeseriesRequest,
    TimeseriesResult,
)
from pvdiag.observability.tracing import mark_request_sent, observe

CONTRACTS = {
    "list_devices": (SiteRequest, DevicesResult),
    "get_site_info": (SiteRequest, SiteResult),
    "get_timeseries": (TimeseriesRequest, TimeseriesResult),
    "data_quality": (QualityRequest, QualityResult),
    "peer_compare": (PeerRequest, PeersResult),
    "detect_anomalies": (DetectRequest, AnomaliesResult),
    "profile_pattern": (PeerRequest, PatternsResult),
    "compare_periods": (CompareRequest, PeriodsResult),
    "expected_power": (ExpectedRequest, ExpectedResult),
    "search_fault_kb": (SearchRequest, SearchResult),
    "get_playbook": (PlaybookRequest, PlaybookResult),
}
BOOTSTRAP = {"list_devices", "get_site_info"}
INVESTIGATE = set(CONTRACTS) - BOOTSTRAP


class MCPGateway:
    def __init__(self, clients, tools, knowledge_source_sha256=None, allowed_tools=None):
        self.clients = clients
        self.allowed_tools = frozenset(INVESTIGATE if allowed_tools is None else allowed_tools)
        self.tools = (
            tools
            if allowed_tools is None
            else {name: tool for name, tool in tools.items() if name in self.allowed_tools}
        )
        self.knowledge_source_sha256 = knowledge_source_sha256

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        data_transport,
        analytics_transport=None,
        knowledge_transport=None,
        *,
        knowledge_source_sha256=None,
        allowed_tools=None,
    ):
        if (knowledge_transport is None) != (knowledge_source_sha256 is None):
            raise ValueError("knowledge transport requires an explicitly pinned corpus fingerprint")
        required = set(DATA_TOOLS)
        transports = [data_transport]
        if analytics_transport is not None:
            required |= ANALYTICS_TOOLS
            transports.append(analytics_transport)
        if knowledge_transport is not None:
            required |= KNOWLEDGE_TOOLS
            transports.append(knowledge_transport)
        async with AsyncExitStack() as stack:
            clients, catalog = {}, {}
            for transport in transports:
                client = await stack.enter_async_context(Client(transport, timeout=30))
                for tool in await client.list_tools():
                    if tool.name not in required:
                        continue
                    if tool.name in clients:
                        raise ValueError("duplicate MCP tool names")
                    if not tool.annotations or not tool.annotations.read_only_hint:
                        raise ValueError("diagnosis requires read-only tools")
                    if tool.annotations.destructive_hint:
                        raise ValueError("destructive tool is not allowed")
                    clients[tool.name] = client
                    catalog[tool.name] = {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
            if set(clients) != required:
                raise ValueError("MCP servers do not expose the required tools")
            if allowed_tools is not None and not set(allowed_tools) <= required - BOOTSTRAP:
                raise ValueError("ablation tools are not mounted")
            yield cls(clients, catalog, knowledge_source_sha256, allowed_tools)

    async def call(self, call: ToolCall, request: DiagRequest, scope: Scope | None, node: str):
        with observe("tool", call.name, call.arguments, {"node": node}) as observation:
            evidence, trace = await self._call(call, request, scope, node)
            observation.update(
                output=evidence, metadata={"error_code": trace.error}, success=trace.success
            )
            return evidence, trace

    async def _call(self, call: ToolCall, request: DiagRequest, scope: Scope | None, node: str):
        start = perf_counter()
        try:
            allowed = {
                "guard_in": BOOTSTRAP,
                "fixed_summary": SUMMARY_TOOLS,
                "investigate": self.allowed_tools,
            }.get(node, frozenset())
            if call.name not in allowed:
                raise ValueError("tool is not allowed in this node")
            if call.name not in self.clients:
                raise ValueError("tool is not enabled")
            input_type, result_type = CONTRACTS[call.name]
            validated = input_type.model_validate(call.arguments)
            if call.name not in KNOWLEDGE_TOOLS and validated.system_id != request.system_id:
                raise ValueError("system outside request scope")
            if hasattr(validated, "device_ids"):
                allowed_devices = set(scope.device_ids) | {
                    scope.poa_device_id,
                    scope.ambient_device_id,
                }
                if not set(validated.device_ids) <= allowed_devices:
                    raise ValueError("devices outside request scope")
                if validated.start < request.start or validated.end > request.end:
                    raise ValueError("window outside request scope")
                if hasattr(validated, "reference_start") and (
                    validated.reference_start < request.start
                    or validated.reference_end > request.end
                ):
                    raise ValueError("reference outside request scope")
                for field in ("poa_device_id", "ambient_device_id"):
                    if hasattr(validated, field) and getattr(validated, field) != getattr(
                        scope, field
                    ):
                        raise ValueError("sensor outside request scope")
            mark_request_sent()
            response = await asyncio.wait_for(
                self.clients[call.name].call_tool(
                    call.name, {"request": validated.model_dump(mode="json")}
                ),
                timeout=30,
            )
            evidence = (
                Evidence[result_type]
                .model_validate(response.structured_content)
                .model_dump(mode="json")
            )
            payload = {key: value for key, value in evidence.items() if key != "evidence_id"}
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if evidence["evidence_id"] != f"ev_{digest[:24]}" or evidence["tool"] != call.name:
                raise ValueError("invalid evidence identity")
            if evidence["request"] != validated.model_dump(mode="json"):
                raise ValueError("evidence request mismatch")
            expected_source = (
                self.knowledge_source_sha256
                if call.name in KNOWLEDGE_TOOLS
                else scope.source_sha256
                if scope is not None
                else None
            )
            if expected_source is not None and evidence["source_sha256"] != expected_source:
                raise ValueError("MCP servers use different observation snapshots")
            return evidence, ToolTrace(
                name=call.name,
                node=node,
                success=True,
                duration_seconds=perf_counter() - start,
                evidence_id=evidence["evidence_id"],
            )
        except Exception as exc:  # noqa: BLE001 -- protocol boundary: sanitize all remote failures
            # Never copy provider/server exception text (which may contain keys,
            # paths or response bodies) into model-visible traces.
            return None, ToolTrace(
                name=call.name,
                node=node,
                success=False,
                duration_seconds=perf_counter() - start,
                error=type(exc).__name__,
            )
