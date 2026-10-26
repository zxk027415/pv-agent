"""Owned background diagnoses, durable SSE progress and identity-bound approvals."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import Field, StrictBool

from pvdiag.agent.persistence import PostgresStore
from pvdiag.agent.runtime import SessionReply, open_runtime
from pvdiag.agent.sessions import inspect_run
from pvdiag.agent.state import ApprovalDecision, DiagRequest, PendingAction
from pvdiag.api.config import Identity
from pvdiag.api.store import ApiStore
from pvdiag.mcp_servers.models import Model

LOGGER = logging.getLogger(__name__)


class DiagnosisInput(Model):
    profile_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    request: DiagRequest


class ApprovalInput(Model):
    approved: StrictBool
    action_id: str = Field(pattern=r"^act_[0-9a-f]{32}$")
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(default="", max_length=1000)


class ApiService:
    def __init__(self, persistence, profiles, max_active, trace_store=None):
        self.persistence, self.profiles = persistence, profiles
        self.repository = ApiStore(persistence)
        self.max_active = max_active
        self.jobs = {}
        self.lock = asyncio.Lock()
        self.trace_store = trace_store

    async def inspect(self, row):
        try:
            return await inspect_run(self.persistence, row["run_id"])
        except ValueError:
            return SessionReply(
                run_id=row["run_id"],
                status="running" if row["run_id"] in self.jobs else "interrupted",
            )

    async def spawn(self, row, decision=None):
        run_id = row["run_id"]
        if run_id in self.jobs:
            return
        if len(self.jobs) >= self.max_active:
            raise HTTPException(429, "diagnosis capacity reached; retry this run later")
        await self.repository.append(run_id, "accepted", {"run_id": run_id})
        self.jobs[run_id] = asyncio.create_task(self._drive(row, decision))

    async def _drive(self, row, decision):
        run_id = row["run_id"]
        try:
            reply = await self.inspect(row)
            if reply.status != "completed":
                profile = self.profiles[row["profile_id"]]

                async def event_sink(kind, payload):
                    await self.repository.append(run_id, kind, payload)

                async with open_runtime(
                    profile.observations,
                    profile.reasoner(),
                    knowledge_store=profile.knowledge,
                    ops_store=profile.ops,
                    persistence=self.persistence,
                    verification_mode=profile.config.verification,
                    event_sink=event_sink,
                    trace_store=self.trace_store,
                ) as runtime:
                    try:
                        await self.persistence.get_run(run_id)
                    except ValueError:
                        reply = await runtime.start(
                            run_id, DiagRequest.model_validate(row["request"])
                        )
                    else:
                        reply = await runtime.resume(
                            run_id, decision or self.stored_decision(profile, reply)
                        )
                    if reply.status == "awaiting_approval":
                        recorded = self.stored_decision(profile, reply)
                        if recorded is not None:
                            reply = await runtime.resume(run_id, recorded)
            await self.repository.append(run_id, reply.status, reply.model_dump(mode="json"))
        except asyncio.CancelledError:
            await self._record_error(run_id, "worker_interrupted")
            raise
        except ValueError as error:
            if "already active" not in str(error):
                await self._record_error(run_id, "run_context_or_state_conflict")
        except Exception:  # noqa: BLE001 -- no provider/database/filesystem details enter SSE
            await self._record_error(run_id, "run_failed")
        finally:
            self.jobs.pop(run_id, None)

    async def _record_error(self, run_id, code):
        try:
            await self.repository.append(run_id, "error", {"run_id": run_id, "code": code})
        except Exception:  # noqa: BLE001 -- an unavailable database cannot persist another event
            LOGGER.warning("api_event_storage_unavailable")

    @staticmethod
    def stored_decision(profile, reply):
        if profile.ops is None or reply.status != "awaiting_approval":
            return None
        proposal = reply.approval_request["proposal"]
        for record in profile.ops.approvals():
            if (
                record.action_id == proposal["action_id"]
                and record.payload_sha256 == proposal["payload_sha256"]
            ):
                return record.model_dump(mode="json", exclude={"approval_id", "decided_at"})
        return None

    async def close(self):
        jobs = list(self.jobs.values())
        for task in jobs:
            task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)


def create_app(
    profiles,
    authentication,
    dsn,
    *,
    schema="pvdiag",
    max_active=2,
    poll_interval=0.1,
    heartbeat_interval=5,
    trace_store=None,
):
    if not 1 <= max_active <= 16:
        raise ValueError("invalid API worker limit")
    if any(not identity.profiles <= profiles.keys() for identity in authentication.identities):
        raise ValueError("identity references an unknown observation profile")

    @asynccontextmanager
    async def lifespan(app):
        async with PostgresStore.open(dsn, schema=schema) as persistence:
            service = ApiService(persistence, profiles, max_active, trace_store)
            await service.repository.ensure_initialized()
            app.state.service = service
            try:
                yield
            finally:
                await service.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse(status_code=422, content={"detail": "invalid request"})

    @app.middleware("http")
    async def unavailable_boundary(request, call_next):
        try:
            return await call_next(request)
        except Exception as error:  # noqa: BLE001 -- do not log credential-bearing database errors
            LOGGER.warning("api_request_failed: %s", type(error).__name__)
            return JSONResponse(status_code=503, content={"detail": "service unavailable"})

    def principal(authorization: str | None = Header(default=None)):
        identity = authentication.authenticate(authorization)
        if identity is None:
            raise HTTPException(
                401, "authentication required", headers={"WWW-Authenticate": "Bearer"}
            )
        return identity

    Principal = Annotated[Identity, Depends(principal)]

    async def visible_run(run_id, identity):
        row = await app.state.service.repository.get(run_id)
        if (
            row is None
            or row["profile_id"] not in identity.profiles
            or not (
                "approve" in identity.roles
                or row["owner"] == identity.subject
                and identity.roles & {"read", "diagnose"}
            )
        ):
            raise HTTPException(404, "run not found")
        return row

    def cursor(run_id, value):
        if value is None:
            return 0
        prefix, separator, sequence = value.partition(":")
        if not separator or prefix != run_id or not sequence.isdigit() or len(sequence) > 12:
            raise HTTPException(400, "invalid event cursor")
        return int(sequence)

    def stream(row, request, after=0):
        service = app.state.service
        run_id = row["run_id"]

        async def messages():
            nonlocal after
            heartbeat = asyncio.get_running_loop().time()
            try:
                while not await request.is_disconnected():
                    events = await service.repository.events(run_id, after)
                    if events:
                        for event in events:
                            after = event["sequence"]
                            data = json.dumps(event["payload"], ensure_ascii=False, allow_nan=False)
                            yield f"id: {run_id}:{after}\nevent: {event['event_type']}\ndata: {data}\n\n"
                            if event["event_type"] in {
                                "completed",
                                "awaiting_approval",
                                "error",
                            } and after == await service.repository.latest(run_id):
                                return
                    elif run_id not in service.jobs and not await service.persistence.is_active(
                        run_id
                    ):
                        reply = await service.inspect(row)
                        if reply.status in {"completed", "awaiting_approval"}:
                            previous = await service.repository.last_event(run_id)
                            if (
                                previous
                                and previous["event_type"] == reply.status
                                and previous["sequence"] <= after
                            ):
                                return
                            await service.repository.append(
                                run_id, reply.status, reply.model_dump(mode="json")
                            )
                        elif reply.status == "interrupted":
                            await service._record_error(run_id, "run_interrupted")
                    if asyncio.get_running_loop().time() - heartbeat >= heartbeat_interval:
                        yield ": keep-alive\n\n"
                        heartbeat = asyncio.get_running_loop().time()
                    await asyncio.sleep(poll_interval)
            except Exception:  # noqa: BLE001 -- redact connection/provider errors at the stream boundary
                yield 'event: error\ndata: {"code":"stream_unavailable"}\n\n'

        return StreamingResponse(
            messages(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Run-ID": run_id,
            },
        )

    @app.get("/health")
    async def health():
        return {"status": "ready"}

    @app.get("/profiles")
    async def list_profiles(identity: Principal):
        return [profiles[key].public() for key in sorted(identity.profiles)]

    @app.post("/diagnose")
    async def diagnose(
        body: DiagnosisInput,
        request: Request,
        identity: Principal,
        idempotency_key: str = Header(min_length=8, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$"),
    ):
        if "diagnose" not in identity.roles:
            raise HTTPException(403, "diagnosis role required")
        if body.profile_id not in identity.profiles or body.profile_id not in profiles:
            raise HTTPException(404, "profile not found")
        try:
            profiles[body.profile_id].validate(body.request)
        except ValueError:
            raise HTTPException(422, "diagnosis request is outside profile scope") from None
        service = app.state.service
        async with service.lock:
            existing = await service.repository.find(identity.subject, idempotency_key)
            if existing is None and len(service.jobs) >= service.max_active:
                raise HTTPException(429, "diagnosis capacity reached")
            try:
                row = await service.repository.register(
                    identity.subject, idempotency_key, body.profile_id, body.request
                )
            except ValueError:
                raise HTTPException(409, "idempotency key binds a different request") from None
            reply = await service.inspect(row)
            if reply.status in {"interrupted", "running"} and row["run_id"] not in service.jobs:
                await service.spawn(row)
        return stream(row, request)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, identity: Principal):
        row = await visible_run(run_id, identity)
        return (await app.state.service.inspect(row)).model_dump(mode="json")

    @app.get("/runs/{run_id}/events")
    async def events(
        run_id: str,
        request: Request,
        identity: Principal,
        last_event_id: str | None = Header(default=None),
    ):
        row = await visible_run(run_id, identity)
        after = cursor(run_id, last_event_id)
        last = await app.state.service.repository.last_event(run_id)
        if after > (last["sequence"] if last else 0):
            raise HTTPException(400, "event cursor is ahead of this run")
        if (
            last
            and after == last["sequence"]
            and last["event_type"] in {"completed", "awaiting_approval", "error"}
        ):
            return Response(status_code=204)
        return stream(row, request, after)

    @app.post("/runs/{run_id}/resume")
    async def resume(run_id: str, request: Request, identity: Principal):
        row = await visible_run(run_id, identity)
        if row["owner"] != identity.subject or "diagnose" not in identity.roles:
            raise HTTPException(403, "only the diagnosing owner can resume this run")
        service = app.state.service
        async with service.lock:
            after = await service.repository.latest(run_id)
            reply = await service.inspect(row)
            if (
                reply.status == "interrupted"
                or reply.status == "awaiting_approval"
                and service.stored_decision(profiles[row["profile_id"]], reply) is not None
            ):
                await service.spawn(row)
            elif reply.status in {"completed", "awaiting_approval"}:
                return reply.model_dump(mode="json")
        return stream(row, request, after)

    @app.post("/runs/{run_id}/approve")
    async def approve(run_id: str, body: ApprovalInput, request: Request, identity: Principal):
        if "approve" not in identity.roles:
            raise HTTPException(403, "approval role required")
        row = await visible_run(run_id, identity)
        decision = ApprovalDecision(**body.model_dump(), reviewer=identity.subject)
        service = app.state.service
        async with service.lock:
            reply = await service.inspect(row)
            if reply.status == "completed":
                previous = reply.result.approval
                if (
                    previous
                    and previous.model_dump(exclude={"approval_id", "decided_at"})
                    == decision.model_dump()
                ):
                    return reply.model_dump(mode="json")
                raise HTTPException(409, "run already decided")
            if reply.status != "awaiting_approval":
                raise HTTPException(409, "run is not awaiting approval")
            proposal = reply.approval_request["proposal"]
            if (body.action_id, body.payload_sha256) != (
                proposal["action_id"],
                proposal["payload_sha256"],
            ):
                raise HTTPException(409, "approval content does not match the displayed proposal")
            ledger = profiles[row["profile_id"]].ops
            if ledger is None:
                raise HTTPException(409, "work orders are not enabled for this profile")
            try:
                ledger.decide(PendingAction.model_validate(proposal), decision)
            except ValueError:
                raise HTTPException(409, "a different decision is already recorded") from None
            active = service.jobs.get(run_id)
        if active is not None:
            await asyncio.shield(active)
        async with service.lock:
            after = await service.repository.latest(run_id)
            await service.spawn(row, decision.model_dump(mode="json"))
        return stream(row, request, after)

    return app
