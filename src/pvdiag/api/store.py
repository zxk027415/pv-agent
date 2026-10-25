"""Durable API ownership, idempotency and replayable progress events."""

import hashlib
import json
from uuid import uuid4


class ApiStore:
    def __init__(self, persistence):
        self.persistence = persistence

    async def initialize(self):
        async with self.persistence._mutex:
            await self.persistence.connection.execute("""
                CREATE TABLE IF NOT EXISTS pvdiag_api_runs (
                    run_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    request JSONB NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    idempotency_sha256 TEXT NOT NULL,
                    UNIQUE(owner, idempotency_sha256)
                )
            """)
            await self.persistence.connection.execute("""
                CREATE TABLE IF NOT EXISTS pvdiag_api_events (
                    run_id TEXT NOT NULL REFERENCES pvdiag_api_runs(run_id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    PRIMARY KEY(run_id, sequence)
                )
            """)

    async def ensure_initialized(self):
        async with self.persistence._mutex:
            row = await (
                await self.persistence.connection.execute(
                    "SELECT to_regclass('pvdiag_api_runs') AS runs, to_regclass('pvdiag_api_events') AS events"
                )
            ).fetchone()
        if not row["runs"] or not row["events"]:
            raise ValueError("API tables are not initialized; run the explicit API init command")

    async def find(self, owner, key):
        digest = hashlib.sha256(key.encode()).hexdigest()
        async with self.persistence._mutex:
            return await (
                await self.persistence.connection.execute(
                    "SELECT * FROM pvdiag_api_runs WHERE owner=%s AND idempotency_sha256=%s",
                    (owner, digest),
                )
            ).fetchone()

    async def register(self, owner, key, profile_id, request):
        payload = {"profile_id": profile_id, "request": request.model_dump(mode="json")}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        async with self.persistence._mutex, self.persistence.connection.transaction():
            row = await (
                await self.persistence.connection.execute(
                    "INSERT INTO pvdiag_api_runs VALUES (%s,%s,%s,%s::jsonb,%s,%s) ON CONFLICT(owner,idempotency_sha256) DO NOTHING RETURNING *",
                    (
                        uuid4().hex,
                        owner,
                        profile_id,
                        request.model_dump_json(),
                        digest,
                        hashlib.sha256(key.encode()).hexdigest(),
                    ),
                )
            ).fetchone()
            if row is None:
                row = await (
                    await self.persistence.connection.execute(
                        "SELECT * FROM pvdiag_api_runs WHERE owner=%s AND idempotency_sha256=%s",
                        (owner, hashlib.sha256(key.encode()).hexdigest()),
                    )
                ).fetchone()
            if row["request_sha256"] != digest:
                raise ValueError("idempotency key already binds a different diagnosis request")
            return row

    async def get(self, run_id):
        async with self.persistence._mutex:
            return await (
                await self.persistence.connection.execute(
                    "SELECT * FROM pvdiag_api_runs WHERE run_id=%s", (run_id,)
                )
            ).fetchone()

    async def append(self, run_id, event_type, payload):
        async with self.persistence._mutex, self.persistence.connection.transaction():
            await self.persistence.connection.execute(
                "SELECT run_id FROM pvdiag_api_runs WHERE run_id=%s FOR UPDATE", (run_id,)
            )
            if event_type in {"completed", "awaiting_approval", "error"}:
                previous = await (
                    await self.persistence.connection.execute(
                        "SELECT sequence,event_type,payload FROM pvdiag_api_events WHERE run_id=%s ORDER BY sequence DESC LIMIT 1",
                        (run_id,),
                    )
                ).fetchone()
                if (
                    previous
                    and previous["event_type"] == event_type
                    and previous["payload"] == payload
                ):
                    return previous["sequence"]
            seq = (
                await (
                    await self.persistence.connection.execute(
                        "SELECT coalesce(max(sequence),0)+1 AS next FROM pvdiag_api_events WHERE run_id=%s",
                        (run_id,),
                    )
                ).fetchone()
            )["next"]
            await self.persistence.connection.execute(
                "INSERT INTO pvdiag_api_events VALUES (%s,%s,%s,%s::jsonb)",
                (run_id, seq, event_type, json.dumps(payload, allow_nan=False)),
            )
            return seq

    async def events(self, run_id, after):
        async with self.persistence._mutex:
            return await (
                await self.persistence.connection.execute(
                    "SELECT sequence,event_type,payload FROM pvdiag_api_events WHERE run_id=%s AND sequence>%s ORDER BY sequence LIMIT 100",
                    (run_id, after),
                )
            ).fetchall()

    async def latest(self, run_id):
        async with self.persistence._mutex:
            return (
                await (
                    await self.persistence.connection.execute(
                        "SELECT coalesce(max(sequence),0) AS latest FROM pvdiag_api_events WHERE run_id=%s",
                        (run_id,),
                    )
                ).fetchone()
            )["latest"]

    async def last_event(self, run_id):
        async with self.persistence._mutex:
            return await (
                await self.persistence.connection.execute(
                    "SELECT sequence,event_type,payload FROM pvdiag_api_events WHERE run_id=%s ORDER BY sequence DESC LIMIT 1",
                    (run_id,),
                )
            ).fetchone()
