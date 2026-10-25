"""Explicit PostgreSQL initialization, isolated checkpoints and durable model budgets."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from pvdiag.agent import state
from pvdiag.mcp_servers.models import Model


def checkpoint_serializer():
    allowed = [
        (value.__module__, value.__name__)
        for value in vars(state).values()
        if inspect.isclass(value) and issubclass(value, Model)
    ]
    return JsonPlusSerializer(
        pickle_fallback=False, allowed_msgpack_modules=allowed, allowed_json_modules=allowed
    )


class PostgresStore:
    def __init__(self, connection, saver, dsn, schema):
        self.connection, self.saver = connection, saver
        self._dsn, self.schema = dsn, schema
        self._mutex = asyncio.Lock()

    @classmethod
    @asynccontextmanager
    async def open(cls, dsn, *, schema="pvdiag", initialize=False):
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema)
            or schema in {"public", "information_schema"}
            or schema.startswith("pg_")
        ):
            raise ValueError("use a dedicated non-system PostgreSQL schema")
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg import AsyncConnection, Error, sql
            from psycopg.rows import dict_row
        except ImportError:
            raise RuntimeError("install the optional persistence requirements") from None
        try:
            async with AsyncExitStack() as stack:
                connection = await stack.enter_async_context(
                    await AsyncConnection.connect(
                        dsn,
                        autocommit=True,
                        row_factory=dict_row,
                        connect_timeout=10,
                        prepare_threshold=0,
                    )
                )
                if initialize:
                    await connection.execute(
                        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
                    )
                await connection.execute(
                    sql.SQL("SET search_path TO {}").format(sql.Identifier(schema))
                )
                saver_connection = await stack.enter_async_context(
                    await AsyncConnection.connect(
                        dsn,
                        autocommit=True,
                        row_factory=dict_row,
                        connect_timeout=10,
                        prepare_threshold=0,
                    )
                )
                await saver_connection.execute(
                    sql.SQL("SET search_path TO {}").format(sql.Identifier(schema))
                )
                saver = AsyncPostgresSaver(saver_connection, serde=checkpoint_serializer())
                store = cls(connection, saver, dsn, schema)
                if initialize:
                    await saver.setup()
                    await connection.execute("""
                        CREATE TABLE IF NOT EXISTS pvdiag_runs (
                            run_id TEXT PRIMARY KEY,
                            request JSONB NOT NULL,
                            contract JSONB NOT NULL,
                            max_model_calls INTEGER NOT NULL CHECK (max_model_calls >= 0),
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                    """)
                    await connection.execute("""
                        CREATE TABLE IF NOT EXISTS pvdiag_model_calls (
                            run_id TEXT NOT NULL REFERENCES pvdiag_runs(run_id),
                            ordinal INTEGER NOT NULL,
                            input_tokens BIGINT,
                            output_tokens BIGINT,
                            usage_missing BOOLEAN NOT NULL DEFAULT TRUE,
                            reported_model TEXT,
                            completed BOOLEAN NOT NULL DEFAULT FALSE,
                            PRIMARY KEY(run_id, ordinal)
                        )
                    """)
                else:
                    row = await (
                        await connection.execute(
                            "SELECT to_regclass('pvdiag_runs') AS runs, to_regclass('checkpoints') AS checkpoints"
                        )
                    ).fetchone()
                    if not row["runs"] or not row["checkpoints"]:
                        raise ValueError(
                            "persistence schema is not initialized; run the explicit init command"
                        )
                yield store
        except Error:
            raise RuntimeError("persistent_storage_unavailable") from None

    @asynccontextmanager
    async def exclusive_run(self, run_id):
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row

        key = int.from_bytes(
            hashlib.sha256(f"{self.schema}:{run_id}".encode()).digest()[:8], "big", signed=True
        )
        async with await AsyncConnection.connect(
            self._dsn, autocommit=True, row_factory=dict_row, connect_timeout=10
        ) as connection:
            row = await (
                await connection.execute("SELECT pg_try_advisory_lock(%s) AS locked", (key,))
            ).fetchone()
            if not row["locked"]:
                raise ValueError("run is already active in another process")
            try:
                yield
            finally:
                await connection.execute("SELECT pg_advisory_unlock(%s)", (key,))

    async def create_run(self, run_id, request, contract, max_model_calls):
        async with self._mutex:
            result = await self.connection.execute(
                "INSERT INTO pvdiag_runs(run_id, request, contract, max_model_calls) VALUES (%s, %s::jsonb, %s::jsonb, %s) ON CONFLICT DO NOTHING RETURNING run_id",
                (
                    run_id,
                    request.model_dump_json(),
                    json.dumps(contract, sort_keys=True),
                    max_model_calls,
                ),
            )
            if await result.fetchone() is None:
                raise ValueError("run already exists; use resume or select another run ID")

    async def is_active(self, run_id):
        try:
            async with self.exclusive_run(run_id):
                return False
        except ValueError:
            return True

    async def get_run(self, run_id):
        async with self._mutex:
            row = await (
                await self.connection.execute(
                    "SELECT request, contract, max_model_calls FROM pvdiag_runs WHERE run_id=%s",
                    (run_id,),
                )
            ).fetchone()
        if row is None:
            raise ValueError("unknown persistent run")
        return row

    def tracker(self, run_id):
        return ModelCallTracker(self, run_id)


class ModelCallTracker:
    def __init__(self, store, run_id):
        self.store, self.run_id = store, run_id

    async def reserve(self):
        async with self.store._mutex, self.store.connection.transaction():
            row = await (
                await self.store.connection.execute(
                    "SELECT max_model_calls FROM pvdiag_runs WHERE run_id=%s FOR UPDATE",
                    (self.run_id,),
                )
            ).fetchone()
            if not row:
                raise ValueError("unknown persistent run")
            count = (
                await (
                    await self.store.connection.execute(
                        "SELECT count(*) AS calls FROM pvdiag_model_calls WHERE run_id=%s",
                        (self.run_id,),
                    )
                ).fetchone()
            )["calls"]
            if count >= row["max_model_calls"]:
                raise ValueError("model_call_budget_exhausted")
            ordinal = count + 1
            await self.store.connection.execute(
                "INSERT INTO pvdiag_model_calls(run_id, ordinal) VALUES (%s, %s)",
                (self.run_id, ordinal),
            )
            return ordinal

    async def finish(self, ordinal, usage, reported_model):
        async with self.store._mutex:
            await self.store.connection.execute(
                "UPDATE pvdiag_model_calls SET input_tokens=%s, output_tokens=%s, usage_missing=%s, reported_model=%s, completed=TRUE WHERE run_id=%s AND ordinal=%s AND completed=FALSE",
                (
                    usage.get("input_tokens", 0) if usage else None,
                    usage.get("output_tokens", 0) if usage else None,
                    not bool(usage),
                    reported_model,
                    self.run_id,
                    ordinal,
                ),
            )

    async def snapshot(self):
        async with self.store._mutex:
            row = await (
                await self.store.connection.execute(
                    "SELECT count(*) AS model_calls, coalesce(sum(input_tokens),0) AS input_tokens, coalesce(sum(output_tokens),0) AS output_tokens, count(*) FILTER (WHERE usage_missing) AS usage_missing_calls FROM pvdiag_model_calls WHERE run_id=%s",
                    (self.run_id,),
                )
            ).fetchone()
            models = await (
                await self.store.connection.execute(
                    "SELECT DISTINCT reported_model FROM pvdiag_model_calls WHERE run_id=%s AND reported_model IS NOT NULL ORDER BY reported_model",
                    (self.run_id,),
                )
            ).fetchall()
        return {key: int(value) for key, value in row.items()}, [
            item["reported_model"] for item in models
        ]
