"""Append-only pgvector cases/lessons and immutable revision-bounded read views."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from uuid import uuid4

from pvdiag.memory.models import MemoryCase, MemoryLesson, MemorySnapshot, scored_development_case
from pvdiag.observability.store import canonical, digest
from pvdiag.observability.tracing import observe


def vector_text(values, dimensions):
    values = [float(v) for v in values]
    norm = math.sqrt(sum(v * v for v in values))
    if (
        len(values) != dimensions
        or not all(math.isfinite(v) for v in values)
        or not math.isfinite(norm)
        or norm <= 0
    ):
        raise ValueError(
            "memory embeddings must be finite nonzero vectors with the configured dimensions"
        )
    return "[" + ",".join(format(v / norm, ".17g") for v in values) + "]"


class MemoryStore(ABC):
    @abstractmethod
    async def add_case(self, manifest, scenario_id, result, *, review_bundle=None): ...

    @abstractmethod
    async def add_lesson(self, lesson): ...

    @abstractmethod
    async def search_cases(self, text, request, *, snapshot=None, limit=3): ...


class PgVectorStore(MemoryStore):
    def __init__(self, connection, embedder, config, schema, vector_schema, *, read_only):
        self.connection, self.embedder, self.config = connection, embedder, config
        self.schema, self.vector_schema, self.read_only = schema, vector_schema, read_only
        self.mutex = asyncio.Lock()
        self.lock_key = int.from_bytes(
            hashlib.sha256(config["store_id"].encode()).digest()[:8], "big", signed=True
        )

    @classmethod
    @asynccontextmanager
    async def open(cls, dsn, embedder, *, schema="pvdiag_memory", initialize=False, read_only=True):
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema)
            or schema in {"public", "information_schema"}
            or schema.startswith("pg_")
        ):
            raise ValueError("use a dedicated memory schema")
        if initialize and read_only:
            raise ValueError("memory initialization requires explicit write mode")
        if type(embedder.dimensions) is not int or not 1 <= embedder.dimensions <= 2000:
            raise ValueError("invalid memory embedding dimensions")
        from psycopg import AsyncConnection, Error, sql
        from psycopg.rows import dict_row

        try:
            async with await AsyncConnection.connect(
                dsn, autocommit=True, row_factory=dict_row, connect_timeout=10
            ) as connection:
                if initialize:
                    await connection.execute(
                        "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"
                    )
                    await connection.execute(
                        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
                    )
                row = await (
                    await connection.execute(
                        "SELECT n.nspname AS schema FROM pg_catalog.pg_extension e JOIN pg_catalog.pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='vector'"
                    )
                ).fetchone()
                if row is None:
                    raise ValueError(
                        "pgvector is not enabled; initialize the memory store explicitly"
                    )
                vector_schema = row["schema"]
                await connection.execute(
                    sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(schema))
                )
                if initialize:
                    await connection.execute(
                        "CREATE TABLE IF NOT EXISTS memory_config (id INT PRIMARY KEY CHECK(id=1),store_id TEXT NOT NULL,embedder JSONB NOT NULL,dimensions INT NOT NULL)"
                    )
                    await connection.execute(
                        sql.SQL(
                            "CREATE TABLE IF NOT EXISTS memory_entries (revision BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,id TEXT UNIQUE NOT NULL,kind TEXT NOT NULL CHECK(kind IN ('case','lesson')),body JSONB NOT NULL,embedding {}.vector({}) NOT NULL)"
                        ).format(sql.Identifier(vector_schema), sql.Literal(embedder.dimensions))
                    )
                    await connection.execute(
                        "INSERT INTO memory_config VALUES (1,%s,%s::jsonb,%s) ON CONFLICT(id) DO NOTHING",
                        (uuid4().hex, canonical(embedder.identity), embedder.dimensions),
                    )
                exists = await (
                    await connection.execute(
                        "SELECT to_regclass(%s) AS config,to_regclass(%s) AS entries",
                        (schema + ".memory_config", schema + ".memory_entries"),
                    )
                ).fetchone()
                if not exists["config"] or not exists["entries"]:
                    raise ValueError("memory schema is not initialized")
                config = await (
                    await connection.execute(
                        "SELECT store_id,embedder,dimensions FROM memory_config WHERE id=1"
                    )
                ).fetchone()
                if (
                    config is None
                    or config["embedder"] != embedder.identity
                    or config["dimensions"] != embedder.dimensions
                ):
                    raise ValueError("memory embedding identity or dimensions changed")
                if read_only:
                    await connection.execute("SET default_transaction_read_only=on")
                yield cls(connection, embedder, config, schema, vector_schema, read_only=read_only)
        except Error:
            raise RuntimeError("memory_storage_unavailable") from None

    def _writable(self):
        if self.read_only:
            raise ValueError("memory is read-only")

    async def add_case(self, manifest, scenario_id, result, *, review_bundle=None):
        self._writable()
        case = scored_development_case(manifest, scenario_id, result, review_bundle=review_bundle)
        return await self._insert("case", case)

    async def add_lesson(self, lesson):
        self._writable()
        lesson = MemoryLesson.model_validate(
            lesson.model_dump(mode="json") if isinstance(lesson, MemoryLesson) else lesson
        )
        return await self._insert("lesson", lesson)

    async def _insert(self, kind, record):
        from psycopg import sql

        text = vector_text(
            self.embedder.encode(record.search_text, query=False), self.embedder.dimensions
        )
        async with self.mutex, self.connection.transaction():
            # Serialize allocation and commit order, so a later commit cannot
            # appear below an already frozen revision watermark.
            await self.connection.execute("SELECT pg_advisory_xact_lock(%s)", (self.lock_key,))
            if kind == "lesson":
                rows = await (
                    await self.connection.execute(
                        "SELECT id FROM memory_entries WHERE kind='case' AND id=ANY(%s)",
                        (record.source_case_ids,),
                    )
                ).fetchall()
                if {r["id"] for r in rows} != set(record.source_case_ids):
                    raise ValueError("lessons require existing development support cases")
            await self.connection.execute(
                sql.SQL(
                    "INSERT INTO memory_entries(id,kind,body,embedding) VALUES (%s,%s,%s::jsonb,%s::{}.vector) ON CONFLICT(id) DO NOTHING"
                ).format(sql.Identifier(self.vector_schema)),
                (record.identity, kind, canonical(record.model_dump(mode="json")), text),
            )
            row = await (
                await self.connection.execute(
                    "SELECT revision,body FROM memory_entries WHERE id=%s", (record.identity,)
                )
            ).fetchone()
            if row["body"] != record.model_dump(mode="json"):
                raise ValueError("memory identity conflict")
            return {"id": record.identity, "revision": row["revision"], "kind": kind}

    async def _snapshot(self, revision=None):
        config = await (
            await self.connection.execute(
                "SELECT store_id,embedder,dimensions FROM memory_config WHERE id=1"
            )
        ).fetchone()
        if config != self.config:
            raise ValueError("memory configuration changed")
        maximum = (
            await (
                await self.connection.execute(
                    "SELECT coalesce(max(revision),0) AS revision FROM memory_entries"
                )
            ).fetchone()
        )["revision"]
        if revision is None:
            revision = maximum
        if type(revision) is not int or not 0 <= revision <= maximum:
            raise ValueError("memory revision is unavailable")
        rows = await (
            await self.connection.execute(
                "SELECT revision,id,kind,body,embedding::text AS vector FROM memory_entries WHERE revision<=%s ORDER BY revision",
                (revision,),
            )
        ).fetchall()
        for row in rows:
            model = MemoryCase if row["kind"] == "case" else MemoryLesson
            if model.model_validate(row["body"]).identity != row["id"]:
                raise ValueError("memory content identity mismatch")
        case_ids = {row["id"] for row in rows if row["kind"] == "case"}
        if any(set(row["body"]["source_case_ids"]) - case_ids for row in rows if row["kind"] == "lesson"):
            raise ValueError("memory lesson support cases are missing")
        backend = await (await self.connection.execute("SELECT current_setting('server_version_num') AS postgres,e.extversion AS pgvector FROM pg_catalog.pg_extension e WHERE e.extname='vector'")).fetchone()
        return MemorySnapshot(
            store_id=config["store_id"],
            revision=revision,
            content_sha256=digest(rows),
            embedder=config["embedder"],
            backend=backend,
            dimensions=config["dimensions"],
            cases=sum(r["kind"] == "case" for r in rows),
            lessons=sum(r["kind"] == "lesson" for r in rows),
        )

    async def view(self, *, revision=None):
        async with self.mutex, self.connection.transaction():
            await self.connection.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
            )
            snapshot = await self._snapshot(revision)
        return MemoryView(self, snapshot)

    async def search_cases(self, text, request, *, snapshot=None, limit=3):
        snapshot = snapshot or (await self.view()).snapshot
        result = await self._recall(text, request, snapshot, limit=limit)
        return result["cases"]

    async def _recall(self, text, request, snapshot, *, limit=3):
        if not 1 <= limit <= 3 or not 1 <= len(text) <= 6000:
            raise ValueError("invalid memory recall budget")
        from psycopg import sql

        query = vector_text(self.embedder.encode(text, query=True), self.embedder.dimensions)
        schema = sql.Identifier(self.vector_schema)
        async with self.mutex, self.connection.transaction():
            await self.connection.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
            )
            if (await self._snapshot(snapshot.revision)) != snapshot:
                raise ValueError("frozen memory snapshot changed")
            exclusion = "NOT ((body->'request'->>'system_id')::int=%s AND (body->'request'->>'start')::timestamptz<%s AND (body->'request'->>'end')::timestamptz>%s)"
            cases = await (
                await self.connection.execute(
                    sql.SQL(
                        "SELECT body,1-(embedding OPERATOR({}.<=>) %s::{}.vector) AS similarity FROM memory_entries WHERE kind='case' AND revision<=%s AND "
                        + exclusion
                        + " ORDER BY embedding OPERATOR({}.<=>) %s::{}.vector,id LIMIT %s"
                    ).format(schema, schema, schema, schema),
                    (
                        query,
                        snapshot.revision,
                        request.system_id,
                        request.end,
                        request.start,
                        query,
                        limit,
                    ),
                )
            ).fetchall()
            lessons = await (
                await self.connection.execute(
                    sql.SQL(
                        "SELECT e.body,1-(e.embedding OPERATOR({}.<=>) %s::{}.vector) AS similarity FROM memory_entries e WHERE e.kind='lesson' AND e.body->>'outcome'='accepted' AND e.revision<=%s AND NOT EXISTS (SELECT 1 FROM memory_entries c WHERE c.kind='case' AND c.id IN (SELECT jsonb_array_elements_text(e.body->'source_case_ids')) AND NOT ("
                        + exclusion.replace("body->", "c.body->")
                        + ")) ORDER BY e.embedding OPERATOR({}.<=>) %s::{}.vector,e.id LIMIT %s"
                    ).format(schema, schema, schema, schema),
                    (
                        query,
                        snapshot.revision,
                        request.system_id,
                        request.end,
                        request.start,
                        query,
                        limit,
                    ),
                )
            ).fetchall()
        result = {
            "snapshot": {"sha256": snapshot.sha256, "revision": snapshot.revision, "cases": snapshot.cases, "lessons": snapshot.lessons},
            "reference_policy": "historical_predictions_and_reviewed_notes_not_current_measurements",
            "same_system_overlapping_windows_excluded": True,
            "cases": [],
            "lessons": [],
            "truncated": False,
        }
        used = 0
        for name, rows, model in (("cases", cases, MemoryCase), ("lessons", lessons, MemoryLesson)):
            for row in rows:
                item = model.model_validate(row["body"]).reference() | {
                    "cosine_similarity_not_probability": float(row["similarity"])
                }
                size = len(canonical(item))
                if used + size > 6000:
                    result["truncated"] = True
                    continue
                result[name].append(item)
                used += size
        return result


class MemoryView:
    def __init__(self, store, snapshot):
        self.store, self.snapshot = store, snapshot

    async def recall(self, request, evidence):
        parts = [request.question]
        for item in evidence.values():
            if item["tool"] == "data_quality":
                parts.extend(
                    f"{s['metric']} coverage={s['coverage']} missing={s['missing_bins']} zero={s['zero_bins']}"
                    for s in item["result"]["series"]
                )
        text = "\n".join(parts)[:6000]
        with observe(
            "retriever",
            "memory.recall",
            {"request": request, "query": text},
            {"snapshot_sha256": self.snapshot.sha256},
        ) as observation:
            result = await self.store._recall(text, request, self.snapshot)
            observation.update(output=result)
            return result
