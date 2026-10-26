"""Durable, credential-free trace outbox; no network operations."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bounded(value):
    """Drop credential-bearing fields; retain a digest instead of oversized payloads."""

    def clean(item, depth=0):
        if depth > 30:
            return "[depth-limit]"
        if hasattr(item, "model_dump"):
            item = item.model_dump(mode="json")
        if isinstance(item, dict):
            hidden = {"apikey", "secretkey", "authorization", "password", "dsn", "token", "headers"}
            return {
                str(k): "[redacted]"
                if any(
                    str(k).lower().replace("_", "").replace("-", "").endswith(name)
                    for name in hidden
                )
                else clean(v, depth + 1)
                for k, v in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [clean(v, depth + 1) for v in item]
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        if isinstance(item, float) and not math.isfinite(item):
            return "[non-finite]"
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        return {"unsupported_type": type(item).__name__}

    cleaned = clean(value)
    encoded = canonical(cleaned).encode()
    return (
        cleaned
        if len(encoded) <= 131_072
        else {
            "truncated": True,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "original_bytes": len(encoded),
        }
    )


class TraceStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connection() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS objects (
                        kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                        PRIMARY KEY(kind, id)
                    )
                """)
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS deliveries (
                        target TEXT NOT NULL, operation TEXT NOT NULL, digest TEXT NOT NULL,
                        state TEXT NOT NULL, response TEXT NOT NULL,
                        PRIMARY KEY(target, operation, digest)
                    )
                """)
        except OSError:
            raise RuntimeError("trace_storage_unavailable") from None

    @contextmanager
    def connection(self):
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=5)
            with connection:
                yield connection
        except sqlite3.Error:
            raise RuntimeError("trace_storage_unavailable") from None
        finally:
            if connection:
                connection.close()

    def put(self, kind, identity, payload, *, immutable=False):
        encoded = canonical(payload)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT payload FROM objects WHERE kind=? AND id=?", (kind, identity)
            ).fetchone()
            if immutable and previous and previous[0] != encoded:
                raise ValueError("trace_object_identity_conflict")
            connection.execute(
                "INSERT INTO objects VALUES (?, ?, ?) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload",
                (kind, identity, encoded),
            )

    def get(self, kind, identity):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM objects WHERE kind=? AND id=?", (kind, identity)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, kind):
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM objects WHERE kind=? ORDER BY id", (kind,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def snapshot(self):
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT kind,id,payload FROM objects ORDER BY kind,id"
            ).fetchall()
        result = {}
        for kind, identity, payload in rows:
            result.setdefault(kind, {})[identity] = json.loads(payload)
        return result

    def delivery(self, target, operation, payload, *, state=None, response=None):
        fingerprint = digest(payload)
        with self.connection() as connection:
            if state:
                connection.execute(
                    "INSERT INTO deliveries VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(target,operation,digest) DO UPDATE SET state=excluded.state,response=excluded.response",
                    (target, operation, fingerprint, state, canonical(response)),
                )
            row = connection.execute(
                "SELECT state,response FROM deliveries WHERE target=? AND operation=? AND digest=?",
                (target, operation, fingerprint),
            ).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def delivery_report(self):
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT target,operation,digest,state FROM deliveries ORDER BY target,operation,digest"
            ).fetchall()
        return {
            "complete": sum(row[3] == "complete" for row in rows),
            "partial": [
                {"target_sha256": target, "operation": operation, "payload_sha256": fingerprint}
                for target, operation, fingerprint, state in rows
                if state == "partial"
            ],
        }
