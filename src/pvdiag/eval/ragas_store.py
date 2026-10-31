"""Separate durable answer/Judge budgets and validated-output caches for Ragas."""

from __future__ import annotations

import fcntl
import json
import re
from contextlib import contextmanager

from pvdiag.observability.store import TraceStore, canonical, digest


class RagasStore(TraceStore):
    def __init__(self, path):
        super().__init__(path)
        with self.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS ragas_jobs (
                    id TEXT PRIMARY KEY, contract TEXT NOT NULL,
                    answer_budget INTEGER NOT NULL, judge_budget INTEGER NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS ragas_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL, question_id TEXT NOT NULL, role TEXT NOT NULL,
                    cache_key TEXT NOT NULL, usage TEXT, reported_model TEXT,
                    output TEXT, completed INTEGER NOT NULL DEFAULT 0
                )
            """)

    @contextmanager
    def exclusive_job(self, job_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", job_id):
            raise ValueError("invalid Ragas job ID")
        folder = self.path.with_suffix(".locks")
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / (digest(job_id) + ".lock")).open("a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Ragas job is already active") from None
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def bind(self, job_id, contract, budgets, *, resume=False):
        if any(
            not isinstance(budgets[r], int) or not 0 <= budgets[r] <= 1000
            for r in ("answer", "judge")
        ):
            raise ValueError("invalid Ragas model budgets")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT contract,answer_budget,judge_budget FROM ragas_jobs WHERE id=?", (job_id,)
            ).fetchone()
            expected = (canonical(contract), budgets["answer"], budgets["judge"])
            if row:
                if not resume:
                    raise ValueError("Ragas job already exists; explicit resume is required")
                if row != expected:
                    raise ValueError("Ragas job context or budgets changed")
            elif resume:
                raise ValueError("unknown Ragas job")
            else:
                connection.execute(
                    "INSERT INTO ragas_jobs VALUES (?, ?, ?, ?)", (job_id, *expected)
                )

    def tracker(self, job_id, question_id, role, cache_key):
        if role not in {"answer", "judge"}:
            raise ValueError("invalid evaluation role")
        return RagasCallTracker(self, job_id, question_id, role, cache_key)

    def cached(self, job_id, cache_key):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT output FROM ragas_calls WHERE job_id=? AND cache_key=? AND output IS NOT NULL ORDER BY id DESC LIMIT 1",
                (job_id, cache_key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_output(self, job_id, cache_key, output):
        with self.connection() as connection:
            connection.execute(
                "UPDATE ragas_calls SET output=? WHERE id=(SELECT max(id) FROM ragas_calls WHERE job_id=? AND cache_key=? AND completed=1)",
                (canonical(output), job_id, cache_key),
            )

    def usage(self, job_id, *, question_id=None, role=None):
        clause, values = "job_id=?", [job_id]
        if question_id is not None:
            clause += " AND question_id=?"
            values.append(question_id)
        if role is not None:
            clause += " AND role=?"
            values.append(role)
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT usage,reported_model FROM ragas_calls WHERE " + clause, values
            ).fetchall()
        usage = {
            "model_calls": len(rows),
            "review_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "usage_missing_calls": 0,
        }
        models = set()
        for raw, model in rows:
            if raw is None:
                usage["usage_missing_calls"] += 1
            else:
                data = json.loads(raw)
                for name in ("input_tokens", "output_tokens"):
                    usage[name] += data.get(name, 0)
            if model:
                models.add(model)
        return usage, sorted(models)


class RagasCallTracker:
    def __init__(self, store, job_id, question_id, role, cache_key):
        self.store, self.job_id = store, job_id
        self.question_id, self.role, self.cache_key = question_id, role, cache_key

    async def reserve(self, purpose="diagnosis"):
        # Ragas reasoners never issue review calls; budgets here are per-role.
        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT answer_budget,judge_budget FROM ragas_jobs WHERE id=?", (self.job_id,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown Ragas job")
            count = connection.execute(
                "SELECT count(*) FROM ragas_calls WHERE job_id=? AND role=?",
                (self.job_id, self.role),
            ).fetchone()[0]
            if count >= row[0 if self.role == "answer" else 1]:
                raise ValueError("evaluation_model_call_budget_exhausted")
            cursor = connection.execute(
                "INSERT INTO ragas_calls(job_id,question_id,role,cache_key) VALUES (?, ?, ?, ?)",
                (self.job_id, self.question_id, self.role, self.cache_key),
            )
            return cursor.lastrowid

    async def finish(self, reservation, usage, reported):
        with self.store.connection() as connection:
            connection.execute(
                "UPDATE ragas_calls SET usage=?,reported_model=?,completed=1 WHERE id=? AND job_id=? AND completed=0",
                (canonical(usage) if usage else None, reported, reservation, self.job_id),
            )

    async def snapshot(self):
        return self.store.usage(self.job_id, question_id=self.question_id, role=self.role)
