"""Durable proposal/approval ledger and transactional idempotent local work orders."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pvdiag.agent.state import (
    ApprovalDecision,
    ApprovalRecord,
    CreateWorkOrderRequest,
    PendingAction,
    WorkOrderReceipt,
)


def canonical(model):
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, allow_nan=False)


class WorkOrderStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS proposals (
                    action_id TEXT PRIMARY KEY,
                    payload_sha256 TEXT NOT NULL,
                    content_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    action_id TEXT NOT NULL UNIQUE REFERENCES proposals(action_id),
                    content_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS work_orders (
                    work_order_id TEXT PRIMARY KEY,
                    action_id TEXT NOT NULL UNIQUE REFERENCES proposals(action_id),
                    approval_id TEXT NOT NULL UNIQUE REFERENCES approvals(approval_id),
                    content_json TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def register(self, proposal):
        proposal = PendingAction.model_validate(proposal.model_dump())
        content = canonical(proposal)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT content_json FROM proposals WHERE action_id=?", (proposal.action_id,)
            ).fetchone()
            if previous and previous[0] != content:
                raise ValueError("proposal identity already binds different content")
            connection.execute(
                "INSERT OR IGNORE INTO proposals VALUES (?, ?, ?)",
                (proposal.action_id, proposal.payload_sha256, content),
            )
        return proposal

    def decide(self, proposal, decision):
        decision = ApprovalDecision.model_validate(decision.model_dump())
        if (decision.action_id, decision.payload_sha256) != (
            proposal.action_id,
            proposal.payload_sha256,
        ):
            raise ValueError("approval is not bound to the pending action")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT content_json FROM proposals WHERE action_id=?", (proposal.action_id,)
            ).fetchone()
            if not row or row[0] != canonical(proposal):
                raise ValueError("pending proposal is absent or changed")
            previous = connection.execute(
                "SELECT content_json FROM approvals WHERE action_id=?", (proposal.action_id,)
            ).fetchone()
            if previous:
                record = ApprovalRecord.model_validate_json(previous[0])
                if (
                    record.model_dump(exclude={"approval_id", "decided_at"})
                    != decision.model_dump()
                ):
                    raise ValueError("an existing approval decision cannot be replaced")
                return record
            approval_id = "ap_" + hashlib.sha256(canonical(decision).encode()).hexdigest()[:32]
            record = ApprovalRecord(
                **decision.model_dump(), approval_id=approval_id, decided_at=datetime.now(UTC)
            )
            connection.execute(
                "INSERT INTO approvals VALUES (?, ?, ?)",
                (approval_id, proposal.action_id, canonical(record)),
            )
            return record

    def create_work_order(self, request):
        request = CreateWorkOrderRequest.model_validate(request.model_dump())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            proposal_row = connection.execute(
                "SELECT content_json FROM proposals WHERE action_id=?", (request.action_id,)
            ).fetchone()
            approval_row = connection.execute(
                "SELECT content_json FROM approvals WHERE approval_id=?", (request.approval_id,)
            ).fetchone()
            if not proposal_row or not approval_row:
                raise ValueError("work order requires a recorded human approval")
            proposal = PendingAction.model_validate_json(proposal_row[0])
            approval = ApprovalRecord.model_validate_json(approval_row[0])
            if not approval.approved or (
                approval.action_id != request.action_id
                or approval.payload_sha256 != request.payload_sha256
                or proposal.payload_sha256 != request.payload_sha256
            ):
                raise ValueError("approval rejected or does not bind these contents")
            previous = connection.execute(
                "SELECT content_json FROM work_orders WHERE action_id=?", (request.action_id,)
            ).fetchone()
            if previous:
                return WorkOrderReceipt.model_validate_json(previous[0])
            receipt = WorkOrderReceipt(
                **request.model_dump(),
                work_order_id="wo_" + request.action_id[4:],
                created_at=datetime.now(UTC),
            )
            connection.execute(
                "INSERT INTO work_orders VALUES (?, ?, ?, ?)",
                (receipt.work_order_id, receipt.action_id, receipt.approval_id, canonical(receipt)),
            )
            return receipt

    def work_orders(self):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT content_json FROM work_orders ORDER BY work_order_id"
            ).fetchall()
        return [WorkOrderReceipt.model_validate_json(row[0]) for row in rows]

    def approvals(self):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT content_json FROM approvals ORDER BY approval_id"
            ).fetchall()
        return [ApprovalRecord.model_validate_json(row[0]) for row in rows]
