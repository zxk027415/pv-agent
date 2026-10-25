"""A separate write client; never included in the model's read-only tool catalog."""

from contextlib import asynccontextmanager
from time import perf_counter

from fastmcp import Client

from pvdiag.agent.state import CreateWorkOrderRequest, ToolTrace, WorkOrderReceipt


class OpsGateway:
    def __init__(self, client, store):
        self.client, self.store = client, store

    @classmethod
    @asynccontextmanager
    async def connect(cls, transport, store):
        async with Client(transport, timeout=30) as client:
            tools = await client.list_tools()
            if len(tools) != 1 or tools[0].name != "create_work_order":
                raise ValueError("invalid operations tool catalog")
            annotations = tools[0].annotations
            if not annotations or annotations.read_only_hint or annotations.destructive_hint:
                raise ValueError("invalid operations tool annotations")
            if not annotations.idempotent_hint or annotations.open_world_hint:
                raise ValueError("operations require idempotent closed-world tools")
            yield cls(client, store)

    async def create(self, proposal, approval):
        started = perf_counter()
        request = None
        try:
            # The local ledger must contain this exact approved record before any request.
            if approval not in self.store.approvals() or not approval.approved:
                raise ValueError("missing human approval")
            if (approval.action_id, approval.payload_sha256) != (
                proposal.action_id,
                proposal.payload_sha256,
            ):
                raise ValueError("approval content mismatch")
            request = CreateWorkOrderRequest(
                action_id=proposal.action_id,
                payload_sha256=proposal.payload_sha256,
                approval_id=approval.approval_id,
            )
            response = await self.client.call_tool(
                "create_work_order", {"request": request.model_dump(mode="json")}
            )
            receipt = WorkOrderReceipt.model_validate(response.structured_content)
            if (
                receipt.model_dump(include={"action_id", "payload_sha256", "approval_id"})
                != request.model_dump()
            ):
                raise ValueError("work order response mismatch")
            if receipt not in self.store.work_orders():
                raise ValueError("operations receipt not recorded in the local ledger")
            return receipt, ToolTrace(
                name="create_work_order",
                node="act",
                success=True,
                duration_seconds=perf_counter() - started,
                action_id=request.action_id,
                approval_id=request.approval_id,
            )
        except Exception as exc:  # noqa: BLE001 -- redact remote and database error details
            return None, ToolTrace(
                name="create_work_order",
                node="act",
                success=False,
                duration_seconds=perf_counter() - started,
                error=type(exc).__name__,
                action_id=request.action_id if request else None,
                approval_id=request.approval_id if request else None,
            )
