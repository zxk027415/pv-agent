import asyncio
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langgraph.types import Command
from test_verification import context

from pvdiag.agent.__main__ import run_diagnosis
from pvdiag.agent.gateway import MCPGateway
from pvdiag.agent.graph import build_graph
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.state import (
    ApprovalDecision,
    CreateWorkOrderRequest,
    DiagnosisReport,
    PendingAction,
    Plan,
    ToolCall,
    WorkOrderDraft,
)
from pvdiag.eval.scorer import score_case
from pvdiag.mcp_servers.pv_analytics import create_server as analytics_server
from pvdiag.mcp_servers.pv_data import create_server as data_server
from pvdiag.mcp_servers.pv_ops import create_server as ops_server
from pvdiag.ops.gateway import OpsGateway
from pvdiag.ops.store import WorkOrderStore


def decision(proposal, *, approved=True, **updates):
    if isinstance(proposal, PendingAction):
        proposal = proposal.model_dump(mode="json")
    return {
        "action_id": proposal["action_id"],
        "payload_sha256": proposal["payload_sha256"],
        "approved": approved,
        "reviewer": "unit-test-reviewer",
        "reason": "temporary fixture approval",
    } | updates


@pytest.fixture
def proposal(tmp_path):
    _, request, scope, evidence, report = context(tmp_path, "F3")
    draft = WorkOrderDraft(
        request=request,
        report=report,
        observation_source_sha256=scope.source_sha256,
        harness_sha256=hashlib.sha256(b"fixture").hexdigest(),
        ops_policy_sha256=hashlib.sha256(b"fixture ops policy").hexdigest(),
        evidence=evidence,
    )
    return PendingAction.propose("fixture-run", draft)


def test_default_diagnosis_does_not_mount_ops_or_create_a_database(tmp_path):
    store, request, _, _, _ = context(tmp_path, "F3")
    result = asyncio.run(run_diagnosis(store, request, RuleReasoner()))
    assert result.report.status == "fault"
    assert not result.ops_enabled and result.pending_action is None
    assert not any(t.name == "create_work_order" for t in result.trace)
    assert not list(tmp_path.rglob("*.sqlite3"))
    with pytest.raises(ValueError):
        ToolCall(name="create_work_order", arguments={})


def test_graph_interrupt_precedes_every_write_and_records_approved_execution(tmp_path):
    store, request, _, _, _ = context(tmp_path, "F3")
    ledger = WorkOrderStore(tmp_path / "work_orders.sqlite3")

    async def verify():
        async with (
            MCPGateway.connect(data_server(store), analytics_server(store)) as reader,
            OpsGateway.connect(ops_server(ledger), ledger) as writer,
        ):
            assert "create_work_order" not in reader.tools
            graph = build_graph(
                reader, RuleReasoner(), ops_gateway=writer, verification_mode="domain"
            )
            config = {"configurable": {"thread_id": "approval-test"}}
            paused = await graph.ainvoke({"request": request}, config=config)
            assert paused["__interrupt__"] and (await graph.aget_state(config)).next == (
                "approval",
            )
            assert ledger.work_orders() == ledger.approvals() == []
            displayed = paused["__interrupt__"][0].value["proposal"]
            assert displayed["payload"]["report"]["status"] == "fault"
            finished = await graph.ainvoke(Command(resume=decision(displayed)), config=config)
            result = finished["result"]
            assert result.action_outcome.status == "created" and result.approval.approved
            assert result.ops_enabled and len(result.ops_policy_sha256) == 64
            assert result.trace[-1].name == "create_work_order" and result.trace[-1].node == "act"
            assert len(ledger.work_orders()) == len(ledger.approvals()) == 1
            assert ledger.work_orders()[0] == result.action_outcome.receipt
            truth = {
                "scenario_id": "controlled-ops-fixture",
                "system_id": request.system_id,
                "query_window": [request.start.isoformat(), request.end.isoformat()],
                "faults": [
                    {
                        "type": "F3",
                        "device_id": "inv01",
                        "start": "2019-01-04T16:00:00Z",
                        "end": "2019-01-04T18:00:00Z",
                    }
                ],
            }
            assert score_case(truth, result)["unapproved_write_executions"] == 0
            for mutate in (
                lambda run: setattr(run, "ops_enabled", False),
                lambda run: setattr(run, "ops_policy_sha256", "0" * 64),
                lambda run: setattr(run.approval, "approved", False),
                lambda run: setattr(run.trace[-1], "node", "investigate"),
                lambda run: setattr(run.trace[-1], "approval_id", "ap_" + "0" * 32),
                lambda run: setattr(run.action_outcome.receipt, "payload_sha256", "0" * 64),
                lambda run: setattr(run.report, "summary", "已变更的报告。"),
                lambda run: setattr(run.pending_action.payload, "harness_sha256", "0" * 64),
                lambda run: next(
                    e for e in run.evidence.values() if e["tool"] == "list_devices"
                ).update(source_sha256="0" * 64),
            ):
                tampered = result.model_copy(deep=True)
                mutate(tampered)
                assert score_case(truth, tampered)["unapproved_write_executions"] == 1
            assert (await graph.aget_state(config)).next == ()
            # A completed checkpoint replay cannot produce a second order.
            await graph.ainvoke(None, config=config)
            assert len(ledger.work_orders()) == 1

    asyncio.run(verify())


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({"approved": False}, "rejected"),
        ({"approved": "true"}, "invalid_approval"),
        ({"approved": "false"}, "invalid_approval"),
        ({"approved": 1}, "invalid_approval"),
        ({"payload_sha256": "0" * 64}, "invalid_approval"),
        ({"action_id": "act_" + "0" * 32}, "invalid_approval"),
        ({"reviewer": " "}, "invalid_approval"),
    ],
)
def test_rejected_invalid_and_cross_action_decisions_make_no_work_order(
    tmp_path, updates, expected
):
    store, request, _, _, _ = context(tmp_path, "F3")
    ledger = WorkOrderStore(tmp_path / "work_orders.sqlite3")

    async def respond(payload):
        assert ledger.work_orders() == ledger.approvals() == []
        return decision(payload["proposal"], **updates)

    result = asyncio.run(
        run_diagnosis(store, request, RuleReasoner(), ops_store=ledger, approval_handler=respond)
    )
    assert result.action_outcome.status == expected
    assert result.report.status == "fault"  # The diagnosis is independent of action rejection.
    assert ledger.work_orders() == []
    assert not any(t.name == "create_work_order" for t in result.trace)
    if expected == "rejected":
        assert len(ledger.approvals()) == 1 and not ledger.approvals()[0].approved
    else:
        assert ledger.approvals() == [] and result.approval is None


def test_inconclusive_and_domain_unsupported_reports_never_request_approval(tmp_path):
    store, request, _, _, _ = context(tmp_path, "F3")
    ledger = WorkOrderStore(tmp_path / "work_orders.sqlite3")

    class Inconclusive(RuleReasoner):
        async def plan(self, state, tools, prompt):
            return Plan(calls=[])

        async def hypothesize(self, state, prompt):
            return DiagnosisReport(status="inconclusive", summary="无法判断。")

    class WrongType(RuleReasoner):
        async def hypothesize(self, state, prompt):
            result = await super().hypothesize(state, prompt)
            for finding in result.findings:
                finding.fault_type = "F4"
            return result

    async def forbidden(payload):
        pytest.fail("unsupported reports cannot request approval")

    for reasoner, expected in [(Inconclusive(), "not_needed"), (WrongType(), "blocked")]:
        result = asyncio.run(
            run_diagnosis(store, request, reasoner, ops_store=ledger, approval_handler=forbidden)
        )
        assert result.action_outcome.status == expected
    assert ledger.work_orders() == ledger.approvals() == []


def test_report_changed_while_paused_cannot_execute_the_old_approval(tmp_path):
    store, request, _, _, _ = context(tmp_path, "F3")
    ledger = WorkOrderStore(tmp_path / "work_orders.sqlite3")

    async def verify():
        async with (
            MCPGateway.connect(data_server(store), analytics_server(store)) as reader,
            OpsGateway.connect(ops_server(ledger), ledger) as writer,
        ):
            graph = build_graph(reader, RuleReasoner(), ops_gateway=writer)
            config = {"configurable": {"thread_id": "changed-report"}}
            paused = await graph.ainvoke({"request": request}, config=config)
            response = decision(paused["__interrupt__"][0].value["proposal"])
            changed = paused["verdict"].model_copy(deep=True)
            changed.summary = "报告描述已变更。"
            await graph.aupdate_state(config, {"verdict": changed})
            result = (await graph.ainvoke(Command(resume=response), config=config))["result"]
            assert result.action_outcome.status == "blocked"
            assert result.action_outcome.error == "approved_payload_changed"
            assert ledger.work_orders() == []

    asyncio.run(verify())


def test_ledger_rejects_missing_mismatched_and_replaced_approvals(proposal, tmp_path):
    ledger = WorkOrderStore(tmp_path / "ledger.sqlite3")
    ledger.register(proposal)
    request = CreateWorkOrderRequest(
        action_id=proposal.action_id,
        payload_sha256=proposal.payload_sha256,
        approval_id="ap_" + "0" * 32,
    )
    with pytest.raises(ValueError, match="approval"):
        ledger.create_work_order(request)
    approved = ledger.decide(proposal, ApprovalDecision(**decision(proposal)))
    assert ledger.decide(proposal, ApprovalDecision(**decision(proposal))) == approved
    with pytest.raises(ValueError, match="replaced"):
        ledger.decide(proposal, ApprovalDecision(**decision(proposal, approved=False)))
    wrong = request.model_copy(
        update={"approval_id": approved.approval_id, "payload_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="bind"):
        ledger.create_work_order(wrong)
    assert ledger.work_orders() == []

    async def forbidden(*args, **kwargs):
        pytest.fail("missing approval must be rejected before contacting the write service")

    async def verify_gateway():
        gateway = OpsGateway(SimpleNamespace(call_tool=forbidden), ledger)
        receipt, trace = await gateway.create(proposal, None)
        assert receipt is None and not trace.success and trace.error == "ValueError"
        assert trace.approval_id is None

    asyncio.run(verify_gateway())


def test_concurrent_retries_and_reopened_ledger_return_exactly_one_order(proposal, tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = WorkOrderStore(path)
    ledger.register(proposal)
    approval = ledger.decide(proposal, ApprovalDecision(**decision(proposal)))
    request = CreateWorkOrderRequest(
        action_id=proposal.action_id,
        payload_sha256=proposal.payload_sha256,
        approval_id=approval.approval_id,
    )
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(
            pool.map(lambda _: WorkOrderStore(path).create_work_order(request), range(12))
        )
    assert all(r == receipts[0] for r in receipts)
    assert WorkOrderStore(path).work_orders() == [receipts[0]]
    assert WorkOrderStore(path).approvals() == [approval]


def test_real_stdio_server_cannot_self_approve_and_enforces_recorded_decision(proposal, tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = WorkOrderStore(path)
    ledger.register(proposal)
    request = CreateWorkOrderRequest(
        action_id=proposal.action_id,
        payload_sha256=proposal.payload_sha256,
        approval_id="ap_" + "0" * 32,
    )

    async def verify():
        transport = StdioTransport(
            sys.executable, ["-m", "pvdiag.mcp_servers.pv_ops", "--db", str(path)]
        )
        async with Client(transport) as client:
            tools = await client.list_tools()
            assert [t.name for t in tools] == ["create_work_order"]
            assert not tools[0].annotations.read_only_hint
            denied = await client.call_tool(
                "create_work_order", {"request": request.model_dump()}, raise_on_error=False
            )
            assert denied.is_error and ledger.work_orders() == []
            approval = ledger.decide(proposal, ApprovalDecision(**decision(proposal)))
            request.approval_id = approval.approval_id
            first = await client.call_tool("create_work_order", {"request": request.model_dump()})
            second = await client.call_tool("create_work_order", {"request": request.model_dump()})
            assert first.structured_content == second.structured_content
            assert len(ledger.work_orders()) == 1

    asyncio.run(verify())


def test_response_loss_then_retry_cannot_duplicate_a_committed_work_order(proposal, tmp_path):
    ledger = WorkOrderStore(tmp_path / "ledger.sqlite3")
    ledger.register(proposal)
    approval = ledger.decide(proposal, ApprovalDecision(**decision(proposal)))
    attempts = []

    async def transport(name, arguments):
        receipt = ledger.create_work_order(CreateWorkOrderRequest(**arguments["request"]))
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("private transport error after commit")
        return SimpleNamespace(structured_content=receipt.model_dump(mode="json"))

    async def verify():
        gateway = OpsGateway(SimpleNamespace(call_tool=transport), ledger)
        result, trace = await gateway.create(proposal, approval)
        assert result is None and not trace.success and trace.error == "RuntimeError"
        assert len(ledger.work_orders()) == 1
        result, trace = await gateway.create(proposal, approval)
        assert trace.success and ledger.work_orders() == [result]

    asyncio.run(verify())


def test_mutated_proposal_identity_and_missing_handler_are_rejected(proposal, tmp_path):
    altered = proposal.model_dump()
    altered["payload"]["report"]["summary"] = "未重新审批的修改。"
    with pytest.raises(ValueError, match="changed"):
        PendingAction.model_validate(altered)
    (tmp_path / "other").mkdir()
    store, request, _, _, _ = context(tmp_path / "other", "F3")
    with pytest.raises(ValueError, match="approval handler"):
        asyncio.run(
            run_diagnosis(
                store,
                request,
                RuleReasoner(),
                ops_store=WorkOrderStore(tmp_path / "ledger.sqlite3"),
            )
        )


def test_one_runs_approval_cannot_authorize_another_run_with_identical_payload(proposal, tmp_path):
    ledger = WorkOrderStore(tmp_path / "ledger.sqlite3")
    ledger.register(proposal)
    approval = ledger.decide(proposal, ApprovalDecision(**decision(proposal)))
    other = PendingAction.propose("a-different-run", proposal.payload)
    ledger.register(other)
    request = CreateWorkOrderRequest(
        action_id=other.action_id,
        payload_sha256=other.payload_sha256,
        approval_id=approval.approval_id,
    )
    with pytest.raises(ValueError, match="bind"):
        ledger.create_work_order(request)
    assert ledger.work_orders() == []


def test_cli_rejects_unenabled_database_and_treats_eof_as_rejection(tmp_path, monkeypatch, capsys):
    from pvdiag.agent.__main__ import main

    context(tmp_path, "F3")
    # context stores the two input files at these explicit paths.
    arguments = [
        "2107",
        "--observations",
        str(tmp_path / "observations.parquet"),
        "--metadata",
        str(tmp_path / "metadata.json"),
        "--devices",
        "inv01",
        "inv02",
        "inv03",
        "--start",
        "2019-01-01",
        "--end",
        "2019-02-12",
        "--ops-db",
        str(tmp_path / "cli.sqlite3"),
        "--output",
        str(tmp_path / "run.json"),
    ]
    with pytest.raises(SystemExit):
        main(arguments)
    assert not (tmp_path / "cli.sqlite3").exists()

    def eof(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert main([*arguments, "--enable-ops"]) == 0
    result = json.loads((tmp_path / "run.json").read_text())
    assert result["action_outcome"]["status"] == "rejected"
    assert WorkOrderStore(tmp_path / "cli.sqlite3").work_orders() == []
    assert "work_order_approval" in capsys.readouterr().out
