import asyncio
import json
import os
import subprocess
import sys
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from test_verification import context

from pvdiag.agent.llm import LLMReasoner
from pvdiag.agent.persistence import PostgresStore, checkpoint_serializer
from pvdiag.agent.rules import RuleReasoner
from pvdiag.agent.runtime import open_runtime
from pvdiag.agent.state import DiagRequest, Plan, Scope
from pvdiag.ops.store import WorkOrderStore


@pytest.fixture
def pg():
    dsn = os.getenv("PVDIAG_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set an explicitly isolated PVDIAG_TEST_POSTGRES_DSN for real PostgreSQL tests")
    return dsn, "pvdiag_test_" + uuid4().hex


def test_checkpoint_serializer_restores_explicit_types_without_pickle():
    serializer = checkpoint_serializer()
    request = DiagRequest(
        system_id=2107, device_ids=["inv01"], start="2019-01-01", end="2019-01-02"
    )
    restored = serializer.loads_typed(
        serializer.dumps_typed({"request": request, "plan": Plan(calls=[])})
    )
    assert isinstance(restored["request"], DiagRequest)
    assert isinstance(restored["plan"], Plan)
    assert restored["request"] == request
    assert serializer.pickle_fallback is False


def test_schema_validation_precedes_connection_attempts():
    async def verify():
        for schema in ("public", "information_schema", "pg_catalog", "not a schema"):
            with pytest.raises(ValueError, match="dedicated"):
                async with PostgresStore.open("must-not-connect", schema=schema, initialize=True):
                    pytest.fail("system/invalid schemas cannot initialize storage")

    asyncio.run(verify())


def test_schema_initialization_is_explicit_and_run_ids_cannot_be_overwritten(pg, tmp_path):
    dsn, schema = pg
    _, request, _, _, _ = context(tmp_path, "F3")

    async def verify():
        with pytest.raises(ValueError, match="not initialized"):
            async with PostgresStore.open(dsn, schema=schema):
                pytest.fail("opening an absent schema must not initialize it")
        async with PostgresStore.open(dsn, schema=schema, initialize=True) as persistence:
            await persistence.create_run("run-1", request, {"fixture": True}, 1)
            with pytest.raises(ValueError, match="already exists"):
                await persistence.create_run("run-1", request, {}, 6)
            assert (await persistence.get_run("run-1"))["max_model_calls"] == 1

    asyncio.run(verify())


def test_postgres_lock_excludes_a_second_controller_and_releases_on_disconnect(pg):
    dsn, schema = pg

    async def verify():
        async with (
            PostgresStore.open(dsn, schema=schema, initialize=True) as first,
            PostgresStore.open(dsn, schema=schema) as second,
        ):
            async with first.exclusive_run("run-1"):
                assert await second.is_active("run-1")
                with pytest.raises(ValueError, match="already active"):
                    async with second.exclusive_run("run-1"):
                        pytest.fail("concurrent controllers must not own the same run")
                async with second.exclusive_run("run-2"):
                    pass
            async with second.exclusive_run("run-1"):
                pass
            assert not await first.is_active("run-1")

    asyncio.run(verify())


def test_approval_survives_process_exit_and_completed_resume_is_idempotent(pg, tmp_path):
    dsn, schema = pg
    context(tmp_path, "F3")

    async def initialize():
        async with PostgresStore.open(dsn, schema=schema, initialize=True):
            pass

    asyncio.run(initialize())
    environment = os.environ.copy()
    environment["PVDIAG_TEST_SESSION_DSN"] = dsn
    common = [
        "--dsn-env",
        "PVDIAG_TEST_SESSION_DSN",
        "--schema",
        schema,
        "--observations",
        str(tmp_path / "observations.parquet"),
        "--metadata",
        str(tmp_path / "metadata.json"),
        "--mode",
        "b0",
        "--verification",
        "domain",
        "--enable-ops",
        "--ops-db",
        str(tmp_path / "orders.sqlite3"),
    ]

    def invoke(arguments, *, reply=None):
        completed = subprocess.run(
            [sys.executable, "-m", "pvdiag.agent.sessions", *arguments],
            env=environment,
            input=reply,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    invoke(
        [
            "start",
            "restart-approval",
            "2107",
            *common,
            "--devices",
            "inv01",
            "inv02",
            "inv03",
            "--start",
            "2019-01-01",
            "--end",
            "2019-02-12",
            "--output",
            str(tmp_path / "paused.json"),
        ]
    )
    paused = json.loads((tmp_path / "paused.json").read_text())
    assert paused["status"] == "awaiting_approval"
    assert WorkOrderStore(tmp_path / "orders.sqlite3").work_orders() == []
    invoke(
        [
            "show",
            "restart-approval",
            "--dsn-env",
            "PVDIAG_TEST_SESSION_DSN",
            "--schema",
            schema,
            "--output",
            str(tmp_path / "pending-view.json"),
        ]
    )
    assert json.loads((tmp_path / "pending-view.json").read_text()) == paused
    # This is a second, newly started interpreter; approval is a controlled test input.
    invoke(
        [
            "resume",
            "restart-approval",
            *common,
            "--interactive",
            "--output",
            str(tmp_path / "finished.json"),
        ],
        reply="approve\n",
    )
    finished = json.loads((tmp_path / "finished.json").read_text())
    assert (
        finished["status"] == "completed"
        and finished["result"]["action_outcome"]["status"] == "created"
    )
    assert len(WorkOrderStore(tmp_path / "orders.sqlite3").work_orders()) == 1
    invoke(["resume", "restart-approval", *common, "--output", str(tmp_path / "again.json")])
    assert json.loads((tmp_path / "again.json").read_text()) == finished
    invoke(
        [
            "show",
            "restart-approval",
            "--dsn-env",
            "PVDIAG_TEST_SESSION_DSN",
            "--schema",
            schema,
            "--output",
            str(tmp_path / "shown.json"),
        ]
    )
    assert json.loads((tmp_path / "shown.json").read_text()) == finished
    assert len(WorkOrderStore(tmp_path / "orders.sqlite3").work_orders()) == 1


def test_changed_resume_context_is_rejected_before_advancing_the_graph(pg, tmp_path):
    dsn, schema = pg
    store, request, _, _, _ = context(tmp_path, "F3")
    ledger = WorkOrderStore(tmp_path / "orders.sqlite3")

    async def verify():
        async with PostgresStore.open(dsn, schema=schema, initialize=True) as persistence:
            async with open_runtime(
                store,
                RuleReasoner(),
                persistence=persistence,
                ops_store=ledger,
                verification_mode="domain",
            ) as runtime:
                assert (await runtime.start("bound-context", request)).status == "awaiting_approval"
            async with open_runtime(
                store,
                RuleReasoner(),
                persistence=persistence,
                ops_store=ledger,
                verification_mode="structural",
            ) as runtime:

                async def forbidden(*args, **kwargs):
                    pytest.fail("changed context must fail before graph/model/tool execution")

                runtime.graph.ainvoke = forbidden
                with pytest.raises(ValueError, match="context changed"):
                    await runtime.resume("bound-context")
            async with open_runtime(
                store,
                RuleReasoner(),
                persistence=persistence,
                ops_store=ledger,
                verification_mode="domain",
            ) as runtime:
                runtime.contract["observation_source_sha256"] = "0" * 64
                with pytest.raises(ValueError, match="context changed"):
                    await runtime.resume("bound-context")
        assert ledger.work_orders() == ledger.approvals() == []

    asyncio.run(verify())


def test_crash_after_call_reservation_does_not_reset_budget_in_a_new_process(pg, tmp_path):
    dsn, schema = pg
    _, request, _, _, _ = context(tmp_path, "F3")

    async def initialize():
        async with PostgresStore.open(dsn, schema=schema, initialize=True) as persistence:
            await persistence.create_run("crashed-model", request, {}, 1)

    asyncio.run(initialize())
    environment = os.environ.copy()
    environment["PVDIAG_TEST_SESSION_DSN"] = dsn
    environment["PVDIAG_TEST_SCHEMA"] = schema
    code = """
import asyncio, os
from pvdiag.agent.persistence import PostgresStore
async def main():
    async with PostgresStore.open(os.environ['PVDIAG_TEST_SESSION_DSN'],schema=os.environ['PVDIAG_TEST_SCHEMA']) as persistence:
        async with persistence.exclusive_run('crashed-model'):
            await persistence.tracker('crashed-model').reserve()
            os._exit(17)
asyncio.run(main())
"""
    child = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, timeout=30, check=False
    )
    assert child.returncode == 17

    async def verify():
        async with (
            PostgresStore.open(dsn, schema=schema) as persistence,
            persistence.exclusive_run("crashed-model"),
        ):
            tracker = persistence.tracker("crashed-model")
            usage, _ = await tracker.snapshot()
            assert usage["model_calls"] == usage["usage_missing_calls"] == 1
            with pytest.raises(ValueError, match="budget_exhausted"):
                await tracker.reserve()

    asyncio.run(verify())


def test_ablation_checkpoint_preserves_summary_and_rejects_level_change(pg, tmp_path, monkeypatch):
    from test_ablation import inconclusive, offline_model

    dsn, schema = pg
    store, request, _, _, _ = context(tmp_path, "F3")
    attempts = offline_model(monkeypatch, inconclusive)

    async def verify():
        async with (
            PostgresStore.open(dsn, schema=schema, initialize=True) as persistence,
            open_runtime(
                store, LLMReasoner("offline"), persistence=persistence, ablation="A0"
            ) as runtime,
        ):
            first = await runtime.start("ablation-bound", request)
            original_contract = runtime.contract
            assert first.status == "completed"
            assert first.result.model_usage["model_calls"] == 1
            assert len(first.result.ablation["summary_evidence_ids"]) == 8
        # Reconstruct both the model and runtime after closing the PG connection.
        async with PostgresStore.open(dsn, schema=schema) as persistence:
            async with open_runtime(
                store, LLMReasoner("offline"), persistence=persistence, ablation="A1"
            ) as runtime:
                assert {
                    k for k in runtime.contract if runtime.contract[k] != original_contract[k]
                } == {"ablation"}
                with pytest.raises(ValueError, match="context changed"):
                    await runtime.resume("ablation-bound")
            fresh = LLMReasoner("offline")
            async with open_runtime(
                store, fresh, persistence=persistence, ablation="A0"
            ) as runtime:
                restored = await runtime.resume("ablation-bound")
                assert restored == first
                assert fresh.usage["model_calls"] == 1
                assert len(attempts) == 1

    asyncio.run(verify())


def test_llm_accounting_is_committed_before_requests_and_restored_on_reconstruction(
    pg, tmp_path, monkeypatch
):
    import pvdiag.agent.llm as module

    dsn, schema = pg
    _, request, _, _, _ = context(tmp_path, "F3")
    attempts = []
    current = {}

    class OfflineSDK:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            usage, _ = await current["tracker"].snapshot()
            assert usage["model_calls"] == len(attempts) + 1
            attempts.append(messages)
            return {
                "raw": AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                    response_metadata={"model_name": "offline-sdk-fixture"},
                ),
                "parsed": Plan(calls=[]),
                "parsing_error": None,
            }

    monkeypatch.setattr(module, "init_chat_model", lambda *args, **kwargs: OfflineSDK())

    async def verify():
        async with PostgresStore.open(dsn, schema=schema, initialize=True) as persistence:
            await persistence.create_run("accounting", request, {}, 2)
            tracker = current["tracker"] = persistence.tracker("accounting")
            first = LLMReasoner("offline", max_model_calls=2, call_tracker=tracker)
            await first._call(Plan, {}, "fixture")
        async with PostgresStore.open(dsn, schema=schema) as persistence:
            tracker = current["tracker"] = persistence.tracker("accounting")
            resumed = LLMReasoner("offline", max_model_calls=2, call_tracker=tracker)
            await resumed.sync_accounting()
            assert resumed.usage["model_calls"] == 1
            await resumed._call(Plan, {}, "fixture")
            with pytest.raises(ValueError, match="budget_exhausted"):
                await resumed._call(Plan, {}, "fixture")
            assert resumed.usage == {
                "model_calls": 2,
                "input_tokens": 14,
                "output_tokens": 6,
                "usage_missing_calls": 0,
            }
            assert resumed.metadata["reported_models"] == ["offline-sdk-fixture"]

    asyncio.run(verify())
    assert len(attempts) == 2


def test_suspended_llm_run_restores_usage_without_resetting_the_six_call_budget(
    pg, tmp_path, monkeypatch
):
    import pvdiag.agent.llm as module

    dsn, schema = pg
    store, request, _, _, _ = context(tmp_path, "F3")
    ledger = WorkOrderStore(tmp_path / "orders.sqlite3")
    attempts = []

    class StructuredOfflineSDK:
        def with_structured_output(self, schema, **kwargs):
            return self

        async def ainvoke(self, messages):
            payload = json.loads(messages[1][1])
            attempts.append(payload["task"])
            state = {
                "request": DiagRequest.model_validate(payload["request"]),
                "evidence": payload["evidence"],
            }
            if payload["task"] == "plan":
                tools = {item["tool"] for item in payload["evidence"].values()}
                state["loops"] = (
                    0 if "get_timeseries" not in tools else 1 if "peer_compare" not in tools else 2
                )
                state["scope"] = Scope.model_validate(payload["scope"])
                parsed = await RuleReasoner().plan(state, {}, "fixture")
            else:
                parsed = await RuleReasoner().hypothesize(state, "fixture")
            return {
                "raw": AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                    response_metadata={"model_name": "structured-offline-fixture"},
                ),
                "parsed": parsed,
                "parsing_error": None,
            }

    monkeypatch.setattr(module, "init_chat_model", lambda *args, **kwargs: StructuredOfflineSDK())

    async def verify():
        async with (
            PostgresStore.open(dsn, schema=schema, initialize=True) as persistence,
            open_runtime(
                store,
                LLMReasoner("offline", max_model_calls=6),
                persistence=persistence,
                ops_store=ledger,
                verification_mode="domain",
            ) as runtime,
        ):
            paused = await runtime.start("llm-approval", request)
            assert paused.status == "awaiting_approval" and len(attempts) == 6
        async with PostgresStore.open(dsn, schema=schema) as persistence:
            fresh = LLMReasoner("offline", max_model_calls=6)
            assert fresh.usage["model_calls"] == 0
            async with open_runtime(
                store, fresh, persistence=persistence, ops_store=ledger, verification_mode="domain"
            ) as runtime:
                proposal = paused.approval_request["proposal"]
                result = await runtime.resume(
                    "llm-approval",
                    {
                        "action_id": proposal["action_id"],
                        "payload_sha256": proposal["payload_sha256"],
                        "approved": True,
                        "reviewer": "controlled-test-reviewer",
                    },
                )
                assert result.result.action_outcome.status == "created"
                assert result.result.model_usage == {
                    "model_calls": 6,
                    "input_tokens": 42,
                    "output_tokens": 18,
                    "usage_missing_calls": 0,
                }
                assert result.result.model["reported_models"] == ["structured-offline-fixture"]
                assert len(attempts) == 6

    asyncio.run(verify())
